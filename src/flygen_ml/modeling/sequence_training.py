from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from flygen_ml.modeling.baselines import NON_FEATURE_COLUMNS
from flygen_ml.modeling.metrics import evidence_bin_for_n_segments, summarize_metrics
from flygen_ml.modeling.sequence_models import (
    FlySequenceExample,
    load_sequence_npz,
    predict_sequence_meanpool,
    serializable_sequence_model,
    train_sequence_meanpool,
)
from flygen_ml.modeling.splits import grouped_k_fold_splits, grouped_split
from flygen_ml.modeling.train import load_feature_rows, load_simple_yaml


def _example_rows(examples: list[FlySequenceExample], *, split_label_key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        row = {
            "fly_id": example.fly_id,
            "genotype": example.genotype,
            "cohort": example.cohort,
            "joint_label": f"{example.genotype}||{example.cohort}",
        }
        row["split_label"] = row[split_label_key]
        rows.append(row)
    return rows


def _examples_by_fly(examples: list[FlySequenceExample]) -> dict[str, FlySequenceExample]:
    return {example.fly_id: example for example in examples}


def _rows_to_examples(
    rows: list[dict[str, object]],
    examples_by_fly: dict[str, FlySequenceExample],
) -> list[FlySequenceExample]:
    return [examples_by_fly[str(row["fly_id"])] for row in rows]


def _parse_csv_list(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def _side_feature_names(feature_rows: list[dict[str, object]], config: dict[str, object]) -> list[str]:
    if "side_feature_names" in config:
        return _parse_csv_list(config["side_feature_names"])
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.update(_parse_csv_list(config.get("exclude_side_feature_names", "")))
    if not bool(config.get("include_side_evidence_features", False)):
        excluded.update({"n_segments", "n_segments_with_qc_flags"})
    first = feature_rows[0]
    return [
        key
        for key in first
        if key not in excluded
        and any(_numeric_value(row.get(key)) is not None for row in feature_rows)
    ]


def _load_side_inputs(
    config: dict[str, object],
    examples: list[FlySequenceExample],
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    side_features_path = str(config.get("side_features_path", "")).strip()
    if not side_features_path:
        return None, []
    feature_rows = load_feature_rows(side_features_path)
    feature_names = _side_feature_names(feature_rows, config)
    if not feature_names:
        raise ValueError("side_features_path was provided but no numeric side features were selected")
    rows_by_key = {
        (str(row["fly_id"]), str(row["sample_key"])): row
        for row in feature_rows
    }
    side_inputs: dict[str, np.ndarray] = {}
    for example in examples:
        key = (example.fly_id, example.sample_key)
        row = rows_by_key.get(key)
        if row is None:
            raise ValueError(f"missing side feature row for fly_id={example.fly_id!r}, sample_key={example.sample_key!r}")
        values = [_numeric_value(row.get(name)) for name in feature_names]
        side_inputs[example.fly_id] = np.asarray(
            [0.0 if value is None else value for value in values],
            dtype=np.float32,
        )
    return side_inputs, feature_names


def _sequence_metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    genotype_metrics = summarize_metrics(
        [str(row["actual_genotype"]) for row in predictions],
        [str(row["predicted_genotype"]) for row in predictions],
    )
    cohort_metrics = summarize_metrics(
        [str(row["actual_cohort"]) for row in predictions],
        [str(row["predicted_cohort"]) for row in predictions],
    )
    both_correct = sum(
        int(
            row["actual_genotype"] == row["predicted_genotype"]
            and row["actual_cohort"] == row["predicted_cohort"]
        )
        for row in predictions
    )
    return {
        "n_examples": len(predictions),
        "joint_accuracy": both_correct / len(predictions) if predictions else 0.0,
        "both_correct": both_correct,
        "genotype": genotype_metrics,
        "cohort": cohort_metrics,
    }


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_prediction_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "fold",
        "split",
        "fly_id",
        "sample_key",
        "actual_genotype",
        "predicted_genotype",
        "genotype_probability",
        "actual_cohort",
        "predicted_cohort",
        "cohort_probability",
        "both_correct",
        "n_segments",
        "n_segments_with_qc_flags",
        "evidence_bin",
    ]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _enrich_predictions(
    predictions: list[dict[str, object]],
    *,
    split: str,
    fold: int | None = None,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in predictions:
        enriched.append(
            {
                "fold": "" if fold is None else fold,
                "split": split,
                **row,
                "both_correct": (
                    row["actual_genotype"] == row["predicted_genotype"]
                    and row["actual_cohort"] == row["predicted_cohort"]
                ),
                "evidence_bin": evidence_bin_for_n_segments(row.get("n_segments")),
            }
        )
    return enriched


def _train_and_evaluate(
    x,
    train_examples: list[FlySequenceExample],
    valid_examples: list[FlySequenceExample],
    *,
    config: dict[str, object],
    side_inputs: dict[str, np.ndarray] | None = None,
    side_feature_names: list[str] | None = None,
) -> dict[str, object]:
    model_kind = str(config.get("model_kind", "sequence_meanpool_mlp_numpy_v1"))
    if model_kind == "sequence_conv1d_meanpool_torch_v1":
        from flygen_ml.modeling.torch_sequence_models import (
            predict_torch_sequence_meanpool,
            train_torch_sequence_meanpool,
        )

        model = train_torch_sequence_meanpool(
            x,
            train_examples,
            config=config,
            side_inputs=side_inputs,
            side_feature_names=side_feature_names,
        )
        train_predictions = predict_torch_sequence_meanpool(
            x,
            train_examples,
            model=model,
            side_inputs=side_inputs,
        )
        valid_predictions = predict_torch_sequence_meanpool(
            x,
            valid_examples,
            model=model,
            side_inputs=side_inputs,
        )
    elif model_kind in {"sequence_meanpool_mlp_numpy_v1", "sequence_meanpool"}:
        model = train_sequence_meanpool(x, train_examples, config=config)
        train_predictions = predict_sequence_meanpool(x, train_examples, model=model)
        valid_predictions = predict_sequence_meanpool(x, valid_examples, model=model)
    else:
        raise ValueError(f"unsupported sequence model_kind: {model_kind}")
    return {
        "model": model,
        "train_predictions": train_predictions,
        "valid_predictions": valid_predictions,
        "train_metrics": _sequence_metrics(train_predictions),
        "valid_metrics": _sequence_metrics(valid_predictions),
    }


def _model_training_summary(model: dict[str, object]) -> dict[str, object]:
    return {
        "model_kind": model.get("model_kind"),
        "hidden_dim": model.get("hidden_dim"),
        "conv_channels": model.get("conv_channels"),
        "embedding_dim": model.get("embedding_dim"),
        "fusion_hidden_dim": model.get("fusion_hidden_dim"),
        "pooling": model.get("pooling"),
        "attention_hidden_dim": model.get("attention_hidden_dim"),
        "n_side_features": model.get("n_side_features"),
        "side_feature_names": model.get("side_feature_names"),
        "dropout": model.get("dropout"),
        "learning_rate": model.get("learning_rate"),
        "weight_decay": model.get("weight_decay"),
        "max_iter": model.get("max_iter"),
        "device": model.get("device"),
        "train_max_segments_per_fly": model.get("train_max_segments_per_fly"),
        "eval_max_segments_per_fly": model.get("eval_max_segments_per_fly"),
        "scaler_max_segments_per_fly": model.get("scaler_max_segments_per_fly"),
        "segment_sampling": model.get("segment_sampling"),
    }


def train_and_save_sequence_run(
    *,
    config_path: str | Path,
    sequence_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    config = load_simple_yaml(config_path)
    x, examples, sequence_metadata = load_sequence_npz(str(sequence_path))
    side_inputs, side_feature_names = _load_side_inputs(config, examples)
    split_label_key = str(config.get("split_label_key", "genotype"))
    valid_fraction = float(config.get("valid_fraction", 0.25))
    random_seed = int(config.get("random_seed", 0))
    rows = _example_rows(examples, split_label_key=split_label_key)
    train_rows, valid_rows = grouped_split(
        rows,
        group_key="fly_id",
        label_key="split_label",
        random_seed=random_seed,
        valid_fraction=valid_fraction,
    )
    examples_by_fly = _examples_by_fly(examples)
    split_result = _train_and_evaluate(
        x,
        _rows_to_examples(train_rows, examples_by_fly),
        _rows_to_examples(valid_rows, examples_by_fly),
        config=config,
        side_inputs=side_inputs,
        side_feature_names=side_feature_names,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = dict(split_result["model"])
    if model.get("model_kind") == "sequence_conv1d_meanpool_torch_v1":
        from flygen_ml.modeling.torch_sequence_models import serializable_torch_sequence_model

        model_artifact = serializable_torch_sequence_model(model)
    else:
        model_artifact = serializable_sequence_model(model)
    model_kind = str(model["model_kind"])
    _write_json(out_dir / "model_artifact.json", model_artifact)
    _write_json(
        out_dir / "metrics_summary.json",
        {
            "model_kind": model_kind,
            "sequence_metadata": sequence_metadata,
            "training": _model_training_summary(model),
            "train": split_result["train_metrics"],
            "valid": split_result["valid_metrics"],
        },
    )
    predictions = _enrich_predictions(split_result["train_predictions"], split="train")
    predictions += _enrich_predictions(split_result["valid_predictions"], split="valid")
    _write_prediction_rows(out_dir / "predictions.csv", predictions)
    payload = {
        "status": "completed",
        "model_kind": model_kind,
        "evaluation_kind": "grouped_holdout",
        "config_path": str(config_path),
        "sequence_path": str(sequence_path),
        "split_label_key": split_label_key,
        **_model_training_summary(model),
        "train_flies": len(train_rows),
        "valid_flies": len(valid_rows),
        "metrics_summary_path": str(out_dir / "metrics_summary.json"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "model_artifact_path": str(out_dir / "model_artifact.json"),
    }
    _write_json(out_dir / "run_metadata.json", payload)
    return payload


def train_and_save_sequence_cross_validation_run(
    *,
    config_path: str | Path,
    sequence_path: str | Path,
    output_dir: str | Path,
    n_splits: int | None = None,
) -> dict[str, object]:
    config = load_simple_yaml(config_path)
    x, examples, sequence_metadata = load_sequence_npz(str(sequence_path))
    side_inputs, side_feature_names = _load_side_inputs(config, examples)
    split_label_key = str(config.get("split_label_key", "genotype"))
    random_seed = int(config.get("random_seed", 0))
    resolved_n_splits = int(n_splits or config.get("cv_folds", 5))
    rows = _example_rows(examples, split_label_key=split_label_key)
    folds = grouped_k_fold_splits(
        rows,
        group_key="fly_id",
        label_key="split_label",
        random_seed=random_seed,
        n_splits=resolved_n_splits,
    )
    examples_by_fly = _examples_by_fly(examples)
    fold_summaries: list[dict[str, object]] = []
    combined_predictions: list[dict[str, object]] = []
    training_summary: dict[str, object] | None = None
    model_kind = "sequence_meanpool_mlp_numpy_v1"
    for fold_idx, (train_rows, valid_rows) in enumerate(folds):
        split_result = _train_and_evaluate(
            x,
            _rows_to_examples(train_rows, examples_by_fly),
            _rows_to_examples(valid_rows, examples_by_fly),
            config={**config, "random_seed": random_seed + fold_idx},
            side_inputs=side_inputs,
            side_feature_names=side_feature_names,
        )
        model_kind = str(dict(split_result["model"])["model_kind"])
        training_summary = _model_training_summary(dict(split_result["model"]))
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train": split_result["train_metrics"],
                "valid": split_result["valid_metrics"],
                "train_flies": len(train_rows),
                "valid_flies": len(valid_rows),
                "train_groups": sorted(str(row["fly_id"]) for row in train_rows),
                "valid_groups": sorted(str(row["fly_id"]) for row in valid_rows),
            }
        )
        combined_predictions += _enrich_predictions(
            split_result["train_predictions"], split="train", fold=fold_idx
        )
        combined_predictions += _enrich_predictions(
            split_result["valid_predictions"], split="valid", fold=fold_idx
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "cv_metrics_summary.json",
        {
            "model_kind": model_kind,
            "sequence_metadata": sequence_metadata,
            "training": training_summary or {},
            "n_folds": resolved_n_splits,
            "folds": fold_summaries,
        },
    )
    _write_prediction_rows(out_dir / "cv_predictions.csv", combined_predictions)
    payload = {
        "status": "completed",
        "model_kind": model_kind,
        "evaluation_kind": "grouped_k_fold_cv",
        "config_path": str(config_path),
        "sequence_path": str(sequence_path),
        "split_label_key": split_label_key,
        **(training_summary or {}),
        "n_folds": resolved_n_splits,
        "flies": len(examples),
        "metrics_summary_path": str(out_dir / "cv_metrics_summary.json"),
        "predictions_path": str(out_dir / "cv_predictions.csv"),
    }
    _write_json(out_dir / "run_metadata.json", payload)
    return payload
