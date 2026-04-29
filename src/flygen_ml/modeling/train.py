from __future__ import annotations

import csv
import json
from pathlib import Path

from flygen_ml.modeling.baselines import predict_fly_level_baseline, train_fly_level_baseline
from flygen_ml.modeling.metrics import evidence_bin_for_n_segments, summarize_metrics, summarize_metrics_by_evidence_bin
from flygen_ml.modeling.splits import grouped_k_fold_splits, grouped_split


def _parse_scalar(raw_value: str) -> object:
    value = raw_value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_simple_yaml(path: str | Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"unsupported config line: {line!r}")
        key, raw_value = stripped.split(":", 1)
        payload[key.strip()] = _parse_scalar(raw_value)
    return payload


def load_feature_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = ""
                elif value == "":
                    parsed[key] = float("nan")
                else:
                    parsed[key] = _parse_scalar(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"feature table is empty: {path}")
    return rows


def write_run_metadata(output_dir: str | Path, payload: dict[str, object]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _prediction_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    fieldnames = [
        "split",
        "fly_id",
        "sample_key",
        "label_key",
        "actual_label",
        "predicted_label",
        "predicted_probability",
        "n_segments",
        "n_segments_with_qc_flags",
        "evidence_bin",
    ]
    if any("fold" in row for row in rows):
        fieldnames.insert(0, "fold")
    if any("actual_genotype" in row for row in rows):
        insert_at = fieldnames.index("predicted_probability")
        fieldnames[insert_at:insert_at] = ["actual_genotype", "predicted_genotype"]
    return fieldnames


def write_prediction_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _prediction_fieldnames(rows)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _with_evidence_bins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        enriched.append(
            {
                **row,
                "evidence_bin": evidence_bin_for_n_segments(row.get("n_segments")),
            }
        )
    return enriched


def _resolved_label_key(config: dict[str, object]) -> str:
    return str(config.get("label_key", config.get("target_key", "genotype")))


def _train_and_evaluate_split(
    train_rows: list[dict[str, object]],
    valid_rows: list[dict[str, object]],
    *,
    config: dict[str, object],
) -> dict[str, object]:
    label_key = _resolved_label_key(config)
    model = train_fly_level_baseline(train_rows, config=config)
    train_predictions = _with_evidence_bins(predict_fly_level_baseline(train_rows, model=model))
    valid_predictions = _with_evidence_bins(predict_fly_level_baseline(valid_rows, model=model))
    labels = [str(label) for label in model["labels"]]

    train_metrics = summarize_metrics(
        [str(row["actual_label"]) for row in train_predictions],
        [str(row["predicted_label"]) for row in train_predictions],
        labels=labels,
    )
    valid_metrics = summarize_metrics(
        [str(row["actual_label"]) for row in valid_predictions],
        [str(row["predicted_label"]) for row in valid_predictions],
        labels=labels,
    )
    return {
        "model": model,
        "label_key": label_key,
        "labels": labels,
        "train_predictions": train_predictions,
        "valid_predictions": valid_predictions,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "train_by_evidence_bin": summarize_metrics_by_evidence_bin(train_predictions, labels=labels),
        "valid_by_evidence_bin": summarize_metrics_by_evidence_bin(valid_predictions, labels=labels),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values),
        "max": max(values),
    }


def _summarize_metric_distributions(
    metric_payloads: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric_name in ("accuracy", "balanced_accuracy"):
        values = [float(payload[metric_name]) for payload in metric_payloads]
        summary[metric_name] = _distribution(values)

    label_names = sorted(
        {
            str(label)
            for payload in metric_payloads
            for label in dict(payload.get("label_recall", {})).keys()
        }
    )
    summary["label_recall"] = {
        label: _distribution(
            [
                float(dict(payload.get("label_recall", {}))[label])
                for payload in metric_payloads
                if label in dict(payload.get("label_recall", {}))
            ]
        )
        for label in label_names
    }
    summary["n_examples"] = _distribution([float(payload["n_examples"]) for payload in metric_payloads])
    return summary


def _summarize_fold_metrics(fold_summaries: list[dict[str, object]]) -> dict[str, object]:
    valid_payloads = [dict(fold["valid"]) for fold in fold_summaries]
    valid_by_bin: dict[str, list[dict[str, object]]] = {}
    for fold in fold_summaries:
        for evidence_bin, metrics in dict(fold["valid_by_evidence_bin"]).items():
            valid_by_bin.setdefault(str(evidence_bin), []).append(dict(metrics))
    return {
        "valid": _summarize_metric_distributions(valid_payloads),
        "valid_by_evidence_bin": {
            evidence_bin: _summarize_metric_distributions(payloads)
            for evidence_bin, payloads in sorted(valid_by_bin.items())
        },
    }


def train_and_save_run(
    *,
    config_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    config = load_simple_yaml(config_path)
    rows = load_feature_rows(features_path)
    group_key = str(config.get("group_key", "fly_id"))
    label_key = _resolved_label_key(config)
    valid_fraction = float(config.get("valid_fraction", 0.25))
    random_seed = int(config.get("random_seed", 0))

    train_rows, valid_rows = grouped_split(
        rows,
        group_key=group_key,
        label_key=label_key,
        random_seed=random_seed,
        valid_fraction=valid_fraction,
    )
    split_result = _train_and_evaluate_split(train_rows, valid_rows, config=config)
    model = dict(split_result["model"])
    train_predictions = list(split_result["train_predictions"])
    valid_predictions = list(split_result["valid_predictions"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model_artifact.json", model)
    write_json(
        out_dir / "metrics_summary.json",
        {
            "label_key": label_key,
            "train": split_result["train_metrics"],
            "valid": split_result["valid_metrics"],
            "train_by_evidence_bin": split_result["train_by_evidence_bin"],
            "valid_by_evidence_bin": split_result["valid_by_evidence_bin"],
        },
    )
    combined_predictions = (
        [{"split": "train", **row} for row in train_predictions]
        + [{"split": "valid", **row} for row in valid_predictions]
    )
    write_prediction_rows(out_dir / "predictions.csv", combined_predictions)

    payload = {
        "status": "completed",
        "config_path": str(config_path),
        "features_path": str(features_path),
        "group_key": group_key,
        "label_key": label_key,
        "model_name": config.get("model_name", "baseline"),
        "model_kind": model["model_kind"],
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "labels": model["labels"],
        "excluded_feature_names": model.get("excluded_feature_names", []),
        "metrics_summary_path": str(out_dir / "metrics_summary.json"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "model_artifact_path": str(out_dir / "model_artifact.json"),
    }
    write_run_metadata(out_dir, payload)
    return payload


def train_and_save_cross_validation_run(
    *,
    config_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path,
    n_splits: int | None = None,
) -> dict[str, object]:
    config = load_simple_yaml(config_path)
    rows = load_feature_rows(features_path)
    group_key = str(config.get("group_key", "fly_id"))
    label_key = _resolved_label_key(config)
    random_seed = int(config.get("random_seed", 0))
    resolved_n_splits = int(n_splits or config.get("cv_folds", 5))

    folds = grouped_k_fold_splits(
        rows,
        group_key=group_key,
        label_key=label_key,
        random_seed=random_seed,
        n_splits=resolved_n_splits,
    )

    fold_summaries: list[dict[str, object]] = []
    combined_predictions: list[dict[str, object]] = []
    labels: list[str] | None = None
    excluded_feature_names: list[str] = []
    model_kind = "baseline"
    for fold_idx, (train_rows, valid_rows) in enumerate(folds):
        split_result = _train_and_evaluate_split(train_rows, valid_rows, config=config)
        model = dict(split_result["model"])
        labels = [str(label) for label in model["labels"]]
        excluded_feature_names = [str(name) for name in model.get("excluded_feature_names", [])]
        model_kind = str(model["model_kind"])
        train_groups = sorted({str(row[group_key]) for row in train_rows})
        valid_groups = sorted({str(row[group_key]) for row in valid_rows})
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train": split_result["train_metrics"],
                "valid": split_result["valid_metrics"],
                "train_by_evidence_bin": split_result["train_by_evidence_bin"],
                "valid_by_evidence_bin": split_result["valid_by_evidence_bin"],
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "train_groups": train_groups,
                "valid_groups": valid_groups,
            }
        )
        combined_predictions.extend(
            {"fold": fold_idx, "split": "train", **row}
            for row in list(split_result["train_predictions"])
        )
        combined_predictions.extend(
            {"fold": fold_idx, "split": "valid", **row}
            for row in list(split_result["valid_predictions"])
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "cv_metrics_summary.json",
        {
            "label_key": label_key,
            "n_folds": resolved_n_splits,
            "folds": fold_summaries,
            "summary": _summarize_fold_metrics(fold_summaries),
        },
    )
    write_prediction_rows(out_dir / "cv_predictions.csv", combined_predictions)

    payload = {
        "status": "completed",
        "evaluation_kind": "grouped_stratified_k_fold_cv",
        "config_path": str(config_path),
        "features_path": str(features_path),
        "group_key": group_key,
        "label_key": label_key,
        "model_name": config.get("model_name", "baseline"),
        "model_kind": model_kind,
        "n_folds": resolved_n_splits,
        "rows": len(rows),
        "labels": labels or [],
        "excluded_feature_names": excluded_feature_names,
        "metrics_summary_path": str(out_dir / "cv_metrics_summary.json"),
        "predictions_path": str(out_dir / "cv_predictions.csv"),
    }
    write_run_metadata(out_dir, payload)
    return payload
