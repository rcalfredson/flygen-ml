from __future__ import annotations

import csv
import json
from pathlib import Path

from flygen_ml.modeling.baselines import predict_fly_level_baseline, train_fly_level_baseline
from flygen_ml.modeling.metrics import summarize_metrics
from flygen_ml.modeling.splits import grouped_split


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


def write_prediction_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "fly_id",
        "sample_key",
        "actual_genotype",
        "predicted_genotype",
        "predicted_probability",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_and_save_run(
    *,
    config_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    config = load_simple_yaml(config_path)
    rows = load_feature_rows(features_path)
    group_key = str(config.get("group_key", "fly_id"))
    valid_fraction = float(config.get("valid_fraction", 0.25))
    random_seed = int(config.get("random_seed", 0))

    train_rows, valid_rows = grouped_split(
        rows,
        group_key=group_key,
        random_seed=random_seed,
        valid_fraction=valid_fraction,
    )
    model = train_fly_level_baseline(train_rows, config=config)
    train_predictions = predict_fly_level_baseline(train_rows, model=model)
    valid_predictions = predict_fly_level_baseline(valid_rows, model=model)

    train_metrics = summarize_metrics(
        [str(row["actual_genotype"]) for row in train_predictions],
        [str(row["predicted_genotype"]) for row in train_predictions],
        labels=[str(label) for label in model["labels"]],
    )
    valid_metrics = summarize_metrics(
        [str(row["actual_genotype"]) for row in valid_predictions],
        [str(row["predicted_genotype"]) for row in valid_predictions],
        labels=[str(label) for label in model["labels"]],
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model_artifact.json", model)
    write_json(
        out_dir / "metrics_summary.json",
        {
            "train": train_metrics,
            "valid": valid_metrics,
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
        "model_name": config.get("model_name", "baseline"),
        "model_kind": model["model_kind"],
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "labels": model["labels"],
        "metrics_summary_path": str(out_dir / "metrics_summary.json"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "model_artifact_path": str(out_dir / "model_artifact.json"),
    }
    write_run_metadata(out_dir, payload)
    return payload
