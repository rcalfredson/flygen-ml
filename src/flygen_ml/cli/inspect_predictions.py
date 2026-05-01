from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import TextIO

from flygen_ml.modeling.metrics import evidence_bin_for_n_segments
from flygen_ml.modeling.train import load_feature_rows


BASE_FIELDNAMES = [
    "fold",
    "split",
    "fly_id",
    "sample_key",
    "label_key",
    "actual_label",
    "predicted_label",
    "correct",
    "predicted_probability",
    "decision_margin",
    "evidence_bin",
    "n_segments",
    "n_segments_with_qc_flags",
]

METADATA_FIELDNAMES = [
    "genotype",
    "cohort",
    "chamber",
    "chamber_type",
    "training_idx",
    "date",
    "fly_idx",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export prediction rows joined to fly-level metadata/features.")
    parser.add_argument("--run-dir", required=True, help="Run output directory.")
    parser.add_argument(
        "--features",
        help="Feature table path. Defaults to features_path recorded in run_metadata.json.",
    )
    parser.add_argument("--output", help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--split", default="valid", help="Prediction split to export.")
    parser.add_argument("--errors-only", action="store_true", help="Export only misclassified rows.")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include all non-metadata feature columns from the feature table.",
    )
    return parser


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _features_path_from_metadata(run_dir: Path) -> Path:
    metadata = _load_json(run_dir / "run_metadata.json")
    features_path = metadata.get("features_path")
    if not isinstance(features_path, str) or not features_path:
        raise ValueError(f"run metadata is missing features_path: {run_dir / 'run_metadata.json'}")
    return Path(features_path)


def _prediction_path_for_run(run_dir: Path) -> Path:
    cv_path = run_dir / "cv_predictions.csv"
    holdout_path = run_dir / "predictions.csv"
    if cv_path.exists():
        return cv_path
    if holdout_path.exists():
        return holdout_path
    raise FileNotFoundError(f"missing prediction table: expected {cv_path} or {holdout_path}")


def _load_prediction_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"prediction table is empty: {path}")
    return rows


def _feature_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["fly_id"]), str(row["sample_key"])


def _feature_lookup(feature_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {_feature_key(row): row for row in feature_rows}


def _actual_label(row: dict[str, object]) -> str:
    if row.get("actual_label", "") != "":
        return str(row["actual_label"])
    return str(row["actual_genotype"])


def _predicted_label(row: dict[str, object]) -> str:
    if row.get("predicted_label", "") != "":
        return str(row["predicted_label"])
    return str(row["predicted_genotype"])


def _decision_margin(row: dict[str, object]) -> float | str:
    try:
        return abs(float(row["predicted_probability"]) - 0.5)
    except (KeyError, TypeError, ValueError):
        return ""


def _feature_fieldnames(feature_rows: list[dict[str, object]]) -> list[str]:
    if not feature_rows:
        return []
    excluded = set(BASE_FIELDNAMES) | set(METADATA_FIELDNAMES)
    return [
        key
        for key in feature_rows[0]
        if key not in excluded
        and key not in {"actual_genotype", "predicted_genotype"}
    ]


def build_prediction_review_rows(
    *,
    prediction_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    split: str = "valid",
    errors_only: bool = False,
    include_features: bool = False,
) -> list[dict[str, object]]:
    feature_rows_by_key = _feature_lookup(feature_rows)
    feature_names = _feature_fieldnames(feature_rows) if include_features else []
    review_rows: list[dict[str, object]] = []
    for prediction in prediction_rows:
        if str(prediction.get("split", "")) != split:
            continue
        actual_label = _actual_label(prediction)
        predicted_label = _predicted_label(prediction)
        correct = actual_label == predicted_label
        if errors_only and correct:
            continue

        key = _feature_key(prediction)
        feature_row = feature_rows_by_key.get(key)
        if feature_row is None:
            raise ValueError(f"missing feature row for prediction: fly_id={key[0]!r}, sample_key={key[1]!r}")

        row: dict[str, object] = {
            "fold": prediction.get("fold", ""),
            "split": prediction.get("split", ""),
            "fly_id": prediction["fly_id"],
            "sample_key": prediction["sample_key"],
            "label_key": prediction.get("label_key", ""),
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "correct": correct,
            "predicted_probability": prediction.get("predicted_probability", ""),
            "decision_margin": _decision_margin(prediction),
            "evidence_bin": prediction.get("evidence_bin", "")
            or evidence_bin_for_n_segments(feature_row.get("n_segments")),
            "n_segments": prediction.get("n_segments", "") or feature_row.get("n_segments", ""),
            "n_segments_with_qc_flags": prediction.get("n_segments_with_qc_flags", "")
            or feature_row.get("n_segments_with_qc_flags", ""),
        }
        for fieldname in METADATA_FIELDNAMES:
            row[fieldname] = feature_row.get(fieldname, "")
        for feature_name in feature_names:
            row[feature_name] = feature_row.get(feature_name, "")
        review_rows.append(row)
    return review_rows


def write_prediction_review_rows(
    rows: list[dict[str, object]],
    handle: TextIO,
    *,
    include_features: bool = False,
) -> None:
    fieldnames = BASE_FIELDNAMES + METADATA_FIELDNAMES
    if include_features and rows:
        fieldnames.extend(name for name in rows[0] if name not in fieldnames)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    features_path = Path(args.features) if args.features else _features_path_from_metadata(run_dir)
    rows = build_prediction_review_rows(
        prediction_rows=_load_prediction_rows(_prediction_path_for_run(run_dir)),
        feature_rows=load_feature_rows(features_path),
        split=args.split,
        errors_only=args.errors_only,
        include_features=args.include_features,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            write_prediction_review_rows(rows, handle, include_features=args.include_features)
        print(f"wrote {len(rows)} prediction review rows to {output_path}")
    else:
        write_prediction_review_rows(rows, handle=sys.stdout, include_features=args.include_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
