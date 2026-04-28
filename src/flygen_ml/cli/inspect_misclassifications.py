from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flygen_ml.modeling.inspection import (
    build_prediction_inspection_rows,
    load_json,
    load_prediction_rows,
    write_prediction_inspection_rows,
)
from flygen_ml.modeling.train import load_feature_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect misclassified flies from a saved model run.")
    parser.add_argument("--run-dir", required=True, help="Run output directory.")
    parser.add_argument(
        "--features",
        help="Feature table path. Defaults to the features_path recorded in run_metadata.json.",
    )
    parser.add_argument("--output", help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--split", default="valid", help="Split to inspect.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top contributors to include.")
    parser.add_argument("--include-correct", action="store_true", help="Include correctly classified rows too.")
    return parser


def _features_path_from_metadata(run_dir: Path) -> Path:
    metadata = load_json(run_dir / "run_metadata.json")
    features_path = metadata.get("features_path")
    if not isinstance(features_path, str) or not features_path:
        raise ValueError(f"run metadata is missing features_path: {run_dir / 'run_metadata.json'}")
    return Path(features_path)


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    features_path = Path(args.features) if args.features else _features_path_from_metadata(run_dir)
    rows = build_prediction_inspection_rows(
        predictions=load_prediction_rows(run_dir / "predictions.csv"),
        feature_rows=load_feature_rows(features_path),
        model=load_json(run_dir / "model_artifact.json"),
        split=args.split,
        include_correct=args.include_correct,
        top_n=args.top_n,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            write_prediction_inspection_rows(rows, handle)
        print(f"wrote {len(rows)} inspected rows to {output_path}")
    else:
        write_prediction_inspection_rows(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
