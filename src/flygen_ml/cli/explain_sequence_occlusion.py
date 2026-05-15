from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from flygen_ml.modeling.sequence_models import FlySequenceExample, load_sequence_npz
from flygen_ml.modeling.sequence_training import _load_side_inputs
from flygen_ml.modeling.train import load_simple_yaml
from flygen_ml.modeling.torch_sequence_models import (
    explain_torch_sequence_segment_occlusion,
    load_torch_sequence_model_artifact,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _prediction_fly_ids(run_dir: Path, split: str) -> set[str]:
    if split == "all":
        return set()
    prediction_path = run_dir / "predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"missing holdout prediction table for split filtering: {prediction_path}"
        )
    with prediction_path.open("r", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if not rows:
        raise ValueError(f"prediction table has no rows for split {split!r}: {prediction_path}")
    return {row["fly_id"] for row in rows}


def _filter_examples(examples: list[FlySequenceExample], fly_ids: set[str]) -> list[FlySequenceExample]:
    if not fly_ids:
        return examples
    return [example for example in examples if example.fly_id in fly_ids]


def _side_input_config(
    *,
    run_dir: Path,
    model: dict[str, object],
    side_features_path: str | None,
) -> dict[str, object]:
    feature_names = [str(name) for name in model.get("side_feature_names", [])]
    if not feature_names:
        return {}
    if side_features_path:
        return {
            "side_features_path": side_features_path,
            "side_feature_names": ",".join(feature_names),
            "include_side_evidence_features": True,
        }
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "this model uses side features, but run_metadata.json is missing; "
            "pass --side-features-path explicitly"
        )
    metadata = _load_json(metadata_path)
    config_path = Path(str(metadata.get("config_path", "")))
    if not config_path.exists():
        raise FileNotFoundError(
            "this model uses side features, but the saved config path is missing; "
            "pass --side-features-path explicitly"
        )
    config = load_simple_yaml(config_path)
    return {
        **config,
        "side_feature_names": ",".join(feature_names),
    }


def _segment_ids_by_index(sequence_path: str | Path) -> dict[int, str]:
    payload = np.load(sequence_path)
    return {
        idx: str(segment_id)
        for idx, segment_id in enumerate(payload["segment_id"])
    }


def _write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = [
            "split",
            "fly_id",
            "sample_key",
            "segment_id",
            "segment_index",
            "occlusion_status",
        ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explain a saved holdout torch sequence model with leave-one-segment-out occlusion.",
    )
    parser.add_argument("--run-dir", required=True, help="Saved holdout sequence run directory.")
    parser.add_argument("--sequence-path", required=True, help="Input .npz sequence tensor artifact.")
    parser.add_argument("--output-csv", required=True, help="Path for the segment occlusion CSV.")
    parser.add_argument(
        "--split",
        default="valid",
        choices=("train", "valid", "all"),
        help="Prediction split to explain. Use 'all' to explain every fly in the sequence artifact.",
    )
    parser.add_argument(
        "--side-features-path",
        default=None,
        help="Optional side-feature CSV override for fused models.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override. Defaults to the device saved in the model artifact.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    model_path = run_dir / "model_artifact.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing model_artifact.json: {model_path}. "
            "Initial occlusion support expects a saved holdout run, not a CV-only run."
        )
    model_artifact = _load_json(model_path)
    model = load_torch_sequence_model_artifact(model_artifact, device=args.device)
    x, examples, _ = load_sequence_npz(args.sequence_path)
    fly_ids = _prediction_fly_ids(run_dir, args.split)
    selected_examples = _filter_examples(examples, fly_ids)
    side_config = _side_input_config(
        run_dir=run_dir,
        model=model,
        side_features_path=args.side_features_path,
    )
    side_inputs = None
    if side_config:
        side_inputs, _ = _load_side_inputs(side_config, examples)
    segment_ids_by_index = _segment_ids_by_index(args.sequence_path)
    rows = explain_torch_sequence_segment_occlusion(
        x,
        selected_examples,
        model=model,
        side_inputs=side_inputs,
    )
    enriched_rows = [
        {
            "split": args.split,
            "segment_id": segment_ids_by_index[int(row["segment_index"])],
            **row,
        }
        for row in rows
    ]
    _write_rows(args.output_csv, enriched_rows)
    print(
        f"wrote {len(enriched_rows)} segment occlusion rows "
        f"for {len(selected_examples)} flies to {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
