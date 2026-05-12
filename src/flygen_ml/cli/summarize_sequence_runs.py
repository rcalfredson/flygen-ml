from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


METRIC_KEYS = (
    "valid_joint_accuracy",
    "valid_genotype_accuracy",
    "valid_genotype_balanced_accuracy",
    "valid_cohort_accuracy",
    "valid_cohort_balanced_accuracy",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _format_metric(value: object) -> str:
    return f"{float(value):.3f}"


def _axis_accuracy(split: dict[str, object], axis: str) -> float:
    return float(dict(split[axis])["accuracy"])


def _axis_balanced_accuracy(split: dict[str, object], axis: str) -> float:
    return float(dict(split[axis])["balanced_accuracy"])


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _metric_means_for_run(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "cv_metrics_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing sequence CV metrics summary: {metrics_path}")
    metrics = _load_json(metrics_path)
    folds = [dict(fold) for fold in list(metrics["folds"])]
    if not folds:
        raise ValueError(f"sequence CV metrics summary has no folds: {metrics_path}")
    valid_splits = [dict(fold["valid"]) for fold in folds]
    run_metadata_path = run_dir / "run_metadata.json"
    run_metadata = _load_json(run_metadata_path) if run_metadata_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "model_kind": str(metrics.get("model_kind", run_metadata.get("model_kind", ""))),
        "random_seed": int(metrics.get("random_seed", run_metadata.get("random_seed", 0))),
        "n_folds": int(metrics["n_folds"]),
        "valid_joint_accuracy": mean([float(split["joint_accuracy"]) for split in valid_splits]),
        "valid_genotype_accuracy": mean([_axis_accuracy(split, "genotype") for split in valid_splits]),
        "valid_genotype_balanced_accuracy": mean(
            [_axis_balanced_accuracy(split, "genotype") for split in valid_splits]
        ),
        "valid_cohort_accuracy": mean([_axis_accuracy(split, "cohort") for split in valid_splits]),
        "valid_cohort_balanced_accuracy": mean(
            [_axis_balanced_accuracy(split, "cohort") for split in valid_splits]
        ),
        "fold_valid_n_examples": [int(dict(fold["valid"])["n_examples"]) for fold in folds],
    }


def summarize_runs(run_dirs: list[Path]) -> dict[str, object]:
    if not run_dirs:
        raise ValueError("at least one --run-dir is required")
    runs = [_metric_means_for_run(run_dir) for run_dir in run_dirs]
    return {
        "n_runs": len(runs),
        "runs": runs,
        "summary": {
            key: _distribution([float(run[key]) for run in runs])
            for key in METRIC_KEYS
        },
    }


def _print_summary(payload: dict[str, object]) -> None:
    runs = [dict(run) for run in list(payload["runs"])]
    print(f"n_runs: {payload['n_runs']}")
    print()
    for idx, run in enumerate(runs):
        print(
            f"run {idx}: "
            f"seed={run['random_seed']} "
            f"valid_joint_accuracy={_format_metric(run['valid_joint_accuracy'])} "
            f"valid_genotype_accuracy={_format_metric(run['valid_genotype_accuracy'])} "
            f"valid_genotype_balanced_accuracy={_format_metric(run['valid_genotype_balanced_accuracy'])} "
            f"valid_cohort_accuracy={_format_metric(run['valid_cohort_accuracy'])} "
            f"valid_cohort_balanced_accuracy={_format_metric(run['valid_cohort_balanced_accuracy'])} "
            f"run_dir={run['run_dir']}"
        )
    print()
    summary = dict(payload["summary"])
    for key in METRIC_KEYS:
        distribution = dict(summary[key])
        values = [
            float(distribution["mean"]),
            float(distribution["std"]),
            float(distribution["min"]),
            float(distribution["max"]),
        ]
        print(
            f"{key}: "
            f"mean={_format_metric(values[0])}, "
            f"std={_format_metric(values[1])}, "
            f"min={_format_metric(values[2])}, "
            f"max={_format_metric(values[3])}, "
            f"n={int(distribution['n'])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize sequence CV metrics across multiple run directories.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Sequence CV run directory. May be supplied multiple times.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a compact text summary.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = summarize_runs([Path(path) for path in args.run_dir])
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
