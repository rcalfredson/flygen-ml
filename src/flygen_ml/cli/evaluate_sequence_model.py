from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _metrics_path_for_run(run_dir: Path) -> tuple[str, Path]:
    holdout_path = run_dir / "metrics_summary.json"
    cv_path = run_dir / "cv_metrics_summary.json"
    if holdout_path.exists():
        return "holdout", holdout_path
    if cv_path.exists():
        return "cv", cv_path
    raise FileNotFoundError(f"missing sequence metrics summary: expected {holdout_path} or {cv_path}")


def _prediction_path_for_run(run_dir: Path, run_kind: str) -> Path:
    filename = "cv_predictions.csv" if run_kind == "cv" else "predictions.csv"
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing sequence prediction table: {path}")
    return path


def _format_metric(value: object) -> str:
    return f"{float(value):.3f}"


def _axis_accuracy(split: dict[str, object], axis: str) -> float:
    return float(dict(split[axis])["accuracy"])


def _axis_balanced_accuracy(split: dict[str, object], axis: str) -> float:
    return float(dict(split[axis])["balanced_accuracy"])


def _print_split_summary(split_name: str, split: dict[str, object]) -> None:
    print(
        f"{split_name}: "
        f"joint_accuracy={_format_metric(split['joint_accuracy'])}, "
        f"genotype_accuracy={_format_metric(_axis_accuracy(split, 'genotype'))}, "
        f"genotype_balanced_accuracy={_format_metric(_axis_balanced_accuracy(split, 'genotype'))}, "
        f"cohort_accuracy={_format_metric(_axis_accuracy(split, 'cohort'))}, "
        f"cohort_balanced_accuracy={_format_metric(_axis_balanced_accuracy(split, 'cohort'))}, "
        f"n={int(split['n_examples'])}"
    )


def _print_holdout_summary(metrics: dict[str, object]) -> None:
    print("run_type: sequence_holdout")
    print(f"model_kind: {metrics.get('model_kind', 'sequence_meanpool_mlp_numpy_v1')}")
    _print_split_summary("train", dict(metrics["train"]))
    _print_split_summary("valid", dict(metrics["valid"]))


def _format_distribution(values: list[float]) -> str:
    return (
        f"mean={_format_metric(mean(values))}, "
        f"std={_format_metric(pstdev(values))}, "
        f"min={_format_metric(min(values))}, "
        f"max={_format_metric(max(values))}, "
        f"n={len(values)}"
    )


def _print_cv_summary(metrics: dict[str, object]) -> None:
    print("run_type: sequence_grouped_cv")
    print(f"model_kind: {metrics.get('model_kind', 'sequence_meanpool_mlp_numpy_v1')}")
    print(f"n_folds: {metrics['n_folds']}")
    print()
    folds = [dict(fold) for fold in list(metrics["folds"])]
    for fold in folds:
        valid = dict(fold["valid"])
        print(
            f"fold {fold['fold']}: "
            f"valid_joint_accuracy={_format_metric(valid['joint_accuracy'])}, "
            f"valid_genotype_accuracy={_format_metric(_axis_accuracy(valid, 'genotype'))}, "
            f"valid_cohort_accuracy={_format_metric(_axis_accuracy(valid, 'cohort'))}, "
            f"n={int(valid['n_examples'])}"
        )
    print()
    valid_splits = [dict(fold["valid"]) for fold in folds]
    print(f"valid_joint_accuracy: {_format_distribution([float(split['joint_accuracy']) for split in valid_splits])}")
    print(
        "valid_genotype_accuracy: "
        f"{_format_distribution([_axis_accuracy(split, 'genotype') for split in valid_splits])}"
    )
    print(
        "valid_genotype_balanced_accuracy: "
        f"{_format_distribution([_axis_balanced_accuracy(split, 'genotype') for split in valid_splits])}"
    )
    print(
        "valid_cohort_accuracy: "
        f"{_format_distribution([_axis_accuracy(split, 'cohort') for split in valid_splits])}"
    )
    print(
        "valid_cohort_balanced_accuracy: "
        f"{_format_distribution([_axis_balanced_accuracy(split, 'cohort') for split in valid_splits])}"
    )


def _load_prediction_rows(path: Path, *, split: str) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split", "") == split]
    if not rows:
        raise ValueError(f"sequence prediction table has no rows for split {split!r}: {path}")
    return rows


def _actual_label(row: dict[str, str], axis: str) -> str:
    return row[f"actual_{axis}"]


def _predicted_label(row: dict[str, str], axis: str) -> str:
    return row[f"predicted_{axis}"]


def _print_confusion(rows: list[dict[str, str]], *, axis: str) -> None:
    labels = sorted(
        {_actual_label(row, axis) for row in rows}
        | {_predicted_label(row, axis) for row in rows}
    )
    counts = Counter((_actual_label(row, axis), _predicted_label(row, axis)) for row in rows)
    print(f"{axis}_confusion_matrix:")
    print("actual\\predicted\t" + "\t".join(labels))
    for actual in labels:
        values = [str(counts[(actual, predicted)]) for predicted in labels]
        print(f"{actual}\t" + "\t".join(values))


def _print_joint_confusion(rows: list[dict[str, str]]) -> None:
    labels = sorted(
        {
            f"{row['actual_genotype']}|{row['actual_cohort']}"
            for row in rows
        }
        | {
            f"{row['predicted_genotype']}|{row['predicted_cohort']}"
            for row in rows
        }
    )
    counts = Counter(
        (
            f"{row['actual_genotype']}|{row['actual_cohort']}",
            f"{row['predicted_genotype']}|{row['predicted_cohort']}",
        )
        for row in rows
    )
    print("joint_confusion_matrix:")
    print("actual\\predicted\t" + "\t".join(labels))
    for actual in labels:
        values = [str(counts[(actual, predicted)]) for predicted in labels]
        print(f"{actual}\t" + "\t".join(values))


def _print_misclassifications(rows: list[dict[str, str]]) -> None:
    misses = [
        row
        for row in rows
        if row["actual_genotype"] != row["predicted_genotype"]
        or row["actual_cohort"] != row["predicted_cohort"]
    ]
    print(f"misclassifications: {len(misses)}")
    for row in misses:
        prefix = f"fold={row['fold']} " if row.get("fold") else ""
        print(
            f"{prefix}"
            f"actual={row['actual_genotype']}|{row['actual_cohort']} "
            f"predicted={row['predicted_genotype']}|{row['predicted_cohort']} "
            f"genotype_prob={float(row['genotype_probability']):.3f} "
            f"cohort_prob={float(row['cohort_probability']):.3f} "
            f"evidence_bin={row.get('evidence_bin', '')} "
            f"fly_id={row['fly_id']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved sequence model run.")
    parser.add_argument("--run-dir", required=True, help="Sequence run directory path.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw metrics JSON instead of a compact text summary.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        help="Prediction split to use for confusion and misclassification summaries.",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Print joint, genotype, and cohort confusion matrices from the prediction table.",
    )
    parser.add_argument(
        "--misclassifications",
        action="store_true",
        help="Print rows where either sequence output head is wrong.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    run_kind, metrics_path = _metrics_path_for_run(run_dir)
    metrics = _load_json(metrics_path)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    elif run_kind == "cv":
        _print_cv_summary(metrics)
    else:
        _print_holdout_summary(metrics)

    if args.confusion or args.misclassifications:
        prediction_rows = _load_prediction_rows(
            _prediction_path_for_run(run_dir, run_kind),
            split=args.split,
        )
        if not args.json:
            print()
        print(f"prediction_split: {args.split}")
        if args.confusion:
            _print_joint_confusion(prediction_rows)
            print()
            _print_confusion(prediction_rows, axis="genotype")
            print()
            _print_confusion(prediction_rows, axis="cohort")
        if args.confusion and args.misclassifications:
            print()
        if args.misclassifications:
            _print_misclassifications(prediction_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
