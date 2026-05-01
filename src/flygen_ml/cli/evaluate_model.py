from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _metrics_path_for_run(run_dir: Path) -> tuple[str, Path]:
    holdout_path = run_dir / "metrics_summary.json"
    cv_path = run_dir / "cv_metrics_summary.json"
    if holdout_path.exists():
        return "holdout", holdout_path
    if cv_path.exists():
        return "cv", cv_path
    raise FileNotFoundError(
        f"missing metrics summary: expected {holdout_path} or {cv_path}"
    )


def _prediction_path_for_run(run_dir: Path, run_kind: str) -> Path:
    filename = "cv_predictions.csv" if run_kind == "cv" else "predictions.csv"
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing prediction table: {path}")
    return path


def _format_metric(value: object) -> str:
    return f"{float(value):.3f}"


def _format_distribution(payload: object) -> str:
    values = dict(payload)
    return (
        f"mean={_format_metric(values['mean'])}, "
        f"std={_format_metric(values['std'])}, "
        f"min={_format_metric(values['min'])}, "
        f"max={_format_metric(values['max'])}, "
        f"n={int(values['n'])}"
    )


def _format_label_recalls(payload: object) -> str:
    recalls = dict(payload)
    return ", ".join(
        f"{label}={_format_metric(value)}"
        for label, value in sorted(recalls.items())
    )


def _print_holdout_summary(metrics: dict[str, object]) -> None:
    print("run_type: holdout")
    print(f"label_key: {metrics.get('label_key', 'genotype')}")
    for split_name in ("train", "valid"):
        split = dict(metrics[split_name])
        print(
            f"{split_name}: "
            f"accuracy={_format_metric(split['accuracy'])}, "
            f"balanced_accuracy={_format_metric(split['balanced_accuracy'])}, "
            f"n={int(split['n_examples'])}, "
            f"recall=({_format_label_recalls(split['label_recall'])})"
        )


def _print_cv_summary(metrics: dict[str, object]) -> None:
    print("run_type: grouped_cv")
    print(f"label_key: {metrics.get('label_key', 'genotype')}")
    print(f"n_folds: {metrics['n_folds']}")
    print()
    for fold in list(metrics["folds"]):
        fold_payload = dict(fold)
        valid = dict(fold_payload["valid"])
        print(
            f"fold {fold_payload['fold']}: "
            f"valid_accuracy={_format_metric(valid['accuracy'])}, "
            f"valid_balanced_accuracy={_format_metric(valid['balanced_accuracy'])}, "
            f"n={int(valid['n_examples'])}, "
            f"recall=({_format_label_recalls(valid['label_recall'])})"
        )
    print()
    summary = dict(dict(metrics["summary"])["valid"])
    print(f"valid_accuracy: {_format_distribution(summary['accuracy'])}")
    print(f"valid_balanced_accuracy: {_format_distribution(summary['balanced_accuracy'])}")
    label_recall = dict(summary.get("label_recall", {}))
    for label, distribution in sorted(label_recall.items()):
        print(f"valid_recall[{label}]: {_format_distribution(distribution)}")


def _load_prediction_rows(path: Path, *, split: str) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split", "") == split
        ]
    if not rows:
        raise ValueError(f"prediction table has no rows for split {split!r}: {path}")
    return rows


def _actual_label(row: dict[str, str]) -> str:
    return row.get("actual_label") or row["actual_genotype"]


def _predicted_label(row: dict[str, str]) -> str:
    return row.get("predicted_label") or row["predicted_genotype"]


def _labels_from_rows(rows: Iterable[dict[str, str]]) -> list[str]:
    labels: set[str] = set()
    for row in rows:
        labels.add(_actual_label(row))
        labels.add(_predicted_label(row))
    return sorted(labels)


def _print_confusion(rows: list[dict[str, str]]) -> None:
    labels = _labels_from_rows(rows)
    counts = Counter((_actual_label(row), _predicted_label(row)) for row in rows)
    print("confusion_matrix:")
    print("actual\\predicted\t" + "\t".join(labels))
    for actual in labels:
        values = [str(counts[(actual, predicted)]) for predicted in labels]
        print(f"{actual}\t" + "\t".join(values))


def _print_misclassifications(rows: list[dict[str, str]]) -> None:
    misses = [
        row
        for row in rows
        if _actual_label(row) != _predicted_label(row)
    ]
    print(f"misclassifications: {len(misses)}")
    for row in misses:
        prefix = f"fold={row['fold']} " if "fold" in row else ""
        print(
            f"{prefix}"
            f"actual={_actual_label(row)} "
            f"predicted={_predicted_label(row)} "
            f"prob={float(row['predicted_probability']):.3f} "
            f"evidence_bin={row.get('evidence_bin', '')} "
            f"fly_id={row['fly_id']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved model run.")
    parser.add_argument("--run-dir", required=True, help="Run directory path.")
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
        help="Print a confusion matrix from the run's prediction table.",
    )
    parser.add_argument(
        "--misclassifications",
        action="store_true",
        help="Print misclassified prediction rows from the run's prediction table.",
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
            _print_confusion(prediction_rows)
        if args.confusion and args.misclassifications:
            print()
        if args.misclassifications:
            _print_misclassifications(prediction_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
