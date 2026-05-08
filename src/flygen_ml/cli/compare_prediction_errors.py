from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _prediction_path_for_input(path: str | Path) -> Path:
    input_path = Path(path)
    if input_path.is_dir():
        cv_path = input_path / "cv_predictions.csv"
        holdout_path = input_path / "predictions.csv"
        if cv_path.exists():
            return cv_path
        if holdout_path.exists():
            return holdout_path
        raise FileNotFoundError(f"missing prediction table in run directory: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"missing prediction table: {input_path}")
    return input_path


def _actual_label(row: dict[str, str], *, axis: str) -> str:
    axis_key = f"actual_{axis}"
    if axis_key in row and row[axis_key] != "":
        return row[axis_key]
    if "actual_label" in row and row["actual_label"] != "":
        return row["actual_label"]
    raise ValueError(f"prediction row is missing actual label for axis {axis!r}")


def _predicted_label(row: dict[str, str], *, axis: str) -> str:
    axis_key = f"predicted_{axis}"
    if axis_key in row and row[axis_key] != "":
        return row[axis_key]
    if "predicted_label" in row and row["predicted_label"] != "":
        return row["predicted_label"]
    raise ValueError(f"prediction row is missing predicted label for axis {axis!r}")


def _predicted_probability(row: dict[str, str], *, axis: str) -> str:
    axis_key = f"{axis}_probability"
    if axis_key in row and row[axis_key] != "":
        return row[axis_key]
    return row.get("predicted_probability", "")


def _load_rows(path: Path, *, axis: str, split: str | None) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        row_split = row.get("split", "")
        if split is not None and row_split != split:
            continue
        actual = _actual_label(row, axis=axis)
        predicted = _predicted_label(row, axis=axis)
        normalized_rows.append(
            {
                "fly_id": row["fly_id"],
                "sample_key": row.get("sample_key", ""),
                "split": row_split,
                "fold": row.get("fold", ""),
                "actual_label": actual,
                "predicted_label": predicted,
                "predicted_probability": _predicted_probability(row, axis=axis),
                "correct": str(actual == predicted),
                "n_segments": row.get("n_segments", ""),
                "evidence_bin": row.get("evidence_bin", ""),
            }
        )
    if not normalized_rows:
        raise ValueError(f"prediction table has no rows for split {split!r}: {path}")
    return normalized_rows


def _join_key(row: dict[str, str], *, include_fold: bool) -> tuple[str, ...]:
    key = (row["fly_id"], row["sample_key"], row["split"])
    if include_fold:
        return (*key, row.get("fold", ""))
    return key


def _index_rows(
    rows: list[dict[str, str]],
    *,
    include_fold: bool,
    run_name: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = _join_key(row, include_fold=include_fold)
        if key in indexed:
            raise ValueError(
                f"duplicate join keys in {run_name}: {key}. "
                "Use aligned CV folds or compare a split where each fly appears once."
            )
        indexed[key] = row
    return indexed


def _case_for(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "both_correct"
    if a_correct:
        return "run_a_only_correct"
    if b_correct:
        return "run_b_only_correct"
    return "both_wrong"


def _joined_rows(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    *,
    include_fold: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    indexed_a = _index_rows(rows_a, include_fold=include_fold, run_name="run_a")
    indexed_b = _index_rows(rows_b, include_fold=include_fold, run_name="run_b")
    shared_keys = sorted(set(indexed_a) & set(indexed_b))
    if not shared_keys:
        raise ValueError("prediction tables have no shared join keys")
    joined: list[dict[str, object]] = []
    for key in shared_keys:
        row_a = indexed_a[key]
        row_b = indexed_b[key]
        if row_a["actual_label"] != row_b["actual_label"]:
            raise ValueError(
                "joined predictions disagree on actual labels for "
                f"fly_id={row_a['fly_id']!r}: {row_a['actual_label']!r} vs {row_b['actual_label']!r}"
            )
        a_correct = row_a["correct"] == "True"
        b_correct = row_b["correct"] == "True"
        joined.append(
            {
                "fly_id": row_a["fly_id"],
                "sample_key": row_a["sample_key"],
                "split": row_a["split"],
                "fold": row_a.get("fold", ""),
                "actual_label": row_a["actual_label"],
                "run_a_predicted_label": row_a["predicted_label"],
                "run_a_predicted_probability": row_a["predicted_probability"],
                "run_a_correct": a_correct,
                "run_b_predicted_label": row_b["predicted_label"],
                "run_b_predicted_probability": row_b["predicted_probability"],
                "run_b_correct": b_correct,
                "correctness_case": _case_for(a_correct, b_correct),
                "n_segments": row_a.get("n_segments") or row_b.get("n_segments", ""),
                "evidence_bin": row_a.get("evidence_bin") or row_b.get("evidence_bin", ""),
            }
        )
    key_columns = ["fly_id", "sample_key", "split"]
    if include_fold:
        key_columns.append("fold")
    return joined, key_columns


def _summarize(joined: list[dict[str, object]]) -> dict[str, object]:
    n_joined = len(joined)
    run_a_correct = sum(row["run_a_correct"] is True for row in joined)
    run_b_correct = sum(row["run_b_correct"] is True for row in joined)
    counts = {
        "both_correct": 0,
        "run_a_only_correct": 0,
        "run_b_only_correct": 0,
        "both_wrong": 0,
    }
    for row in joined:
        counts[str(row["correctness_case"])] += 1
    return {
        "n_joined_examples": n_joined,
        "run_a_accuracy": run_a_correct / n_joined,
        "run_b_accuracy": run_b_correct / n_joined,
        **counts,
    }


def _format_metric(value: object) -> str:
    return f"{float(value):.3f}"


def _print_summary(summary: dict[str, object], *, run_a_name: str, run_b_name: str) -> None:
    print(f"n_joined_examples: {int(summary['n_joined_examples'])}")
    print(f"{run_a_name}_accuracy: {_format_metric(summary['run_a_accuracy'])}")
    print(f"{run_b_name}_accuracy: {_format_metric(summary['run_b_accuracy'])}")
    print(
        "correctness_counts: "
        f"both_correct={int(summary['both_correct'])}, "
        f"{run_a_name}_only_correct={int(summary['run_a_only_correct'])}, "
        f"{run_b_name}_only_correct={int(summary['run_b_only_correct'])}, "
        f"both_wrong={int(summary['both_wrong'])}"
    )


def _write_joined_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fly_id",
        "sample_key",
        "split",
        "fold",
        "actual_label",
        "run_a_predicted_label",
        "run_a_predicted_probability",
        "run_a_correct",
        "run_b_predicted_label",
        "run_b_predicted_probability",
        "run_b_correct",
        "correctness_case",
        "n_segments",
        "evidence_bin",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fly-level prediction errors between two runs.")
    run_a = parser.add_mutually_exclusive_group(required=True)
    run_a.add_argument("--run-a", help="Run directory for the first experiment.")
    run_a.add_argument("--predictions-a", help="Prediction CSV for the first experiment.")
    run_b = parser.add_mutually_exclusive_group(required=True)
    run_b.add_argument("--run-b", help="Run directory for the second experiment.")
    run_b.add_argument("--predictions-b", help="Prediction CSV for the second experiment.")
    parser.add_argument("--axis", required=True, help="Label axis to compare, e.g. genotype or cohort.")
    parser.add_argument("--run-a-name", default="run_a", help="Display name for the first experiment.")
    parser.add_argument("--run-b-name", default="run_b", help="Display name for the second experiment.")
    parser.add_argument("--split", default="valid", help="Prediction split to compare. Use 'all' for all splits.")
    parser.add_argument(
        "--join-without-fold",
        action="store_true",
        help="Ignore fold when joining prediction rows from CV runs with different fold assignments.",
    )
    parser.add_argument("--output", help="Optional path for joined comparison CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_a = args.run_a or args.predictions_a
    input_b = args.run_b or args.predictions_b
    assert input_a is not None
    assert input_b is not None
    path_a = _prediction_path_for_input(input_a)
    path_b = _prediction_path_for_input(input_b)
    split = None if args.split == "all" else str(args.split)
    rows_a = _load_rows(path_a, axis=str(args.axis), split=split)
    rows_b = _load_rows(path_b, axis=str(args.axis), split=split)
    joined, key_columns = _joined_rows(rows_a, rows_b, include_fold=not args.join_without_fold)
    summary = _summarize(joined)

    print(f"run_a_predictions: {path_a}")
    print(f"run_b_predictions: {path_b}")
    print(f"axis: {args.axis}")
    print(f"join_columns: {', '.join(key_columns)}")
    if split is not None:
        print(f"prediction_split: {split}")
    print()
    _print_summary(summary, run_a_name=args.run_a_name, run_b_name=args.run_b_name)
    if args.output:
        _write_joined_rows(args.output, joined)
        print()
        print(f"wrote_joined_predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
