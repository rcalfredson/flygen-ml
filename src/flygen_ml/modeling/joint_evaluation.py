from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_PREDICTION_COLUMNS = {"fly_id", "sample_key"}


def prediction_path_for_input(path: str | Path) -> Path:
    input_path = Path(path)
    if input_path.is_file():
        return input_path
    if not input_path.is_dir():
        raise FileNotFoundError(f"prediction input does not exist: {input_path}")

    cv_path = input_path / "cv_predictions.csv"
    holdout_path = input_path / "predictions.csv"
    if cv_path.exists():
        return cv_path
    if holdout_path.exists():
        return holdout_path
    raise FileNotFoundError(
        f"missing prediction table: expected {cv_path} or {holdout_path}"
    )


def load_prediction_rows(path: str | Path, *, split: str | None = None) -> list[dict[str, str]]:
    prediction_path = Path(path)
    with prediction_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_PREDICTION_COLUMNS - fieldnames)
        if missing:
            raise ValueError(
                f"prediction table {prediction_path} is missing required columns: {', '.join(missing)}"
            )
        rows = [
            row
            for row in reader
            if split is None or row.get("split", "") == split
        ]
    if not rows:
        split_suffix = "" if split is None else f" for split {split!r}"
        raise ValueError(f"prediction table has no rows{split_suffix}: {prediction_path}")
    return rows


def _has_nonempty_column(rows: list[dict[str, str]], column: str) -> bool:
    return any(row.get(column, "") != "" for row in rows)


def join_key_columns(
    axis_a_rows: list[dict[str, str]],
    axis_b_rows: list[dict[str, str]],
    *,
    include_fold: bool = True,
) -> list[str]:
    columns = ["fly_id", "sample_key"]
    if _has_nonempty_column(axis_a_rows, "split") and _has_nonempty_column(axis_b_rows, "split"):
        columns.append("split")
    if include_fold and _has_nonempty_column(axis_a_rows, "fold") and _has_nonempty_column(axis_b_rows, "fold"):
        columns.append("fold")
    return columns


def _row_key(row: dict[str, str], columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in columns)


def _index_rows(rows: list[dict[str, str]], *, key_columns: list[str], axis_name: str) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = _row_key(row, key_columns)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        preview = ", ".join(str(key) for key in duplicates[:3])
        raise ValueError(f"{axis_name} prediction table has duplicate join keys: {preview}")
    return indexed


def _actual_label(row: dict[str, str]) -> str:
    return row.get("actual_label") or row["actual_genotype"]


def _predicted_label(row: dict[str, str]) -> str:
    return row.get("predicted_label") or row["predicted_genotype"]


def _joint_label(axis_a_label: str, axis_b_label: str) -> str:
    return f"{axis_a_label}|{axis_b_label}"


def _evidence_bin(axis_a_row: dict[str, str], axis_b_row: dict[str, str]) -> str:
    axis_a_bin = axis_a_row.get("evidence_bin", "")
    axis_b_bin = axis_b_row.get("evidence_bin", "")
    if axis_a_bin and axis_b_bin and axis_a_bin != axis_b_bin:
        return f"{axis_a_bin}|{axis_b_bin}"
    return axis_a_bin or axis_b_bin


def join_prediction_rows(
    axis_a_rows: list[dict[str, str]],
    axis_b_rows: list[dict[str, str]],
    *,
    axis_a_name: str,
    axis_b_name: str,
    include_fold: bool = True,
) -> tuple[list[dict[str, str]], list[str]]:
    key_columns = join_key_columns(axis_a_rows, axis_b_rows, include_fold=include_fold)
    axis_a_index = _index_rows(axis_a_rows, key_columns=key_columns, axis_name=axis_a_name)
    axis_b_index = _index_rows(axis_b_rows, key_columns=key_columns, axis_name=axis_b_name)
    common_keys = sorted(set(axis_a_index) & set(axis_b_index))
    if not common_keys:
        raise ValueError(
            f"prediction tables have no overlapping rows on join columns: {', '.join(key_columns)}"
        )

    joined_rows: list[dict[str, str]] = []
    for key in common_keys:
        axis_a_row = axis_a_index[key]
        axis_b_row = axis_b_index[key]
        axis_a_actual = _actual_label(axis_a_row)
        axis_a_predicted = _predicted_label(axis_a_row)
        axis_b_actual = _actual_label(axis_b_row)
        axis_b_predicted = _predicted_label(axis_b_row)
        axis_a_correct = axis_a_actual == axis_a_predicted
        axis_b_correct = axis_b_actual == axis_b_predicted
        actual_joint = _joint_label(axis_a_actual, axis_b_actual)
        predicted_joint = _joint_label(axis_a_predicted, axis_b_predicted)

        output_row: dict[str, str] = {
            "fly_id": axis_a_row.get("fly_id", ""),
            "sample_key": axis_a_row.get("sample_key", ""),
            "split": axis_a_row.get("split") or axis_b_row.get("split", ""),
            "fold": axis_a_row.get("fold") or axis_b_row.get("fold", ""),
            f"{axis_a_name}_actual_label": axis_a_actual,
            f"{axis_a_name}_predicted_label": axis_a_predicted,
            f"{axis_a_name}_correct": str(axis_a_correct),
            f"{axis_a_name}_predicted_probability": axis_a_row.get("predicted_probability", ""),
            f"{axis_a_name}_evidence_bin": axis_a_row.get("evidence_bin", ""),
            f"{axis_b_name}_actual_label": axis_b_actual,
            f"{axis_b_name}_predicted_label": axis_b_predicted,
            f"{axis_b_name}_correct": str(axis_b_correct),
            f"{axis_b_name}_predicted_probability": axis_b_row.get("predicted_probability", ""),
            f"{axis_b_name}_evidence_bin": axis_b_row.get("evidence_bin", ""),
            "actual_joint_label": actual_joint,
            "predicted_joint_label": predicted_joint,
            "joint_correct": str(actual_joint == predicted_joint),
            "evidence_bin": _evidence_bin(axis_a_row, axis_b_row),
        }
        joined_rows.append(output_row)
    return joined_rows, key_columns


def summarize_joint_predictions(
    rows: list[dict[str, str]],
    *,
    axis_a_name: str,
    axis_b_name: str,
) -> dict[str, object]:
    summary = _summarize_joint_prediction_counts(
        rows,
        axis_a_name=axis_a_name,
        axis_b_name=axis_b_name,
    )
    summary["by_evidence_bin"] = _summarize_by_evidence_bin(
        rows,
        axis_a_name=axis_a_name,
        axis_b_name=axis_b_name,
    )
    return summary


def _summarize_joint_prediction_counts(
    rows: list[dict[str, str]],
    *,
    axis_a_name: str,
    axis_b_name: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty joint predictions")

    n_examples = len(rows)
    axis_a_correct_key = f"{axis_a_name}_correct"
    axis_b_correct_key = f"{axis_b_name}_correct"
    both_correct = 0
    axis_a_only_wrong = 0
    axis_b_only_wrong = 0
    both_wrong = 0
    for row in rows:
        axis_a_correct = row[axis_a_correct_key] == "True"
        axis_b_correct = row[axis_b_correct_key] == "True"
        if axis_a_correct and axis_b_correct:
            both_correct += 1
        elif not axis_a_correct and axis_b_correct:
            axis_a_only_wrong += 1
        elif axis_a_correct and not axis_b_correct:
            axis_b_only_wrong += 1
        else:
            both_wrong += 1

    labels = sorted(
        {row["actual_joint_label"] for row in rows}
        | {row["predicted_joint_label"] for row in rows}
    )
    confusion_counts = Counter(
        (row["actual_joint_label"], row["predicted_joint_label"])
        for row in rows
    )

    return {
        "n_joined_examples": n_examples,
        "joint_accuracy": sum(row["joint_correct"] == "True" for row in rows) / n_examples,
        "axis_a_accuracy": sum(row[axis_a_correct_key] == "True" for row in rows) / n_examples,
        "axis_b_accuracy": sum(row[axis_b_correct_key] == "True" for row in rows) / n_examples,
        "both_correct": both_correct,
        "axis_a_only_wrong": axis_a_only_wrong,
        "axis_b_only_wrong": axis_b_only_wrong,
        "both_wrong": both_wrong,
        "joint_labels": labels,
        "joint_confusion_matrix": {
            actual: {
                predicted: confusion_counts[(actual, predicted)]
                for predicted in labels
            }
            for actual in labels
        },
    }


def _summarize_by_evidence_bin(
    rows: list[dict[str, str]],
    *,
    axis_a_name: str,
    axis_b_name: str,
) -> dict[str, dict[str, float | int]]:
    rows_by_bin: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        evidence_bin = row.get("evidence_bin", "")
        if evidence_bin:
            rows_by_bin.setdefault(evidence_bin, []).append(row)

    summaries: dict[str, dict[str, float | int]] = {}
    for evidence_bin, bin_rows in sorted(rows_by_bin.items()):
        bin_summary = _summarize_joint_prediction_counts(
            bin_rows,
            axis_a_name=axis_a_name,
            axis_b_name=axis_b_name,
        )
        summaries[evidence_bin] = {
            "n_joined_examples": int(bin_summary["n_joined_examples"]),
            "joint_accuracy": float(bin_summary["joint_accuracy"]),
            "axis_a_accuracy": float(bin_summary["axis_a_accuracy"]),
            "axis_b_accuracy": float(bin_summary["axis_b_accuracy"]),
        }
    return summaries


def joint_prediction_fieldnames(rows: list[dict[str, str]], *, axis_a_name: str, axis_b_name: str) -> list[str]:
    fieldnames = [
        "fold",
        "split",
        "fly_id",
        "sample_key",
        f"{axis_a_name}_actual_label",
        f"{axis_a_name}_predicted_label",
        f"{axis_a_name}_correct",
        f"{axis_a_name}_predicted_probability",
        f"{axis_a_name}_evidence_bin",
        f"{axis_b_name}_actual_label",
        f"{axis_b_name}_predicted_label",
        f"{axis_b_name}_correct",
        f"{axis_b_name}_predicted_probability",
        f"{axis_b_name}_evidence_bin",
        "actual_joint_label",
        "predicted_joint_label",
        "joint_correct",
        "evidence_bin",
    ]
    if not any(row.get("fold", "") for row in rows):
        fieldnames.remove("fold")
    if not any(row.get("split", "") for row in rows):
        fieldnames.remove("split")
    return fieldnames


def write_joint_prediction_rows(
    path: str | Path,
    rows: list[dict[str, str]],
    *,
    axis_a_name: str,
    axis_b_name: str,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=joint_prediction_fieldnames(
                rows,
                axis_a_name=axis_a_name,
                axis_b_name=axis_b_name,
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
