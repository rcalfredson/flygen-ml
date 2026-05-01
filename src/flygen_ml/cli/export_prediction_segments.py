from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TextIO


PREDICTION_FIELDNAMES = [
    "prediction_fold",
    "prediction_split",
    "prediction_label_key",
    "prediction_actual_label",
    "prediction_predicted_label",
    "prediction_correct",
    "prediction_probability",
    "prediction_decision_margin",
    "prediction_evidence_bin",
    "prediction_n_segments",
    "prediction_n_segments_with_qc_flags",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export segment rows for selected fly-level prediction review rows."
    )
    parser.add_argument("--prediction-review", required=True, help="Prediction review CSV from inspect_predictions.")
    parser.add_argument("--segments", required=True, help="Input segment table path.")
    parser.add_argument("--output", help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--split", default="valid", help="Prediction split to export.")
    parser.add_argument("--errors-only", action="store_true", help="Export only misclassified prediction rows.")
    parser.add_argument(
        "--min-decision-margin",
        type=float,
        default=None,
        help="Keep prediction rows with decision_margin at least this value.",
    )
    parser.add_argument(
        "--max-decision-margin",
        type=float,
        default=None,
        help="Keep prediction rows with decision_margin at most this value.",
    )
    parser.add_argument("--actual-label", help="Keep prediction rows with this actual label.")
    parser.add_argument("--predicted-label", help="Keep prediction rows with this predicted label.")
    parser.add_argument("--evidence-bin", help="Keep prediction rows with this prediction evidence bin.")
    parser.add_argument(
        "--fly-id",
        action="append",
        default=[],
        help="Keep one fly_id. May be supplied multiple times.",
    )
    return parser


def _load_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"CSV table is empty: {path}")
    return rows, fieldnames


def _is_false(value: object) -> bool:
    return str(value).strip().lower() in {"false", "0", "no"}


def _passes_margin_filters(
    row: dict[str, str],
    *,
    min_decision_margin: float | None,
    max_decision_margin: float | None,
) -> bool:
    if min_decision_margin is None and max_decision_margin is None:
        return True
    try:
        margin = float(row["decision_margin"])
    except (KeyError, TypeError, ValueError):
        return False
    if min_decision_margin is not None and margin < min_decision_margin:
        return False
    if max_decision_margin is not None and margin > max_decision_margin:
        return False
    return True


def _filter_prediction_review_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    errors_only: bool,
    min_decision_margin: float | None,
    max_decision_margin: float | None,
    actual_label: str | None,
    predicted_label: str | None,
    evidence_bin: str | None,
    fly_ids: set[str],
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("split", "") != split:
            continue
        if errors_only and not _is_false(row.get("correct", "")):
            continue
        if actual_label is not None and row.get("actual_label", "") != actual_label:
            continue
        if predicted_label is not None and row.get("predicted_label", "") != predicted_label:
            continue
        if evidence_bin is not None and row.get("evidence_bin", "") != evidence_bin:
            continue
        if fly_ids and row.get("fly_id", "") not in fly_ids:
            continue
        if not _passes_margin_filters(
            row,
            min_decision_margin=min_decision_margin,
            max_decision_margin=max_decision_margin,
        ):
            continue
        filtered.append(row)
    return filtered


def _segment_key(row: dict[str, str]) -> tuple[str, str]:
    return row["fly_id"], row["sample_key"]


def _segments_by_prediction_key(segment_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in segment_rows:
        grouped.setdefault(_segment_key(row), []).append(row)
    return grouped


def _prediction_prefix(row: dict[str, str]) -> dict[str, str]:
    return {
        "prediction_fold": row.get("fold", ""),
        "prediction_split": row.get("split", ""),
        "prediction_label_key": row.get("label_key", ""),
        "prediction_actual_label": row.get("actual_label", ""),
        "prediction_predicted_label": row.get("predicted_label", ""),
        "prediction_correct": row.get("correct", ""),
        "prediction_probability": row.get("predicted_probability", ""),
        "prediction_decision_margin": row.get("decision_margin", ""),
        "prediction_evidence_bin": row.get("evidence_bin", ""),
        "prediction_n_segments": row.get("n_segments", ""),
        "prediction_n_segments_with_qc_flags": row.get("n_segments_with_qc_flags", ""),
    }


def build_prediction_segment_rows(
    *,
    prediction_review_rows: list[dict[str, str]],
    segment_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    segment_rows_by_key = _segments_by_prediction_key(segment_rows)
    output_rows: list[dict[str, str]] = []
    missing_keys: list[tuple[str, str]] = []
    for prediction_row in prediction_review_rows:
        key = _segment_key(prediction_row)
        matching_segments = segment_rows_by_key.get(key)
        if not matching_segments:
            missing_keys.append(key)
            continue
        prefix = _prediction_prefix(prediction_row)
        for segment_row in matching_segments:
            output_rows.append({**prefix, **segment_row})
    if missing_keys:
        preview = ", ".join(f"fly_id={fly_id!r}, sample_key={sample_key!r}" for fly_id, sample_key in missing_keys[:5])
        raise ValueError(f"missing segment rows for {len(missing_keys)} prediction rows: {preview}")
    return output_rows


def write_prediction_segment_rows(
    rows: list[dict[str, str]],
    handle: TextIO,
    *,
    segment_fieldnames: list[str],
) -> None:
    writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDNAMES + segment_fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> int:
    args = build_parser().parse_args()
    prediction_review_rows, _ = _load_csv_rows(args.prediction_review)
    segment_rows, segment_fieldnames = _load_csv_rows(args.segments)
    selected_prediction_rows = _filter_prediction_review_rows(
        prediction_review_rows,
        split=args.split,
        errors_only=args.errors_only,
        min_decision_margin=args.min_decision_margin,
        max_decision_margin=args.max_decision_margin,
        actual_label=args.actual_label,
        predicted_label=args.predicted_label,
        evidence_bin=args.evidence_bin,
        fly_ids=set(args.fly_id),
    )
    output_rows = build_prediction_segment_rows(
        prediction_review_rows=selected_prediction_rows,
        segment_rows=segment_rows,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            write_prediction_segment_rows(output_rows, handle, segment_fieldnames=segment_fieldnames)
        print(
            f"wrote {len(output_rows)} segment rows from {len(selected_prediction_rows)} "
            f"prediction rows to {output_path}"
        )
    else:
        write_prediction_segment_rows(output_rows, sys.stdout, segment_fieldnames=segment_fieldnames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
