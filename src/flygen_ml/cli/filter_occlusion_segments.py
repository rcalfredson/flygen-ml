from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SINGLE_DEFICIENCY_LABELS = {
    "pfn-intact": ("PFN>Kir", "antennae-intact"),
    "control-removed": ("Control>Kir", "antennae-removed"),
}


def _load_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"CSV table is empty: {path}")
    return rows, fieldnames


def _float_value(value: object) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _delta_column(*, head: str, delta_class: str) -> str:
    return f"{delta_class}_{head}_logit_delta"


def _filter_deficiency_rows(
    rows: list[dict[str, str]],
    *,
    deficiency: str,
    status: str,
    require_correct: str,
) -> list[dict[str, str]]:
    genotype, cohort = SINGLE_DEFICIENCY_LABELS[deficiency]
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("actual_genotype") != genotype or row.get("actual_cohort") != cohort:
            continue
        if row.get("occlusion_status", "ok") != status:
            continue
        genotype_correct = row.get("actual_genotype", "") == row.get("predicted_genotype", "")
        cohort_correct = row.get("actual_cohort", "") == row.get("predicted_cohort", "")
        if require_correct == "genotype" and not genotype_correct:
            continue
        if require_correct == "cohort" and not cohort_correct:
            continue
        if require_correct == "joint" and not (genotype_correct and cohort_correct):
            continue
        filtered.append(row)
    return filtered


def _split_signed_delta_rows(
    rows: list[dict[str, str]],
    *,
    delta_column: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    positive: list[dict[str, str]] = []
    negative: list[dict[str, str]] = []
    for row in rows:
        delta = _float_value(row.get(delta_column, ""))
        if delta > 0:
            positive.append(row)
        elif delta < 0:
            negative.append(row)
    return positive, negative


def _rank_rows(
    rows: list[dict[str, str]],
    *,
    delta_column: str,
    max_segments_per_fly: int,
    limit: int | None,
) -> list[dict[str, str]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: abs(_float_value(row.get(delta_column, ""))),
        reverse=True,
    )
    counts_by_fly: dict[str, int] = defaultdict(int)
    ranked: list[dict[str, str]] = []
    for row in sorted_rows:
        fly_id = row.get("fly_id", "")
        if counts_by_fly[fly_id] >= max_segments_per_fly:
            continue
        counts_by_fly[fly_id] += 1
        ranked.append(
            {
                "filter_rank": str(len(ranked) + 1),
                "filter_abs_logit_delta": f"{abs(_float_value(row.get(delta_column, ''))):.8g}",
                **row,
            }
        )
        if limit is not None and len(ranked) >= limit:
            break
    return ranked


def _write_rows(path: str | Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filter_rank", "filter_abs_logit_delta"] + fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare positive/negative high-delta occlusion subsets for visual review.",
    )
    parser.add_argument("--occlusion-csv", required=True, help="Source segment occlusion CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for filtered positive/negative CSVs.")
    parser.add_argument(
        "--deficiency",
        choices=tuple(SINGLE_DEFICIENCY_LABELS),
        required=True,
        help="Single-deficiency actual-label group to keep.",
    )
    parser.add_argument(
        "--head",
        choices=("genotype", "cohort"),
        required=True,
        help="Output head whose logit deltas should be split and ranked.",
    )
    parser.add_argument(
        "--delta-class",
        choices=("predicted", "actual"),
        default="predicted",
        help="Use predicted-class or actual-class logit deltas.",
    )
    parser.add_argument(
        "--max-segments-per-fly",
        type=int,
        default=3,
        help="Maximum rows to keep from any one fly in each signed output table.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum rows per signed output table.")
    parser.add_argument("--status", default="ok", help="Occlusion status to keep.")
    parser.add_argument(
        "--require-correct",
        choices=("none", "genotype", "cohort", "joint"),
        default="none",
        help="Keep only flies whose baseline prediction is correct for the selected head or joint label.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_segments_per_fly < 1:
        raise ValueError("--max-segments-per-fly must be at least 1")
    rows, fieldnames = _load_csv_rows(args.occlusion_csv)
    delta_column = _delta_column(head=args.head, delta_class=args.delta_class)
    if delta_column not in fieldnames:
        raise ValueError(f"missing required delta column {delta_column!r} in {args.occlusion_csv}")
    deficiency_rows = _filter_deficiency_rows(
        rows,
        deficiency=args.deficiency,
        status=args.status,
        require_correct=args.require_correct,
    )
    positive_rows, negative_rows = _split_signed_delta_rows(deficiency_rows, delta_column=delta_column)
    positive_ranked = _rank_rows(
        positive_rows,
        delta_column=delta_column,
        max_segments_per_fly=args.max_segments_per_fly,
        limit=args.limit,
    )
    negative_ranked = _rank_rows(
        negative_rows,
        delta_column=delta_column,
        max_segments_per_fly=args.max_segments_per_fly,
        limit=args.limit,
    )
    output_dir = Path(args.output_dir)
    stem = f"{args.deficiency}_{args.head}_{args.delta_class}"
    positive_path = output_dir / f"{stem}_positive_logit_delta.csv"
    negative_path = output_dir / f"{stem}_negative_logit_delta.csv"
    _write_rows(positive_path, positive_ranked, fieldnames=fieldnames)
    _write_rows(negative_path, negative_ranked, fieldnames=fieldnames)
    print(
        f"wrote {len(positive_ranked)} positive and {len(negative_ranked)} negative "
        f"{delta_column} rows from {len(deficiency_rows)} {args.deficiency} rows to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
