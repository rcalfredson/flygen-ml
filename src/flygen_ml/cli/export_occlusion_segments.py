from __future__ import annotations

import argparse
import csv
from pathlib import Path


PLOTTER_COMPAT_COLUMNS = [
    "prediction_actual_label",
    "prediction_predicted_label",
    "prediction_correct",
    "prediction_probability",
    "prediction_decision_margin",
    "prediction_evidence_bin",
    "prediction_n_segments",
    "prediction_n_segments_with_qc_flags",
]


def _load_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"CSV table is empty: {path}")
    return rows, fieldnames


def _segments_by_id(segment_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in segment_rows:
        segment_id = row.get("segment_id", "")
        if not segment_id:
            continue
        if segment_id in indexed:
            duplicates.append(segment_id)
        indexed[segment_id] = row
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"segment table has duplicate segment_id values: {preview}")
    return indexed


def _joint_label(row: dict[str, str], *, genotype_key: str, cohort_key: str) -> str:
    genotype = row.get(genotype_key, "")
    cohort = row.get(cohort_key, "")
    if genotype or cohort:
        return f"{genotype}|{cohort}"
    return ""


def _prediction_compat_prefix(row: dict[str, str]) -> dict[str, str]:
    both_correct = (
        row.get("actual_genotype", "") == row.get("predicted_genotype", "")
        and row.get("actual_cohort", "") == row.get("predicted_cohort", "")
    )
    return {
        "prediction_actual_label": _joint_label(
            row,
            genotype_key="actual_genotype",
            cohort_key="actual_cohort",
        ),
        "prediction_predicted_label": _joint_label(
            row,
            genotype_key="predicted_genotype",
            cohort_key="predicted_cohort",
        ),
        "prediction_correct": str(both_correct),
        "prediction_probability": row.get("baseline_predicted_genotype_probability", ""),
        "prediction_decision_margin": row.get("filter_abs_logit_delta", ""),
        "prediction_evidence_bin": row.get("occlusion_status", ""),
        "prediction_n_segments": row.get("n_segments", ""),
        "prediction_n_segments_with_qc_flags": row.get("n_segments_with_qc_flags", ""),
    }


def build_occlusion_segment_rows(
    *,
    occlusion_rows: list[dict[str, str]],
    segment_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    segment_index = _segments_by_id(segment_rows)
    output_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for row in occlusion_rows:
        segment_id = row.get("segment_id", "")
        segment_row = segment_index.get(segment_id)
        if segment_row is None:
            missing.append(segment_id)
            continue
        merged = {
            **_prediction_compat_prefix(row),
            **segment_row,
            **row,
        }
        output_rows.append(merged)
    if missing:
        preview = ", ".join(repr(segment_id) for segment_id in missing[:5])
        raise ValueError(f"missing segment metadata for {len(missing)} occlusion rows: {preview}")
    return output_rows


def _fieldnames(
    *,
    occlusion_fieldnames: list[str],
    segment_fieldnames: list[str],
) -> list[str]:
    ordered: list[str] = []
    for name in [*PLOTTER_COMPAT_COLUMNS, *segment_fieldnames, *occlusion_fieldnames]:
        if name not in ordered:
            ordered.append(name)
    return ordered


def write_occlusion_segment_rows(
    path: str | Path,
    rows: list[dict[str, str]],
    *,
    occlusion_fieldnames: list[str],
    segment_fieldnames: list[str],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_fieldnames(
                occlusion_fieldnames=occlusion_fieldnames,
                segment_fieldnames=segment_fieldnames,
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join occlusion evidence rows to source segment metadata for trajectory plotting.",
    )
    parser.add_argument("--occlusion-csv", required=True, help="Input occlusion/filter CSV.")
    parser.add_argument("--segments", required=True, help="Source segment table, e.g. artifacts/segments_with_cohort.csv.")
    parser.add_argument("--output", required=True, help="Output plot-ready segment CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    occlusion_rows, occlusion_fieldnames = _load_csv_rows(args.occlusion_csv)
    segment_rows, segment_fieldnames = _load_csv_rows(args.segments)
    output_rows = build_occlusion_segment_rows(
        occlusion_rows=occlusion_rows,
        segment_rows=segment_rows,
    )
    write_occlusion_segment_rows(
        args.output,
        output_rows,
        occlusion_fieldnames=occlusion_fieldnames,
        segment_fieldnames=segment_fieldnames,
    )
    print(f"wrote {len(output_rows)} occlusion segment rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
