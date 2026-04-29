from __future__ import annotations

import argparse
import csv
from pathlib import Path

from flygen_ml.features.aggregation import AGGREGATED_FEATURE_NAMES, aggregate_segment_features
from flygen_ml.features.engineered import compute_engineered_features
from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.trajectory_builder import build_normalized_recording
from flygen_ml.schema import ManifestRow, SegmentRecord
from flygen_ml.segment_table import load_segment_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build feature tables from extracted segments.")
    parser.add_argument("--feature-set", required=True, help="Feature set name.")
    parser.add_argument("--segments", required=True, help="Input segment table path.")
    parser.add_argument("--output", required=True, help="Output feature table path.")
    parser.add_argument(
        "--include-training-end-segments",
        action="store_true",
        help="Include segments that ended only because the selected training ended.",
    )
    return parser


def _manifest_row_from_segment(segment: SegmentRecord) -> ManifestRow:
    return ManifestRow(
        sample_key=segment.sample_key,
        data_path=segment.data_path,
        trx_path=segment.trx_path,
        genotype=segment.genotype,
        cohort=segment.cohort,
        chamber=segment.chamber_type,
        training_idx=segment.training_idx,
        fly_idx=segment.experimental_fly_idx,
    )


def _compute_segment_feature_rows(segments: list[SegmentRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    recordings_by_sample: dict[str, object] = {}
    for segment in segments:
        recording = recordings_by_sample.get(segment.sample_key)
        if recording is None:
            manifest_row = _manifest_row_from_segment(segment)
            raw_data, raw_trx = load_recording_pair(segment.data_path, segment.trx_path)
            recording = build_normalized_recording(manifest_row, raw_data, raw_trx)
            recordings_by_sample[segment.sample_key] = recording
        row: dict[str, object] = {
            "segment_id": segment.segment_id,
            "sample_key": segment.sample_key,
            "fly_id": segment.fly_id,
            "genotype": segment.genotype,
            "cohort": segment.cohort or "",
            "chamber_type": segment.chamber_type,
            "training_idx": segment.training_idx,
            "qc_flags": "|".join(segment.qc_flags),
        }
        row.update(compute_engineered_features(recording, segment))
        rows.append(row)
    return rows


def _filter_segments_for_feature_building(
    segments: list[SegmentRecord],
    *,
    include_training_end_segments: bool = False,
) -> list[SegmentRecord]:
    if include_training_end_segments:
        return segments
    return [segment for segment in segments if not segment.terminated_by_training_end]


def _write_feature_table(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fly_id",
        "sample_key",
        "genotype",
        "cohort",
        "chamber_type",
        "training_idx",
        "n_segments",
        "n_segments_with_qc_flags",
    ] + [f"{name}_mean" for name in AGGREGATED_FEATURE_NAMES]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = build_parser().parse_args()
    if args.feature_set != "engineered_v1":
        raise ValueError(f"unsupported feature set: {args.feature_set}")
    segments = load_segment_table(args.segments)
    feature_segments = _filter_segments_for_feature_building(
        segments,
        include_training_end_segments=args.include_training_end_segments,
    )
    n_filtered = len(segments) - len(feature_segments)
    aggregated_rows = aggregate_segment_features(_compute_segment_feature_rows(feature_segments))
    _write_feature_table(args.output, aggregated_rows)
    print(f"wrote {len(aggregated_rows)} fly-level engineered rows to {args.output}")
    if n_filtered:
        print(f"omitted {n_filtered} training-end-terminated segments from feature building")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
