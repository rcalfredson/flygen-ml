from __future__ import annotations

import csv
from pathlib import Path

from flygen_ml.features.segment_inspection import build_segment_metric_rows, write_segment_metric_rows
from flygen_ml.schema import SegmentRecord


def _segment(segment_id: str, start_frame: int, stop_frame: int) -> SegmentRecord:
    return SegmentRecord(
        segment_id=segment_id,
        sample_key="sample0",
        fly_id="sample0__fly0",
        genotype="A",
        chamber_type="large",
        experimental_fly_idx=0,
        data_path=Path("/tmp/sample.data"),
        trx_path=Path("/tmp/sample.trx"),
        training_idx=1,
        training_start_frame=0,
        training_end_frame=10,
        anchor_reward_frame=start_frame - 1,
        start_frame=start_frame,
        stop_frame=stop_frame,
        end_reward_frame=stop_frame,
        duration_frames=stop_frame - start_frame,
        n_finite_frames=stop_frame - start_frame,
        finite_frame_fraction=1.0,
        reward_center_x=0.0,
        reward_center_y=0.0,
        reward_radius=5.0,
    )


def test_build_segment_metric_rows_ranks_segments(monkeypatch, tmp_path):
    from flygen_ml.features import segment_inspection

    segments = [_segment("s0", 0, 2), _segment("s1", 0, 4)]

    class Recording:
        x_by_fly = [[0.0, 3.0, 0.0, 8.0]]
        y_by_fly = [[0.0, 4.0, 0.0, 6.0]]

    monkeypatch.setattr(segment_inspection, "load_recording_pair", lambda data_path, trx_path: ({}, {}))
    monkeypatch.setattr(segment_inspection, "build_normalized_recording", lambda manifest_row, raw_data, raw_trx: Recording())

    rows = build_segment_metric_rows(segments=segments, metric="end_radius_px")

    assert [row["segment_id"] for row in rows] == ["s1", "s0"]
    assert rows[0]["rank"] == 1
    assert rows[0]["metric_value"] == 10.0
    assert rows[1]["metric_value"] == 5.0

    output_path = tmp_path / "segment_report.csv"
    with output_path.open("w", newline="") as handle:
        write_segment_metric_rows(rows, handle)

    written = list(csv.DictReader(output_path.open()))
    assert written[0]["segment_id"] == "s1"
    assert written[0]["metric"] == "end_radius_px"
