from __future__ import annotations

from pathlib import Path

from flygen_ml.schema import SegmentRecord
from flygen_ml.segment_table import load_segment_table, write_segment_table


def test_segment_table_roundtrip(tmp_path):
    path = tmp_path / "segments.csv"
    rows = [
        SegmentRecord(
            segment_id="sample__tr0__seg0",
            sample_key="sample",
            fly_id="sample__fly1",
            genotype="control",
            chamber_type="ct1",
            experimental_fly_idx=1,
            data_path=Path("/tmp/sample.data"),
            trx_path=Path("/tmp/sample.trx"),
            training_idx=0,
            training_start_frame=100,
            training_end_frame=200,
            anchor_reward_frame=110,
            start_frame=120,
            stop_frame=150,
            end_reward_frame=150,
            duration_frames=30,
            n_finite_frames=28,
            finite_frame_fraction=28 / 30,
            qc_flags=("has_missing_frames",),
            reward_center_x=1.5,
            reward_center_y=2.5,
            reward_radius=3.5,
            terminated_by_training_end=False,
        )
    ]

    write_segment_table(path, rows)
    observed = load_segment_table(path)

    assert observed == rows
