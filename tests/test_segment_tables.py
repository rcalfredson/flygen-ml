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
            cohort="intact",
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


def test_segment_table_loads_legacy_table_without_cohort(tmp_path):
    path = tmp_path / "legacy_segments.csv"
    path.write_text(
        "\n".join(
            [
                "segment_id,sample_key,fly_id,genotype,chamber_type,experimental_fly_idx,data_path,trx_path,training_idx,training_start_frame,training_end_frame,anchor_reward_frame,start_frame,stop_frame,end_reward_frame,duration_frames,n_finite_frames,finite_frame_fraction,qc_flags,reward_center_x,reward_center_y,reward_radius,terminated_by_training_end,anchor_reward_kind",
                "sample__tr0__seg0,sample,sample__fly1,control,ct1,1,/tmp/sample.data,/tmp/sample.trx,0,100,200,110,120,150,150,30,28,0.9333333333333333,has_missing_frames,1.5,2.5,3.5,false,calculated",
            ]
        )
    )

    observed = load_segment_table(path)

    assert observed[0].cohort is None
