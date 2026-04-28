from __future__ import annotations

from pathlib import Path

from flygen_ml.cli.build_features import _filter_segments_for_feature_building
from flygen_ml.schema import SegmentRecord


def _segment(segment_id: str, *, terminated_by_training_end: bool) -> SegmentRecord:
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
        anchor_reward_frame=1,
        start_frame=2,
        stop_frame=5,
        end_reward_frame=None if terminated_by_training_end else 5,
        duration_frames=3,
        n_finite_frames=3,
        finite_frame_fraction=1.0,
        terminated_by_training_end=terminated_by_training_end,
    )


def test_filter_segments_for_feature_building_omits_training_end_segments_by_default():
    reward_terminated = _segment("reward_terminated", terminated_by_training_end=False)
    training_end_terminated = _segment("training_end_terminated", terminated_by_training_end=True)

    observed = _filter_segments_for_feature_building([reward_terminated, training_end_terminated])

    assert observed == [reward_terminated]


def test_filter_segments_for_feature_building_can_include_training_end_segments():
    reward_terminated = _segment("reward_terminated", terminated_by_training_end=False)
    training_end_terminated = _segment("training_end_terminated", terminated_by_training_end=True)

    observed = _filter_segments_for_feature_building(
        [reward_terminated, training_end_terminated],
        include_training_end_segments=True,
    )

    assert observed == [reward_terminated, training_end_terminated]
