from __future__ import annotations

from pathlib import Path

import pytest

from flygen_ml.features.engineered import compute_engineered_features
from flygen_ml.schema import ManifestRow, NormalizedRecording, SegmentRecord


def _recording() -> NormalizedRecording:
    return NormalizedRecording(
        sample_key="sample",
        manifest=ManifestRow(
            sample_key="sample",
            data_path=Path("/tmp/sample.data"),
            trx_path=Path("/tmp/sample.trx"),
            genotype="control",
            chamber="ct1",
            training_idx=0,
            fly_idx=0,
        ),
        chamber_type="ct1",
        experimental_fly_idx=0,
        training_idx=0,
        training_start_frame=0,
        training_end_frame=3,
        fps=float("nan"),
        timestamps=[0.0, 1.0, 2.0],
        x_by_fly=[[1.0, 2.0, 3.0]],
        y_by_fly=[[0.0, 0.0, 0.0]],
        protocol={},
        raw_data={},
        raw_trx={},
    )


def test_compute_engineered_features_from_segment_geometry():
    segment = SegmentRecord(
        segment_id="sample__tr0__seg0",
        sample_key="sample",
        fly_id="sample__fly0",
        genotype="control",
        chamber_type="ct1",
        experimental_fly_idx=0,
        data_path=Path("/tmp/sample.data"),
        trx_path=Path("/tmp/sample.trx"),
        training_idx=0,
        training_start_frame=0,
        training_end_frame=3,
        anchor_reward_frame=0,
        start_frame=0,
        stop_frame=3,
        end_reward_frame=None,
        duration_frames=3,
        n_finite_frames=3,
        finite_frame_fraction=1.0,
        reward_center_x=0.0,
        reward_center_y=0.0,
        reward_radius=1.0,
        terminated_by_training_end=True,
    )

    features = compute_engineered_features(_recording(), segment)

    assert features["duration_frames"] == 3.0
    assert features["finite_frame_fraction"] == 1.0
    assert features["path_length_px"] == pytest.approx(2.0)
    assert features["net_displacement_px"] == pytest.approx(2.0)
    assert features["straightness"] == pytest.approx(1.0)
    assert features["mean_step_distance_px"] == pytest.approx(1.0)
    assert features["mean_radius_px"] == pytest.approx(2.0)
    assert features["radius_std_px"] == pytest.approx(0.81649658)
    assert features["start_radius_px"] == pytest.approx(1.0)
    assert features["end_radius_px"] == pytest.approx(3.0)
    assert features["radius_delta_px"] == pytest.approx(2.0)
