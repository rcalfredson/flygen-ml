from __future__ import annotations

from pathlib import Path

import numpy as np

from flygen_ml.features.sequence import RICH_SEQUENCE_CHANNELS, build_sequence_sample
from flygen_ml.schema import ManifestRow, NormalizedRecording, SegmentRecord


def _recording() -> NormalizedRecording:
    manifest = ManifestRow(
        sample_key="sample0",
        data_path=Path("/tmp/sample.data"),
        trx_path=Path("/tmp/sample.trx"),
        genotype="A",
        chamber="large",
        training_idx=0,
        cohort="intact",
        fly_idx=0,
    )
    return NormalizedRecording(
        sample_key="sample0",
        manifest=manifest,
        chamber_type="large",
        experimental_fly_idx=0,
        training_idx=0,
        training_start_frame=0,
        training_end_frame=5,
        fps=float("nan"),
        timestamps=np.arange(5),
        x_by_fly=np.asarray([[10.0, 11.0, 12.0, 13.0, 14.0]]),
        y_by_fly=np.asarray([[20.0, 20.0, 21.0, 21.0, 22.0]]),
        protocol={},
        raw_data={},
        raw_trx={},
    )


def _segment() -> SegmentRecord:
    return SegmentRecord(
        segment_id="seg0",
        sample_key="sample0",
        fly_id="sample0__fly0",
        genotype="A",
        cohort="intact",
        chamber_type="large",
        experimental_fly_idx=0,
        data_path=Path("/tmp/sample.data"),
        trx_path=Path("/tmp/sample.trx"),
        training_idx=0,
        training_start_frame=0,
        training_end_frame=5,
        anchor_reward_frame=0,
        start_frame=1,
        stop_frame=5,
        end_reward_frame=5,
        duration_frames=4,
        n_finite_frames=4,
        finite_frame_fraction=1.0,
        reward_center_x=10.0,
        reward_center_y=20.0,
        reward_radius=2.0,
    )


def test_build_sequence_sample_exports_reward_normalized_channels():
    sample = build_sequence_sample(_recording(), _segment(), target_length=4)

    assert sample.x.shape == (4, 6)
    assert sample.mask.tolist() == [True, True, True, True]
    assert sample.channels == ("x_rel", "y_rel", "dx_rel", "dy_rel", "speed_rel", "r_rel")
    np.testing.assert_allclose(sample.x[:, 0], [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(sample.x[:, 1], [0.0, 0.5, 0.5, 1.0])


def test_build_sequence_sample_exports_rich_motion_channels():
    sample = build_sequence_sample(
        _recording(),
        _segment(),
        target_length=4,
        channels=RICH_SEQUENCE_CHANNELS,
    )

    assert sample.x.shape == (4, 13)
    assert sample.channels == RICH_SEQUENCE_CHANNELS
    np.testing.assert_allclose(sample.x[:, 4], [0.0, np.sqrt(0.5), 0.5, np.sqrt(0.5)])
    np.testing.assert_allclose(sample.x[:, 5], [0.0, np.sqrt(0.5), 0.5 - np.sqrt(0.5), np.sqrt(0.5) - 0.5])
    np.testing.assert_allclose(sample.x[:, 7], [0.0, 0.6708204, 0.47434166, 0.6708204], rtol=1e-6)
    np.testing.assert_allclose(sample.x[:, 8], [0.0, 0.2236068, -0.15811388, 0.2236068], rtol=1e-6)
    np.testing.assert_allclose(sample.x[:, 9], [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(sample.x[:, 10], [0.0, np.sqrt(0.5), 1.0, np.sqrt(0.5)])
    np.testing.assert_allclose(sample.x[:, 11], [0.0, np.sqrt(0.5), -np.sqrt(0.5), np.sqrt(0.5)])
