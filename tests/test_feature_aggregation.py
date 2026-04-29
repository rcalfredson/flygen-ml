from __future__ import annotations

import pytest

from flygen_ml.features.aggregation import aggregate_segment_features


def test_aggregate_segment_features_to_fly_level():
    rows = [
        {
            "segment_id": "seg0",
            "sample_key": "sample",
            "fly_id": "sample__fly0",
            "genotype": "control",
            "cohort": "intact",
            "chamber_type": "ct1",
            "training_idx": 0,
            "qc_flags": "",
            "duration_frames": 10.0,
            "finite_frame_fraction": 1.0,
            "path_length_px": 2.0,
            "net_displacement_px": 1.0,
            "straightness": 0.5,
            "mean_step_distance_px": 1.0,
            "mean_radius_px": 3.0,
            "radius_std_px": 1.0,
            "start_radius_px": 2.0,
            "end_radius_px": 4.0,
            "radius_delta_px": 2.0,
        },
        {
            "segment_id": "seg1",
            "sample_key": "sample",
            "fly_id": "sample__fly0",
            "genotype": "control",
            "cohort": "intact",
            "chamber_type": "ct1",
            "training_idx": 0,
            "qc_flags": "has_missing_frames",
            "duration_frames": 14.0,
            "finite_frame_fraction": 0.5,
            "path_length_px": 6.0,
            "net_displacement_px": 3.0,
            "straightness": 0.5,
            "mean_step_distance_px": 2.0,
            "mean_radius_px": 5.0,
            "radius_std_px": 2.0,
            "start_radius_px": 4.0,
            "end_radius_px": 6.0,
            "radius_delta_px": 2.0,
        },
    ]

    aggregated = aggregate_segment_features(rows)

    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["fly_id"] == "sample__fly0"
    assert row["cohort"] == "intact"
    assert row["n_segments"] == 2
    assert row["n_segments_with_qc_flags"] == 1
    assert row["duration_frames_mean"] == pytest.approx(12.0)
    assert row["finite_frame_fraction_mean"] == pytest.approx(0.75)
    assert row["path_length_px_mean"] == pytest.approx(4.0)
    assert row["mean_radius_px_mean"] == pytest.approx(4.0)
