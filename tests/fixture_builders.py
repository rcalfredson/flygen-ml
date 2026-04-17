from __future__ import annotations

from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.protocol_parser import (
    get_experimental_fly_indices,
    get_protocol,
    get_selected_training_bounds,
)
from flygen_ml.schema import ManifestRow, NormalizedRecording

from tests.fixture_registry import optional_expected_value


def build_fixture_recording(fixture) -> tuple[NormalizedRecording, dict]:
    raw_data, raw_trx = load_recording_pair(fixture.data_path, fixture.trx_path)
    protocol = get_protocol(raw_data)
    experimental_fly_indices = get_experimental_fly_indices(protocol)
    fly_idx = experimental_fly_indices[0]

    training_number = optional_expected_value(fixture, "training_number")
    if training_number is None:
        raise ValueError("fixture is missing expected training_number")
    training_idx = training_number - 1

    training_start_frame, training_end_frame = get_selected_training_bounds(
        protocol,
        fly_idx=fly_idx,
        training_idx=training_idx,
    )

    recording = NormalizedRecording(
        sample_key=fixture.sample_key,
        manifest=ManifestRow(
            sample_key=fixture.sample_key,
            data_path=fixture.data_path,
            trx_path=fixture.trx_path,
            genotype="unknown",
            chamber=optional_expected_value(fixture, "protocol_ct") or "unknown",
            training_idx=training_idx,
            fly_idx=fly_idx,
        ),
        chamber_type=optional_expected_value(fixture, "protocol_ct") or "unknown",
        experimental_fly_idx=fly_idx,
        training_idx=training_idx,
        training_start_frame=training_start_frame,
        training_end_frame=training_end_frame,
        fps=float("nan"),
        timestamps=raw_trx.get("ts"),
        x_by_fly=raw_trx.get("x"),
        y_by_fly=raw_trx.get("y"),
        protocol=protocol,
        raw_data=raw_data,
        raw_trx=raw_trx,
    )
    context = {
        "raw_data": raw_data,
        "raw_trx": raw_trx,
        "protocol": protocol,
        "fly_idx": fly_idx,
        "training_idx": training_idx,
        "training_start_frame": training_start_frame,
        "training_end_frame": training_end_frame,
    }
    return recording, context
