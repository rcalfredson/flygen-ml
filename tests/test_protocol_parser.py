from __future__ import annotations

import pytest

from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.protocol_parser import (
    get_chamber_type,
    get_experimental_fly_indices,
    get_protocol,
    get_selected_training_bounds,
)

from tests.fixture_registry import optional_expected_value, require_available_fixture


def test_protocol_parser_reads_protocol_and_chamber_type():
    fixture = require_available_fixture()
    raw_data, _ = load_recording_pair(fixture.data_path, fixture.trx_path)
    protocol = get_protocol(raw_data)
    chamber_type = get_chamber_type(protocol)

    assert isinstance(protocol, dict)
    assert isinstance(chamber_type, str)
    assert chamber_type

    expected_ct = optional_expected_value(fixture, "protocol_ct")
    if expected_ct is not None:
        assert chamber_type == expected_ct


def _preview_value(value: object, *, max_items: int = 5) -> str:
    if isinstance(value, (list, tuple)):
        preview = list(value[:max_items])
        suffix = "" if len(value) <= max_items else " ..."
        return f"{type(value).__name__}(len={len(value)}): {preview}{suffix}"
    return repr(value)


def _summarize_training_metadata(protocol: dict) -> str:
    lines: list[str] = []
    frame_nums = protocol.get("frameNums")
    lines.append(f'protocol["frameNums"] type: {type(frame_nums).__name__}')

    experimental_fly_indices = get_experimental_fly_indices(protocol)
    lines.append(f"experimental_fly_indices: {experimental_fly_indices}")

    selected_fly_idx = experimental_fly_indices[0]
    if isinstance(frame_nums, list) and selected_fly_idx < len(frame_nums):
        selected_frame_nums = frame_nums[selected_fly_idx]
        lines.append(
            "selected experimental frameNums entry type: "
            f"{type(selected_frame_nums).__name__}"
        )
        if isinstance(selected_frame_nums, dict):
            lines.append(
                "selected frameNums keys: "
                f"{sorted(selected_frame_nums.keys())}"
            )
            for key in ("startPre", "startTrain", "startPost"):
                if key in selected_frame_nums:
                    lines.append(f'{key}: {_preview_value(selected_frame_nums[key])}')

    info = protocol.get("info")
    if isinstance(info, dict):
        lines.append(f'protocol["info"] keys: {sorted(info.keys())}')

    return "\n".join(lines)


def test_protocol_parser_training_bounds_for_selected_training():
    fixture = require_available_fixture()
    raw_data, _ = load_recording_pair(fixture.data_path, fixture.trx_path)
    protocol = get_protocol(raw_data)

    expected_training_number = optional_expected_value(fixture, "training_number")
    expected_start = optional_expected_value(fixture, "training_start_frame")
    expected_end = optional_expected_value(fixture, "training_end_frame")

    if any(value is None for value in (expected_training_number, expected_start, expected_end)):
        pytest.fail(
            "Fill in training_number, training_start_frame, and training_end_frame "
            "for this fixture in tests/fixtures/local_fixture_registry.json.\n"
            "Protocol preview for choosing those values:\n"
            f"{_summarize_training_metadata(protocol)}"
        )

    experimental_fly_indices = get_experimental_fly_indices(protocol)
    # Fixture metadata uses human-facing training numbering; parser APIs use zero-based indices.
    expected_training_idx = expected_training_number - 1
    training_start_frame, training_end_frame = get_selected_training_bounds(
        protocol,
        fly_idx=experimental_fly_indices[0],
        training_idx=expected_training_idx,
    )

    assert training_start_frame == expected_start
    assert training_end_frame == expected_end
