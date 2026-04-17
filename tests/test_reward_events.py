from __future__ import annotations

import pytest

from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.protocol_parser import (
    get_experimental_fly_indices,
    get_protocol,
    get_selected_training_bounds,
)
from flygen_ml.schema import ManifestRow, NormalizedRecording
from flygen_ml.segments.reward_events import extract_reward_events

from tests.fixture_registry import optional_expected_value, require_available_fixture


def _preview_value(value: object, *, max_items: int = 5) -> str:
    if isinstance(value, (list, tuple)):
        preview = list(value[:max_items])
        suffix = "" if len(value) <= max_items else " ..."
        return f"{type(value).__name__}(len={len(value)}): {preview}{suffix}"
    return repr(value)


def _summarize_reward_metadata(protocol: dict, *, fly_idx: int, training_idx: int) -> str:
    lines: list[str] = []

    circle = protocol.get("circle")
    lines.append(f'protocol["circle"] type: {type(circle).__name__}')
    if isinstance(circle, dict):
        lines.append(f'protocol["circle"] keys: {sorted(circle.keys())}')
        for key in ("r", "pos"):
            if key in circle:
                lines.append(f'circle[{key!r}]: {_preview_value(circle[key])}')

    info = protocol.get("info")
    lines.append(f'protocol["info"] type: {type(info).__name__}')
    if isinstance(info, list) and 0 <= fly_idx < len(info):
        fly_info = info[fly_idx]
        lines.append(f"selected fly info entry type: {type(fly_info).__name__}")
        if isinstance(fly_info, dict):
            lines.append(f"selected fly info keys: {sorted(fly_info.keys())}")
            for key in ("r", "cPos"):
                if key in fly_info:
                    lines.append(f'info[{key!r}]: {_preview_value(fly_info[key])}')
                    value = fly_info[key]
                    if isinstance(value, (list, tuple)) and 0 <= training_idx < len(value):
                        lines.append(f"{key}[training_idx={training_idx}]: {value[training_idx]!r}")

    return "\n".join(lines)


def _build_stub_recording(
    fixture,
    protocol: dict,
    raw_data: dict,
    raw_trx: dict,
    *,
    fly_idx: int,
    training_idx: int,
    training_start_frame: int,
    training_end_frame: int,
) -> NormalizedRecording:
    return NormalizedRecording(
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


def test_reward_event_extraction_fixture():
    fixture = require_available_fixture()
    raw_data, raw_trx = load_recording_pair(fixture.data_path, fixture.trx_path)
    protocol = get_protocol(raw_data)

    expected_training_number = optional_expected_value(fixture, "training_number")
    expected_start = optional_expected_value(fixture, "training_start_frame")
    expected_end = optional_expected_value(fixture, "training_end_frame")
    expected_count = optional_expected_value(fixture, "calculated_reward_count")
    expected_frames_preview = optional_expected_value(fixture, "calculated_reward_frames_preview")

    experimental_fly_indices = get_experimental_fly_indices(protocol)
    fly_idx = experimental_fly_indices[0]

    if any(value is None for value in (expected_training_number, expected_start, expected_end)):
        pytest.fail(
            "Reward-event test needs trusted training metadata first.\n"
            "Fill in training_number, training_start_frame, and training_end_frame "
            "for this fixture in tests/fixtures/local_fixture_registry.json."
        )

    training_idx = expected_training_number - 1

    training_start_frame, training_end_frame = get_selected_training_bounds(
        protocol,
        fly_idx=fly_idx,
        training_idx=training_idx,
    )
    assert training_start_frame == expected_start
    assert training_end_frame == expected_end

    recording = _build_stub_recording(
        fixture,
        protocol,
        raw_data,
        raw_trx,
        fly_idx=fly_idx,
        training_idx=training_idx,
        training_start_frame=training_start_frame,
        training_end_frame=training_end_frame,
    )
    reward_events = extract_reward_events(recording)

    if any(value is None for value in (expected_count, expected_frames_preview)):
        pytest.fail(
            "Fill in calculated_reward_count and calculated_reward_frames_preview "
            "for this fixture in tests/fixtures/local_fixture_registry.json.\n"
            "Reward metadata preview for choosing those values:\n"
            f"{_summarize_reward_metadata(protocol, fly_idx=fly_idx, training_idx=training_idx)}\n"
            f"Observed actual reward count: {len(reward_events.actual_reward_frames)}\n"
            f"Observed actual reward preview: {list(reward_events.actual_reward_frames[:10])}\n"
            f"Observed calculated reward count: {len(reward_events.calculated_reward_frames)}\n"
            f"Observed calculated reward preview: {list(reward_events.calculated_reward_frames[:10])}\n"
            f"Observed reward metadata: {reward_events.metadata}"
        )

    assert len(reward_events.calculated_reward_frames) == expected_count
    assert list(reward_events.calculated_reward_frames[: len(expected_frames_preview)]) == expected_frames_preview
