from __future__ import annotations

import pytest

from flygen_ml.segments.between_reward import extract_between_reward_segments
from flygen_ml.segments.reward_events import extract_reward_events

from tests.fixture_builders import build_fixture_recording
from tests.fixture_registry import optional_expected_value, require_available_fixture


def _segment_preview(segments: list) -> list[dict[str, int | None]]:
    preview: list[dict[str, int | None]] = []
    for segment in segments[:5]:
        preview.append(
            {
                "anchor_reward_frame": segment.anchor_reward_frame,
                "start_frame": segment.start_frame,
                "stop_frame": segment.stop_frame,
                "end_reward_frame": segment.end_reward_frame,
            }
        )
    return preview


def test_between_reward_segment_extraction_fixture():
    fixture = require_available_fixture()
    recording, _ = build_fixture_recording(fixture)
    reward_events = extract_reward_events(recording)
    segments = extract_between_reward_segments(recording, reward_events)

    expected_count = optional_expected_value(fixture, "canonical_segment_count")
    expected_preview = optional_expected_value(fixture, "canonical_segment_preview")

    if expected_count is None or expected_preview is None:
        pytest.fail(
            "Fill in canonical_segment_count and canonical_segment_preview "
            "for this fixture in tests/fixtures/local_fixture_registry.json.\n"
            f"Observed canonical segment count: {len(segments)}\n"
            f"Observed canonical segment preview: {_segment_preview(segments)}"
        )

    assert len(segments) == expected_count
    assert _segment_preview(segments)[: len(expected_preview)] == expected_preview
