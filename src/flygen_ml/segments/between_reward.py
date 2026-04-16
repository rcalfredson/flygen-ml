from __future__ import annotations

from flygen_ml.schema import NormalizedRecording, SegmentRecord
from flygen_ml.segments.reward_events import RewardEvents


def extract_between_reward_segments(
    recording: NormalizedRecording,
    reward_events: RewardEvents,
) -> list[SegmentRecord]:
    del recording
    del reward_events
    # TODO: implement the canonical v1 segment rule after fixture verification.
    raise NotImplementedError(
        "Between-reward segment extraction is not implemented yet. "
        "Verify anchor, exit, and stop-frame semantics against trusted fixtures before implementing."
    )
