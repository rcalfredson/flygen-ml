from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flygen_ml.schema import NormalizedRecording


@dataclass(frozen=True)
class RewardEvents:
    actual_reward_frames: tuple[int, ...] = ()
    calculated_reward_frames: tuple[int, ...] = ()
    metadata: dict[str, Any] | None = None


def extract_reward_events(recording: NormalizedRecording) -> RewardEvents:
    del recording
    # TODO: implement fixture-verified extraction of actual and calculated reward streams.
    raise NotImplementedError(
        "Reward extraction is not implemented yet. "
        "Verify actual-vs-calculated reward semantics against trusted fixtures before implementing."
    )
