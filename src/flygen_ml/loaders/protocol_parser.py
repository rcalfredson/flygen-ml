from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flygen_ml.errors import MalformedRecordingError


def get_protocol(raw_data: dict[str, Any]) -> dict[str, Any]:
    protocol = raw_data.get("protocol")
    if not isinstance(protocol, dict):
        raise MalformedRecordingError("raw .data object is missing protocol dict")
    return protocol


def get_chamber_type(protocol: dict[str, Any]) -> str:
    chamber_type = protocol.get("ct")
    if not isinstance(chamber_type, str) or not chamber_type:
        raise MalformedRecordingError('protocol["ct"] is missing or invalid')
    return chamber_type


def get_experimental_fly_indices(protocol: dict[str, Any]) -> list[int]:
    frame_nums = protocol.get("frameNums")
    if isinstance(frame_nums, list):
        return [idx for idx, entry in enumerate(frame_nums) if bool(entry)]
    # TODO: fixture-verify that defaulting to fly 0 matches upstream expectations for non-list frameNums.
    return [0]


def get_selected_training_bounds(
    protocol: dict[str, Any],
    *,
    fly_idx: int,
    training_idx: int,
) -> tuple[int, int]:
    frame_nums = protocol.get("frameNums")
    if isinstance(frame_nums, list):
        if fly_idx < 0 or fly_idx >= len(frame_nums):
            raise MalformedRecordingError(
                f"fly_idx {fly_idx} out of range for frameNums with length {len(frame_nums)}"
            )
        fly_frame_nums = frame_nums[fly_idx]
    else:
        fly_frame_nums = frame_nums

    if not isinstance(fly_frame_nums, dict):
        raise MalformedRecordingError("selected frameNums entry is missing or is not a mapping")

    start_train = fly_frame_nums.get("startTrain")
    start_post = fly_frame_nums.get("startPost")
    if not isinstance(start_train, Sequence) or isinstance(start_train, (str, bytes)):
        raise MalformedRecordingError('selected frameNums entry is missing "startTrain" sequence')
    if not isinstance(start_post, Sequence) or isinstance(start_post, (str, bytes)):
        raise MalformedRecordingError('selected frameNums entry is missing "startPost" sequence')

    if training_idx < 0 or training_idx >= len(start_train):
        raise MalformedRecordingError(
            f"training_idx {training_idx} out of range for startTrain with length {len(start_train)}"
        )
    if training_idx < 0 or training_idx >= len(start_post):
        raise MalformedRecordingError(
            f"training_idx {training_idx} out of range for startPost with length {len(start_post)}"
        )

    return int(start_train[training_idx]), int(start_post[training_idx])
