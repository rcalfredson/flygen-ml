from __future__ import annotations

from typing import Any


def get_protocol(raw_data: dict[str, Any]) -> dict[str, Any]:
    protocol = raw_data.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("raw .data object is missing protocol dict")
    return protocol


def get_chamber_type(protocol: dict[str, Any]) -> str:
    chamber_type = protocol.get("ct")
    if not isinstance(chamber_type, str) or not chamber_type:
        raise ValueError('protocol["ct"] is missing or invalid')
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
    # TODO: replace this placeholder with fixture-backed parsing of training boundaries.
    raise NotImplementedError(
        "Training-boundary parsing is not implemented yet. "
        "Verify selected-training semantics against trusted fixtures before implementing."
    )
