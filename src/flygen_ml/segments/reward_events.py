from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flygen_ml.errors import MalformedRecordingError
from flygen_ml.schema import NormalizedRecording

DEFAULT_BORDER_WIDTH_PX = 1.0


@dataclass(frozen=True)
class RewardEvents:
    actual_reward_frames: tuple[int, ...] = ()
    calculated_reward_frames: tuple[int, ...] = ()
    metadata: dict[str, Any] | None = None


def _true_regions(mask: np.ndarray) -> list[slice]:
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    stops = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1
    return [slice(int(s), int(e)) for s, e in zip(starts, stops)]


def _selected_fly_frame_nums(recording: NormalizedRecording) -> dict[str, Any]:
    frame_nums = recording.protocol.get("frameNums")
    fly_idx = recording.experimental_fly_idx
    if isinstance(frame_nums, list):
        if fly_idx < 0 or fly_idx >= len(frame_nums):
            raise MalformedRecordingError(f"experimental_fly_idx {fly_idx} out of range for frameNums")
        selected = frame_nums[fly_idx]
    else:
        selected = frame_nums
    if not isinstance(selected, dict):
        raise MalformedRecordingError("selected frameNums entry is missing or is not a mapping")
    return selected


def _selected_fly_info(recording: NormalizedRecording) -> dict[str, Any]:
    info = recording.protocol.get("info")
    fly_idx = recording.experimental_fly_idx
    if isinstance(info, list):
        if fly_idx < 0 or fly_idx >= len(info):
            raise MalformedRecordingError(f"experimental_fly_idx {fly_idx} out of range for info")
        selected = info[fly_idx]
    else:
        selected = info
    if not isinstance(selected, dict):
        raise MalformedRecordingError("selected info entry is missing or is not a mapping")
    return selected


def _reward_circle_for_training(recording: NormalizedRecording) -> tuple[float, float, float]:
    fly_info = _selected_fly_info(recording)
    training_idx = recording.training_idx
    cpos = fly_info.get("cPos")
    radii = fly_info.get("r")
    if not isinstance(cpos, (list, tuple)) or training_idx >= len(cpos):
        raise MalformedRecordingError('selected info entry is missing "cPos" sequence for training')
    if not isinstance(radii, (list, tuple)) or training_idx >= len(radii):
        raise MalformedRecordingError('selected info entry is missing "r" sequence for training')
    cx, cy = cpos[training_idx]
    return float(cx), float(cy), float(radii[training_idx])


def _actual_reward_keys(frame_nums: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in frame_nums:
        if key.startswith("v") and key[1:].isdigit() and key != "v0":
            keys.append(key)
    return sorted(keys)


def _filter_frames_to_training(frames: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.asarray(frames[(frames >= start) & (frames < stop)], dtype=int)


def calc_in_circle(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    *,
    border_width_px: float = DEFAULT_BORDER_WIDTH_PX,
) -> np.ndarray:
    dc = np.linalg.norm(np.vstack((x - cx, y - cy)), axis=0)
    return (dc < radius_px).astype(int) + (dc < radius_px + border_width_px)


def calc_en_ex(in_circle: np.ndarray, *, start: int, mode: str = "en") -> np.ndarray:
    idxs = np.arange(len(in_circle))[in_circle != 1]
    sign = 1 if mode == "en" else -1
    return idxs[np.flatnonzero(np.diff(in_circle[in_circle != 1]) == sign * 2) + 1] + start


def _calculated_reward_frames(recording: NormalizedRecording) -> np.ndarray:
    cx, cy, radius = _reward_circle_for_training(recording)
    fly_idx = recording.experimental_fly_idx
    start = recording.training_start_frame
    stop = recording.training_end_frame
    x = np.asarray(recording.x_by_fly[fly_idx], dtype=float)[start:stop].copy()
    y = np.asarray(recording.y_by_fly[fly_idx], dtype=float)[start:stop].copy()
    in_circle = calc_in_circle(x, y, cx, cy, radius)
    nan_mask = np.isnan(x) | np.isnan(y)
    for region in _true_regions(nan_mask):
        in_circle[region] = in_circle[region.start - 1] if region.start > 0 else 0
    return np.asarray(calc_en_ex(in_circle, start=start, mode="en"), dtype=int)


def _actual_reward_frames(recording: NormalizedRecording) -> np.ndarray:
    frame_nums = _selected_fly_frame_nums(recording)
    actual_keys = _actual_reward_keys(frame_nums)
    if not actual_keys:
        return np.array([], dtype=int)
    frames = [
        np.asarray(frame_nums[key], dtype=int)
        for key in actual_keys
        if isinstance(frame_nums.get(key), (list, tuple, np.ndarray))
    ]
    if not frames:
        return np.array([], dtype=int)
    merged = np.sort(np.concatenate(frames))
    return _filter_frames_to_training(
        merged,
        start=recording.training_start_frame,
        stop=recording.training_end_frame,
    )


def extract_reward_events(recording: NormalizedRecording) -> RewardEvents:
    actual_reward_frames = _actual_reward_frames(recording)
    calculated_reward_frames = _calculated_reward_frames(recording)
    cx, cy, radius = _reward_circle_for_training(recording)
    frame_nums = _selected_fly_frame_nums(recording)
    return RewardEvents(
        actual_reward_frames=tuple(int(frame) for frame in actual_reward_frames),
        calculated_reward_frames=tuple(int(frame) for frame in calculated_reward_frames),
        metadata={
            "reward_center_x": cx,
            "reward_center_y": cy,
            "reward_radius": radius,
            "actual_reward_keys": _actual_reward_keys(frame_nums),
            "border_width_px": DEFAULT_BORDER_WIDTH_PX,
        },
    )
