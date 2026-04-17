from __future__ import annotations

import numpy as np

from flygen_ml.schema import NormalizedRecording, SegmentRecord
from flygen_ml.segments.reward_events import RewardEvents, calc_en_ex, calc_in_circle


def _first_exit_after_anchor(
    recording: NormalizedRecording,
    reward_events: RewardEvents,
    *,
    anchor_frame: int,
    stop_frame: int,
) -> int | None:
    if reward_events.metadata is None:
        raise ValueError("reward_events.metadata is required for segment extraction")
    fly_idx = recording.experimental_fly_idx
    x = np.asarray(recording.x_by_fly[fly_idx], dtype=float)[anchor_frame:stop_frame].copy()
    y = np.asarray(recording.y_by_fly[fly_idx], dtype=float)[anchor_frame:stop_frame].copy()
    in_circle = calc_in_circle(
        x,
        y,
        reward_events.metadata["reward_center_x"],
        reward_events.metadata["reward_center_y"],
        reward_events.metadata["reward_radius"],
        border_width_px=reward_events.metadata.get("border_width_px", 1.0),
    )
    nan_mask = np.isnan(x) | np.isnan(y)
    if np.any(nan_mask):
        starts = np.flatnonzero(nan_mask & np.concatenate(([True], ~nan_mask[:-1])))
        stops = np.flatnonzero(nan_mask & np.concatenate((~nan_mask[1:], [True]))) + 1
        for start, stop in zip(starts, stops):
            in_circle[start:stop] = in_circle[start - 1] if start > 0 else 0
    exits = calc_en_ex(in_circle, start=anchor_frame, mode="ex")
    if len(exits) == 0:
        return None
    return int(exits[0])


def extract_between_reward_segments(
    recording: NormalizedRecording,
    reward_events: RewardEvents,
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    anchors = list(reward_events.calculated_reward_frames)
    training_end = recording.training_end_frame
    sample_key = recording.sample_key
    fly_id = f"{sample_key}__fly{recording.experimental_fly_idx}"
    genotype = recording.manifest.genotype

    for idx, anchor_frame in enumerate(anchors):
        next_reward_frame = int(anchors[idx + 1]) if idx + 1 < len(anchors) else None
        segment_stop = next_reward_frame if next_reward_frame is not None else training_end
        exit_frame = _first_exit_after_anchor(
            recording,
            reward_events,
            anchor_frame=int(anchor_frame),
            stop_frame=segment_stop,
        )
        if exit_frame is None:
            continue
        start_frame = int(exit_frame)
        stop_frame = int(segment_stop)
        if start_frame >= stop_frame:
            continue
        x = np.asarray(recording.x_by_fly[recording.experimental_fly_idx], dtype=float)[start_frame:stop_frame]
        y = np.asarray(recording.y_by_fly[recording.experimental_fly_idx], dtype=float)[start_frame:stop_frame]
        finite_mask = np.isfinite(x) & np.isfinite(y)
        n_finite_frames = int(np.count_nonzero(finite_mask))
        duration_frames = stop_frame - start_frame
        qc_flags: list[str] = []
        if n_finite_frames < duration_frames:
            qc_flags.append("has_missing_frames")
        if n_finite_frames == 0:
            qc_flags.append("no_finite_frames")
        if n_finite_frames < 2:
            qc_flags.append("insufficient_finite_frames")
        segments.append(
            SegmentRecord(
                segment_id=f"{sample_key}__tr{recording.training_idx}__seg{len(segments)}",
                sample_key=sample_key,
                fly_id=fly_id,
                genotype=genotype,
                chamber_type=recording.chamber_type,
                experimental_fly_idx=recording.experimental_fly_idx,
                data_path=recording.manifest.data_path,
                trx_path=recording.manifest.trx_path,
                training_idx=recording.training_idx,
                training_start_frame=recording.training_start_frame,
                training_end_frame=recording.training_end_frame,
                anchor_reward_frame=int(anchor_frame),
                start_frame=start_frame,
                stop_frame=stop_frame,
                end_reward_frame=next_reward_frame,
                duration_frames=duration_frames,
                n_finite_frames=n_finite_frames,
                finite_frame_fraction=n_finite_frames / duration_frames,
                qc_flags=tuple(qc_flags),
                reward_center_x=reward_events.metadata.get("reward_center_x") if reward_events.metadata else None,
                reward_center_y=reward_events.metadata.get("reward_center_y") if reward_events.metadata else None,
                reward_radius=reward_events.metadata.get("reward_radius") if reward_events.metadata else None,
                terminated_by_training_end=next_reward_frame is None,
            )
        )
    return segments
