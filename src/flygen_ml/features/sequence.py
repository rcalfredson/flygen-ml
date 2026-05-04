from __future__ import annotations

from pathlib import Path

import numpy as np

from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.trajectory_builder import build_normalized_recording
from flygen_ml.schema import ManifestRow, SegmentRecord, SequenceSample


DEFAULT_SEQUENCE_CHANNELS = ("x_rel", "y_rel", "dx_rel", "dy_rel", "speed_rel", "r_rel")


def _manifest_row_from_segment(segment: SegmentRecord) -> ManifestRow:
    return ManifestRow(
        sample_key=segment.sample_key,
        data_path=segment.data_path,
        trx_path=segment.trx_path,
        genotype=segment.genotype,
        cohort=segment.cohort,
        chamber=segment.chamber_type,
        training_idx=segment.training_idx,
        fly_idx=segment.experimental_fly_idx,
    )


def _segment_xy(recording, segment: SegmentRecord) -> tuple[np.ndarray, np.ndarray]:
    fly_idx = segment.experimental_fly_idx
    x = np.asarray(recording.x_by_fly[fly_idx], dtype=float)[segment.start_frame : segment.stop_frame]
    y = np.asarray(recording.y_by_fly[fly_idx], dtype=float)[segment.start_frame : segment.stop_frame]
    return x, y


def _interp_channel(values: np.ndarray, sample_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if np.count_nonzero(finite) == 0:
        return np.zeros(sample_positions.shape, dtype=np.float32), np.zeros(sample_positions.shape, dtype=bool)
    positions = np.arange(values.size, dtype=float)
    finite_positions = positions[finite]
    finite_values = values[finite]
    interpolated = np.interp(sample_positions, finite_positions, finite_values)
    mask = (sample_positions >= finite_positions[0]) & (sample_positions <= finite_positions[-1])
    return interpolated.astype(np.float32), mask


def _resample_channels(
    channel_values: dict[str, np.ndarray],
    *,
    target_length: int,
    channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if target_length < 2:
        raise ValueError(f"target_length must be at least 2, got {target_length}")
    n_frames = len(next(iter(channel_values.values())))
    if n_frames == 0:
        return (
            np.zeros((target_length, len(channels)), dtype=np.float32),
            np.zeros(target_length, dtype=bool),
        )
    sample_positions = np.linspace(0, n_frames - 1, num=target_length, dtype=float)
    x = np.zeros((target_length, len(channels)), dtype=np.float32)
    masks: list[np.ndarray] = []
    for channel_idx, channel in enumerate(channels):
        values, mask = _interp_channel(channel_values[channel], sample_positions)
        x[:, channel_idx] = values
        masks.append(mask)
    return x, np.logical_and.reduce(masks)


def build_sequence_sample(
    recording,
    segment: SegmentRecord,
    *,
    target_length: int = 128,
    channels: tuple[str, ...] = DEFAULT_SEQUENCE_CHANNELS,
) -> SequenceSample:
    x_abs, y_abs = _segment_xy(recording, segment)
    center_x = float(segment.reward_center_x or 0.0)
    center_y = float(segment.reward_center_y or 0.0)
    radius = float(segment.reward_radius or 1.0)
    if radius <= 0:
        radius = 1.0

    x_rel = (x_abs - center_x) / radius
    y_rel = (y_abs - center_y) / radius
    dx_rel = np.concatenate(([0.0], np.diff(x_rel))) if x_rel.size else np.array([], dtype=float)
    dy_rel = np.concatenate(([0.0], np.diff(y_rel))) if y_rel.size else np.array([], dtype=float)
    speed_rel = np.sqrt(dx_rel**2 + dy_rel**2)
    r_rel = np.sqrt(x_rel**2 + y_rel**2)
    channel_values = {
        "x_rel": x_rel,
        "y_rel": y_rel,
        "dx_rel": dx_rel,
        "dy_rel": dy_rel,
        "speed_rel": speed_rel,
        "r_rel": r_rel,
    }
    missing = [channel for channel in channels if channel not in channel_values]
    if missing:
        raise ValueError(f"unsupported sequence channels: {missing}")
    tensor, mask = _resample_channels(channel_values, target_length=target_length, channels=channels)
    return SequenceSample(
        segment_id=segment.segment_id,
        sample_key=segment.sample_key,
        fly_id=segment.fly_id,
        genotype=segment.genotype,
        group_id=segment.fly_id,
        channels=channels,
        length=target_length,
        x=tensor,
        mask=mask,
    )


def build_sequence_samples_from_segments(
    segments: list[SegmentRecord],
    *,
    target_length: int = 128,
    channels: tuple[str, ...] = DEFAULT_SEQUENCE_CHANNELS,
) -> list[SequenceSample]:
    samples: list[SequenceSample] = []
    recordings_by_sample: dict[str, object] = {}
    for segment in segments:
        recording = recordings_by_sample.get(segment.sample_key)
        if recording is None:
            manifest_row = _manifest_row_from_segment(segment)
            raw_data, raw_trx = load_recording_pair(segment.data_path, segment.trx_path)
            recording = build_normalized_recording(manifest_row, raw_data, raw_trx)
            recordings_by_sample[segment.sample_key] = recording
        samples.append(build_sequence_sample(recording, segment, target_length=target_length, channels=channels))
    return samples


def write_sequence_npz(
    path: str | Path,
    segments: list[SegmentRecord],
    *,
    target_length: int = 128,
    channels: tuple[str, ...] = DEFAULT_SEQUENCE_CHANNELS,
) -> None:
    samples = build_sequence_samples_from_segments(segments, target_length=target_length, channels=channels)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if samples:
        x = np.stack([np.asarray(sample.x, dtype=np.float32) for sample in samples])
        mask = np.stack([np.asarray(sample.mask, dtype=bool) for sample in samples])
    else:
        x = np.zeros((0, target_length, len(channels)), dtype=np.float32)
        mask = np.zeros((0, target_length), dtype=bool)
    np.savez_compressed(
        out_path,
        x=x,
        mask=mask,
        segment_id=np.asarray([sample.segment_id for sample in samples]),
        sample_key=np.asarray([sample.sample_key for sample in samples]),
        fly_id=np.asarray([sample.fly_id for sample in samples]),
        genotype=np.asarray([sample.genotype for sample in samples]),
        cohort=np.asarray([segment.cohort or "" for segment in segments]),
        qc_flags=np.asarray(["|".join(segment.qc_flags) for segment in segments]),
        terminated_by_training_end=np.asarray([segment.terminated_by_training_end for segment in segments]),
        channels=np.asarray(channels),
        target_length=np.asarray(target_length),
    )
