from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TextIO

from flygen_ml.cli.build_features import _manifest_row_from_segment
from flygen_ml.features.engineered import compute_engineered_features
from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.trajectory_builder import build_normalized_recording
from flygen_ml.schema import SegmentRecord
from flygen_ml.segment_table import load_segment_table


SEGMENT_INSPECTION_BASE_COLUMNS = [
    "rank",
    "metric",
    "metric_value",
    "segment_id",
    "sample_key",
    "fly_id",
    "genotype",
    "training_idx",
    "anchor_reward_frame",
    "start_frame",
    "stop_frame",
    "end_reward_frame",
    "duration_frames",
    "terminated_by_training_end",
    "qc_flags",
    "reward_center_x",
    "reward_center_y",
    "reward_radius",
]


def build_segment_metric_rows(
    *,
    segments: list[SegmentRecord],
    metric: str,
    sample_key: str | None = None,
    fly_id: str | None = None,
    descending: bool = True,
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    recordings_by_sample: dict[str, object] = {}
    for segment in segments:
        if sample_key is not None and segment.sample_key != sample_key:
            continue
        if fly_id is not None and segment.fly_id != fly_id:
            continue

        recording = recordings_by_sample.get(segment.sample_key)
        if recording is None:
            raw_data, raw_trx = load_recording_pair(segment.data_path, segment.trx_path)
            recording = build_normalized_recording(_manifest_row_from_segment(segment), raw_data, raw_trx)
            recordings_by_sample[segment.sample_key] = recording

        features = compute_engineered_features(recording, segment)
        if metric not in features:
            raise ValueError(f"unknown engineered segment metric: {metric}")
        metric_value = float(features[metric])
        if not math.isfinite(metric_value):
            continue
        rows.append(
            {
                "metric": metric,
                "metric_value": metric_value,
                "segment_id": segment.segment_id,
                "sample_key": segment.sample_key,
                "fly_id": segment.fly_id,
                "genotype": segment.genotype,
                "training_idx": segment.training_idx,
                "anchor_reward_frame": segment.anchor_reward_frame,
                "start_frame": segment.start_frame,
                "stop_frame": segment.stop_frame,
                "end_reward_frame": "" if segment.end_reward_frame is None else segment.end_reward_frame,
                "duration_frames": segment.duration_frames,
                "terminated_by_training_end": segment.terminated_by_training_end,
                "qc_flags": "|".join(segment.qc_flags),
                "reward_center_x": "" if segment.reward_center_x is None else segment.reward_center_x,
                "reward_center_y": "" if segment.reward_center_y is None else segment.reward_center_y,
                "reward_radius": "" if segment.reward_radius is None else segment.reward_radius,
                **features,
            }
        )

    rows.sort(key=lambda row: float(row["metric_value"]), reverse=descending)
    if limit is not None:
        rows = rows[:limit]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def build_segment_metric_rows_from_table(
    *,
    segments_path: str | Path,
    metric: str,
    sample_key: str | None = None,
    fly_id: str | None = None,
    descending: bool = True,
    limit: int | None = None,
) -> list[dict[str, object]]:
    return build_segment_metric_rows(
        segments=load_segment_table(segments_path),
        metric=metric,
        sample_key=sample_key,
        fly_id=fly_id,
        descending=descending,
        limit=limit,
    )


def write_segment_metric_rows(rows: list[dict[str, object]], handle: TextIO) -> None:
    feature_columns = [
        name
        for name in rows[0].keys()
        if name not in SEGMENT_INSPECTION_BASE_COLUMNS
    ] if rows else []
    writer = csv.DictWriter(handle, fieldnames=SEGMENT_INSPECTION_BASE_COLUMNS + feature_columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
