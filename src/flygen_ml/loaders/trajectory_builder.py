from __future__ import annotations

from typing import Any

import numpy as np

from flygen_ml.loaders.protocol_parser import (
    get_chamber_type,
    get_experimental_fly_indices,
    get_protocol,
    get_selected_training_bounds,
)
from flygen_ml.schema import ManifestRow, NormalizedRecording


def infer_fps(timestamps: Any) -> float:
    if timestamps is None:
        return float("nan")
    ts = np.asarray(timestamps, dtype=float)
    if ts.ndim == 0 or ts.size < 2:
        return float("nan")
    deltas = np.diff(ts)
    finite_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if finite_deltas.size == 0:
        return float("nan")
    return float(1.0 / np.median(finite_deltas))


def build_normalized_recording(
    manifest_row: ManifestRow,
    raw_data: dict[str, Any],
    raw_trx: dict[str, Any],
) -> NormalizedRecording:
    protocol = get_protocol(raw_data)
    chamber_type = get_chamber_type(protocol)
    experimental_fly_indices = get_experimental_fly_indices(protocol)
    experimental_fly_idx = manifest_row.fly_idx if manifest_row.fly_idx is not None else experimental_fly_indices[0]
    training_start_frame, training_end_frame = get_selected_training_bounds(
        protocol,
        fly_idx=experimental_fly_idx,
        training_idx=manifest_row.training_idx,
    )
    fps = infer_fps(raw_trx.get("ts"))
    return NormalizedRecording(
        sample_key=manifest_row.sample_key,
        manifest=manifest_row,
        chamber_type=chamber_type,
        experimental_fly_idx=experimental_fly_idx,
        training_idx=manifest_row.training_idx,
        training_start_frame=training_start_frame,
        training_end_frame=training_end_frame,
        fps=fps,
        timestamps=raw_trx.get("ts"),
        x_by_fly=raw_trx.get("x"),
        y_by_fly=raw_trx.get("y"),
        protocol=protocol,
        raw_data=raw_data,
        raw_trx=raw_trx,
    )
