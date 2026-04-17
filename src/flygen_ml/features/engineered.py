from __future__ import annotations

import numpy as np

from flygen_ml.schema import NormalizedRecording, SegmentRecord


def _segment_xy(recording: NormalizedRecording, segment: SegmentRecord) -> tuple[np.ndarray, np.ndarray]:
    fly_idx = segment.experimental_fly_idx
    x = np.asarray(recording.x_by_fly[fly_idx], dtype=float)[segment.start_frame:segment.stop_frame]
    y = np.asarray(recording.y_by_fly[fly_idx], dtype=float)[segment.start_frame:segment.stop_frame]
    return x, y


def _finite_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    return x[finite_mask], y[finite_mask]


def _step_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.array([], dtype=float)
    finite_pairs = np.isfinite(x[:-1]) & np.isfinite(y[:-1]) & np.isfinite(x[1:]) & np.isfinite(y[1:])
    dx = np.diff(x)[finite_pairs]
    dy = np.diff(y)[finite_pairs]
    if dx.size == 0:
        return np.array([], dtype=float)
    return np.sqrt(dx**2 + dy**2)


def compute_engineered_features(recording: NormalizedRecording, segment: SegmentRecord) -> dict[str, float]:
    x, y = _segment_xy(recording, segment)
    finite_x, finite_y = _finite_points(x, y)
    step_distances = _step_distances(x, y)

    if finite_x.size == 0 or segment.reward_center_x is None or segment.reward_center_y is None:
        radius = np.array([], dtype=float)
    else:
        radius = np.sqrt((finite_x - segment.reward_center_x) ** 2 + (finite_y - segment.reward_center_y) ** 2)

    path_length = float(step_distances.sum()) if step_distances.size else 0.0
    if finite_x.size >= 2:
        net_displacement = float(
            np.sqrt((finite_x[-1] - finite_x[0]) ** 2 + (finite_y[-1] - finite_y[0]) ** 2)
        )
    else:
        net_displacement = 0.0

    mean_radius = float(radius.mean()) if radius.size else float("nan")
    radius_std = float(radius.std()) if radius.size else float("nan")
    start_radius = float(radius[0]) if radius.size else float("nan")
    end_radius = float(radius[-1]) if radius.size else float("nan")

    return {
        "duration_frames": float(segment.duration_frames),
        "finite_frame_fraction": float(segment.finite_frame_fraction),
        "path_length_px": path_length,
        "net_displacement_px": net_displacement,
        "straightness": net_displacement / path_length if path_length > 0 else 0.0,
        "mean_step_distance_px": float(step_distances.mean()) if step_distances.size else 0.0,
        "mean_radius_px": mean_radius,
        "radius_std_px": radius_std,
        "start_radius_px": start_radius,
        "end_radius_px": end_radius,
        "radius_delta_px": end_radius - start_radius if radius.size else float("nan"),
    }
