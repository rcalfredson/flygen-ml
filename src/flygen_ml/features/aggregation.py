from __future__ import annotations

import math


AGGREGATED_FEATURE_NAMES = (
    "duration_frames",
    "finite_frame_fraction",
    "path_length_px",
    "net_displacement_px",
    "straightness",
    "mean_step_distance_px",
    "mean_radius_px",
    "radius_std_px",
    "start_radius_px",
    "end_radius_px",
    "radius_delta_px",
)


def _numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values

def _assert_constant_metadata(fly_id, fly_rows, key) -> None:
    values = {str(row.get(key, "")) for row in fly_rows}
    if len(values) > 1:
        raise ValueError(f"fly {fly_id!r} has inconsistent {key}: {sorted(values)}")


def aggregate_segment_features(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        fly_id = str(row["fly_id"])
        grouped.setdefault(fly_id, []).append(row)

    aggregated_rows: list[dict[str, object]] = []
    for fly_id, fly_rows in sorted(grouped.items()):
        for key in ("sample_key", "genotype", "cohort", "chamber_type", "training_idx"):
            _assert_constant_metadata(fly_id, fly_rows, key)
        first = fly_rows[0]
        aggregated: dict[str, object] = {
            "fly_id": fly_id,
            "sample_key": first["sample_key"],
            "genotype": first["genotype"],
            "cohort": first.get("cohort", ""),
            "chamber_type": first["chamber_type"],
            "training_idx": first["training_idx"],
            "n_segments": len(fly_rows),
            "n_segments_with_qc_flags": sum(bool(row.get("qc_flags")) for row in fly_rows),
        }
        for feature_name in AGGREGATED_FEATURE_NAMES:
            values = _numeric_values(fly_rows, feature_name)
            aggregated[f"{feature_name}_mean"] = sum(values) / len(values) if values else float("nan")
        aggregated_rows.append(aggregated)
    return aggregated_rows
