from __future__ import annotations

import csv
from pathlib import Path

from flygen_ml.schema import SegmentRecord


SEGMENT_TABLE_COLUMNS = (
    "segment_id",
    "sample_key",
    "fly_id",
    "genotype",
    "cohort",
    "chamber_type",
    "experimental_fly_idx",
    "data_path",
    "trx_path",
    "training_idx",
    "training_start_frame",
    "training_end_frame",
    "anchor_reward_frame",
    "start_frame",
    "stop_frame",
    "end_reward_frame",
    "duration_frames",
    "n_finite_frames",
    "finite_frame_fraction",
    "qc_flags",
    "reward_center_x",
    "reward_center_y",
    "reward_radius",
    "terminated_by_training_end",
    "anchor_reward_kind",
)
OPTIONAL_SEGMENT_TABLE_COLUMNS = {"cohort"}


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else str(value)


def write_segment_table(path: str | Path, rows: list[SegmentRecord]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SEGMENT_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "segment_id": row.segment_id,
                    "sample_key": row.sample_key,
                    "fly_id": row.fly_id,
                    "genotype": row.genotype,
                    "cohort": row.cohort or "",
                    "chamber_type": row.chamber_type,
                    "experimental_fly_idx": row.experimental_fly_idx,
                    "data_path": str(row.data_path),
                    "trx_path": str(row.trx_path),
                    "training_idx": row.training_idx,
                    "training_start_frame": row.training_start_frame,
                    "training_end_frame": row.training_end_frame,
                    "anchor_reward_frame": row.anchor_reward_frame,
                    "start_frame": row.start_frame,
                    "stop_frame": row.stop_frame,
                    "end_reward_frame": _format_optional_int(row.end_reward_frame),
                    "duration_frames": row.duration_frames,
                    "n_finite_frames": row.n_finite_frames,
                    "finite_frame_fraction": row.finite_frame_fraction,
                    "qc_flags": "|".join(row.qc_flags),
                    "reward_center_x": _format_optional_float(row.reward_center_x),
                    "reward_center_y": _format_optional_float(row.reward_center_y),
                    "reward_radius": _format_optional_float(row.reward_radius),
                    "terminated_by_training_end": str(
                        row.terminated_by_training_end
                    ).lower(),
                    "anchor_reward_kind": row.anchor_reward_kind,
                }
            )


def _parse_optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _parse_optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def load_segment_table(path: str | Path) -> list[SegmentRecord]:
    rows: list[SegmentRecord] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            name
            for name in SEGMENT_TABLE_COLUMNS
            if name not in (reader.fieldnames or [])
            and name not in OPTIONAL_SEGMENT_TABLE_COLUMNS
        ]
        if missing:
            raise ValueError(f"segment table missing required columns: {missing}")
        for row in reader:
            qc_flags = tuple(flag for flag in row["qc_flags"].split("|") if flag)
            rows.append(
                SegmentRecord(
                    segment_id=row["segment_id"],
                    sample_key=row["sample_key"],
                    fly_id=row["fly_id"],
                    genotype=row["genotype"],
                    cohort=row.get("cohort") or None,
                    chamber_type=row["chamber_type"],
                    experimental_fly_idx=int(row["experimental_fly_idx"]),
                    data_path=Path(row["data_path"]),
                    trx_path=Path(row["trx_path"]),
                    training_idx=int(row["training_idx"]),
                    training_start_frame=int(row["training_start_frame"]),
                    training_end_frame=int(row["training_end_frame"]),
                    anchor_reward_frame=int(row["anchor_reward_frame"]),
                    start_frame=int(row["start_frame"]),
                    stop_frame=int(row["stop_frame"]),
                    end_reward_frame=_parse_optional_int(row["end_reward_frame"]),
                    duration_frames=int(row["duration_frames"]),
                    n_finite_frames=int(row["n_finite_frames"]),
                    finite_frame_fraction=float(row["finite_frame_fraction"]),
                    qc_flags=qc_flags,
                    reward_center_x=_parse_optional_float(row["reward_center_x"]),
                    reward_center_y=_parse_optional_float(row["reward_center_y"]),
                    reward_radius=_parse_optional_float(row["reward_radius"]),
                    terminated_by_training_end=row["terminated_by_training_end"]
                    .strip()
                    .lower()
                    == "true",
                    anchor_reward_kind=row["anchor_reward_kind"],
                )
            )
    return rows
