from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestRow:
    sample_key: str
    data_path: Path
    trx_path: Path
    genotype: str
    chamber: str
    training_idx: int
    cohort: str | None = None
    date: str | None = None
    fly_idx: int | None = None


@dataclass
class RecordingQC:
    flags: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedRecording:
    sample_key: str
    manifest: ManifestRow
    chamber_type: str
    experimental_fly_idx: int
    training_idx: int
    training_start_frame: int
    training_end_frame: int
    fps: float
    timestamps: Any
    x_by_fly: Any
    y_by_fly: Any
    protocol: dict[str, Any]
    raw_data: dict[str, Any]
    raw_trx: dict[str, Any]
    reward_context: dict[str, Any] = field(default_factory=dict)
    qc: RecordingQC = field(default_factory=RecordingQC)


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    sample_key: str
    fly_id: str
    genotype: str
    cohort: str | None
    chamber_type: str
    experimental_fly_idx: int
    data_path: Path
    trx_path: Path
    training_idx: int
    training_start_frame: int
    training_end_frame: int
    anchor_reward_frame: int
    start_frame: int
    stop_frame: int
    end_reward_frame: int | None
    duration_frames: int
    n_finite_frames: int
    finite_frame_fraction: float
    qc_flags: tuple[str, ...] = ()
    reward_center_x: float | None = None
    reward_center_y: float | None = None
    reward_radius: float | None = None
    terminated_by_training_end: bool = False
    anchor_reward_kind: str = "calculated"


@dataclass
class SequenceSample:
    segment_id: str
    sample_key: str
    fly_id: str
    genotype: str
    group_id: str
    channels: tuple[str, ...]
    length: int
    x: Any
    mask: Any
