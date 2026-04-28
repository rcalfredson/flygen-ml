from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path

from flygen_ml.schema import ManifestRow


@dataclass(frozen=True)
class ManifestGlobSpec:
    genotype: str
    cohort: str | None
    chamber: str
    training_idx: int
    patterns: tuple[str, ...]
    fly_idx: int | None = None


def load_manifest_glob_specs(path: str | Path) -> list[ManifestGlobSpec]:
    specs: list[ManifestGlobSpec] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"genotype", "chamber", "training_idx", "patterns"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest glob spec missing required columns: {sorted(missing)}")
        for row in reader:
            patterns = tuple(pattern.strip() for pattern in row["patterns"].split(",") if pattern.strip())
            if not patterns:
                raise ValueError(f"empty patterns for genotype {row['genotype']!r}")
            specs.append(
                ManifestGlobSpec(
                    genotype=row["genotype"],
                    cohort=row.get("cohort") or None,
                    chamber=row["chamber"],
                    training_idx=int(row["training_idx"]),
                    patterns=patterns,
                    fly_idx=int(row["fly_idx"]) if row.get("fly_idx") else None,
                )
            )
    return specs


def _sample_key_for_stem(stem_path: Path) -> str:
    sanitized = str(stem_path)
    sanitized = sanitized.lstrip("/")
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.replace("/", "__")
    return sanitized


def _sample_key_for_stem_and_fly(stem_path: Path, fly_idx: int | None) -> str:
    base = _sample_key_for_stem(stem_path)
    if fly_idx is None:
        return base
    return f"{base}__fly{fly_idx}"


def _stem_from_match(match_path: str) -> Path:
    path = Path(match_path)
    if path.suffix:
        return path.with_suffix("")
    return path


def _discover_paired_stems(patterns: tuple[str, ...]) -> list[Path]:
    stems: dict[str, Path] = {}
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            stem = _stem_from_match(match)
            data_path = stem.with_suffix(".data")
            trx_path = stem.with_suffix(".trx")
            if data_path.exists() and trx_path.exists():
                stems[str(stem)] = stem
    return [stems[key] for key in sorted(stems)]


def build_manifest_rows_from_glob_specs(
    specs: list[ManifestGlobSpec],
    *,
    repeated_fly_indices: tuple[int, ...] | None = None,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    seen_sample_keys: set[str] = set()
    for spec in specs:
        stems = _discover_paired_stems(spec.patterns)
        if not stems:
            raise ValueError(f"no paired .data/.trx stems found for genotype {spec.genotype!r}")
        for stem in stems:
            fly_indices: tuple[int | None, ...]
            if repeated_fly_indices is not None:
                fly_indices = repeated_fly_indices
            else:
                fly_indices = (spec.fly_idx,)
            for fly_idx in fly_indices:
                sample_key = _sample_key_for_stem_and_fly(stem, fly_idx)
                if sample_key in seen_sample_keys:
                    raise ValueError(f"duplicate sample_key generated from glob spec: {sample_key}")
                seen_sample_keys.add(sample_key)
                rows.append(
                    ManifestRow(
                        sample_key=sample_key,
                        data_path=stem.with_suffix(".data"),
                        trx_path=stem.with_suffix(".trx"),
                        genotype=spec.genotype,
                        chamber=spec.chamber,
                        training_idx=spec.training_idx,
                        cohort=spec.cohort,
                        date=_extract_date_from_stem(stem),
                        fly_idx=fly_idx,
                    )
                )
    return rows


def _extract_date_from_stem(stem: Path) -> str | None:
    for part in stem.parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return None
