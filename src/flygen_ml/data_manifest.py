from __future__ import annotations

import csv
from pathlib import Path

from flygen_ml.schema import ManifestRow


REQUIRED_COLUMNS = (
    "sample_key",
    "data_path",
    "trx_path",
    "genotype",
    "chamber",
    "training_idx",
)


def load_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest missing required columns: {missing}")
        for row in reader:
            rows.append(
                ManifestRow(
                    sample_key=row["sample_key"],
                    data_path=Path(row["data_path"]),
                    trx_path=Path(row["trx_path"]),
                    genotype=row["genotype"],
                    chamber=row["chamber"],
                    training_idx=int(row["training_idx"]),
                    cohort=row.get("cohort") or None,
                    date=row.get("date") or None,
                    fly_idx=int(row["fly_idx"]) if row.get("fly_idx") else None,
                )
            )
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list[ManifestRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.sample_key in seen:
            raise ValueError(f"duplicate sample_key: {row.sample_key}")
        seen.add(row.sample_key)
        if row.data_path.suffix != ".data":
            raise ValueError(f"{row.sample_key}: expected .data path, got {row.data_path}")
        if row.trx_path.suffix != ".trx":
            raise ValueError(f"{row.sample_key}: expected .trx path, got {row.trx_path}")


def write_manifest(path: str | Path, rows: list[ManifestRow]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "data_path",
                "trx_path",
                "genotype",
                "cohort",
                "date",
                "chamber",
                "training_idx",
                "fly_idx",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_key": row.sample_key,
                    "data_path": str(row.data_path),
                    "trx_path": str(row.trx_path),
                    "genotype": row.genotype,
                    "cohort": row.cohort or "",
                    "date": row.date or "",
                    "chamber": row.chamber,
                    "training_idx": row.training_idx,
                    "fly_idx": "" if row.fly_idx is None else row.fly_idx,
                }
            )
