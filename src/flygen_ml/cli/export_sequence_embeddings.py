from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from flygen_ml.cli.explain_sequence_occlusion import _filter_examples, _prediction_fly_ids
from flygen_ml.modeling.sequence_models import load_sequence_npz
from flygen_ml.modeling.torch_sequence_models import (
    export_torch_sequence_embeddings,
    load_torch_sequence_model_artifact,
)


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _load_prediction_rows(run_dir: Path) -> list[dict[str, str]]:
    prediction_path = run_dir / "predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing holdout prediction table: {prediction_path}")
    with prediction_path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _prediction_rows_by_fly(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        fly_id = row.get("fly_id", "")
        if fly_id and fly_id not in indexed:
            indexed[fly_id] = row
    return indexed


def _sequence_metadata_by_index(sequence_path: str | Path) -> dict[int, dict[str, object]]:
    payload = np.load(sequence_path)
    n_segments = int(payload["x"].shape[0])

    def _strings(key: str, default: str = "") -> list[str]:
        if key not in payload:
            return [default] * n_segments
        return [str(value) for value in payload[key]]

    def _bools(key: str, default: bool = False) -> list[bool]:
        if key not in payload:
            return [default] * n_segments
        return [bool(value) for value in payload[key]]

    segment_ids = _strings("segment_id")
    sample_keys = _strings("sample_key")
    fly_ids = _strings("fly_id")
    genotypes = _strings("genotype")
    cohorts = _strings("cohort")
    qc_flags = _strings("qc_flags")
    terminated_by_training_end = _bools("terminated_by_training_end")
    return {
        idx: {
            "segment_id": segment_ids[idx],
            "sample_key": sample_keys[idx],
            "fly_id": fly_ids[idx],
            "genotype": genotypes[idx],
            "cohort": cohorts[idx],
            "qc_flags": qc_flags[idx],
            "terminated_by_training_end": terminated_by_training_end[idx],
        }
        for idx in range(n_segments)
    }


def _segments_by_id(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with Path(path).open("r", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        segment_id = row.get("segment_id", "")
        if not segment_id:
            continue
        if segment_id in indexed:
            duplicates.append(segment_id)
        indexed[segment_id] = row
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"segment table has duplicate segment_id values: {preview}")
    return indexed


def _prediction_metadata(row: dict[str, str] | None, split: str) -> dict[str, object]:
    if row is None:
        return {
            "split": split,
            "actual_genotype": "",
            "predicted_genotype": "",
            "genotype_probability": "",
            "actual_cohort": "",
            "predicted_cohort": "",
            "cohort_probability": "",
            "both_correct": "",
        }
    return {
        "split": row.get("split", split),
        "actual_genotype": row.get("actual_genotype", ""),
        "predicted_genotype": row.get("predicted_genotype", ""),
        "genotype_probability": row.get("genotype_probability", ""),
        "actual_cohort": row.get("actual_cohort", ""),
        "predicted_cohort": row.get("predicted_cohort", ""),
        "cohort_probability": row.get("cohort_probability", ""),
        "both_correct": row.get("both_correct", ""),
    }


def _source_metadata(segment_row: dict[str, str] | None) -> dict[str, str]:
    if segment_row is None:
        return {}
    return {
        f"source_{key}": value
        for key, value in segment_row.items()
        if key != "segment_id"
    }


def _enrich_rows(
    rows: list[dict[str, object]],
    *,
    sequence_metadata: dict[int, dict[str, object]],
    predictions_by_fly: dict[str, dict[str, str]],
    segment_rows_by_id: dict[str, dict[str, str]],
    split: str,
) -> list[dict[str, object]]:
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        segment_index = int(row["segment_index"])
        sequence_row = sequence_metadata[segment_index]
        segment_id = str(sequence_row["segment_id"])
        prediction_row = predictions_by_fly.get(str(row["fly_id"]))
        enriched_rows.append(
            {
                **row,
                **sequence_row,
                **_prediction_metadata(prediction_row, split),
                **_source_metadata(segment_rows_by_id.get(segment_id)),
            }
        )
    return enriched_rows


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "segment_index",
        "segment_id",
        "fly_id",
        "sample_key",
        "genotype",
        "cohort",
        "split",
        "segment_position_in_fly",
        "eval_position",
        "selected_for_model_eval",
        "actual_genotype",
        "predicted_genotype",
        "genotype_probability",
        "actual_cohort",
        "predicted_cohort",
        "cohort_probability",
        "both_correct",
        "n_segments",
        "n_segments_with_qc_flags",
        "qc_flags",
        "terminated_by_training_end",
    ]
    ordered: list[str] = []
    for key in preferred:
        if any(key in row for row in rows):
            ordered.append(key)
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows) if rows else ["segment_index", "segment_id", "fly_id", "sample_key"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _string_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([str(row.get(key, "")) for row in rows])


def _int_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([int(row.get(key, 0)) for row in rows], dtype=np.int64)


def _bool_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray(
        [str(row.get(key, "")).lower() in {"1", "true", "yes"} for row in rows],
        dtype=bool,
    )


def _write_npz(
    path: str | Path,
    *,
    export_payload: dict[str, object],
    rows: list[dict[str, object]],
    sequence_path: str,
    run_dir: str,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {
        "segment_index": _int_array(rows, "segment_index"),
        "segment_position_in_fly": _int_array(rows, "segment_position_in_fly"),
        "eval_position": _int_array(rows, "eval_position"),
        "segment_id": _string_array(rows, "segment_id"),
        "fly_id": _string_array(rows, "fly_id"),
        "sample_key": _string_array(rows, "sample_key"),
        "genotype": _string_array(rows, "genotype"),
        "cohort": _string_array(rows, "cohort"),
        "split": _string_array(rows, "split"),
        "predicted_genotype": _string_array(rows, "predicted_genotype"),
        "predicted_cohort": _string_array(rows, "predicted_cohort"),
        "both_correct": _bool_array(rows, "both_correct"),
        "qc_flags": _string_array(rows, "qc_flags"),
        "terminated_by_training_end": _bool_array(rows, "terminated_by_training_end"),
        "embedding_kind": np.asarray(str(export_payload["embedding_kind"])),
        "model_sequence_unit": np.asarray(str(export_payload["sequence_unit"])),
        "source_sequence_path": np.asarray(sequence_path),
        "source_run_dir": np.asarray(run_dir),
    }
    if "segment_embeddings" in export_payload:
        arrays["segment_embeddings"] = np.asarray(export_payload["segment_embeddings"], dtype=np.float32)
    if "unit_embeddings" in export_payload:
        arrays["unit_embeddings"] = np.asarray(export_payload["unit_embeddings"], dtype=np.float32)
    if str(export_payload["embedding_kind"]) == "segment":
        arrays["embeddings"] = arrays["segment_embeddings"]
    elif str(export_payload["embedding_kind"]) == "unit":
        arrays["embeddings"] = arrays["unit_embeddings"]
    np.savez_compressed(out_path, **arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export learned per-segment embeddings from a saved holdout torch sequence model.",
    )
    parser.add_argument("--run-dir", required=True, help="Saved holdout sequence run directory.")
    parser.add_argument("--sequence-path", required=True, help="Input .npz sequence tensor artifact.")
    parser.add_argument("--output-npz", required=True, help="Path for numeric embedding arrays.")
    parser.add_argument("--output-csv", required=True, help="Path for embedding row metadata CSV.")
    parser.add_argument(
        "--split",
        default="valid",
        choices=("train", "valid", "all"),
        help="Prediction split to export. Use 'all' to export every fly in the sequence artifact.",
    )
    parser.add_argument(
        "--embedding-kind",
        default="segment",
        choices=("segment", "unit", "both"),
        help="Embedding view to export. 'segment' is the raw Conv1D/projection embedding.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        help="Optional source segment table to join by segment_id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override. Defaults to the device saved in the model artifact.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    model_path = run_dir / "model_artifact.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing model_artifact.json: {model_path}. "
            "Initial embedding export expects a saved holdout run, not a CV-only run."
        )
    model = load_torch_sequence_model_artifact(_load_json(model_path), device=args.device)
    x, examples, _ = load_sequence_npz(args.sequence_path)
    fly_ids = _prediction_fly_ids(run_dir, args.split)
    selected_examples = _filter_examples(examples, fly_ids)
    export_payload = export_torch_sequence_embeddings(
        x,
        selected_examples,
        model=model,
        embedding_kind=args.embedding_kind,
    )
    prediction_rows = _load_prediction_rows(run_dir)
    if args.split != "all":
        prediction_rows = [row for row in prediction_rows if row.get("split") == args.split]
    enriched_rows = _enrich_rows(
        list(export_payload["rows"]),
        sequence_metadata=_sequence_metadata_by_index(args.sequence_path),
        predictions_by_fly=_prediction_rows_by_fly(prediction_rows),
        segment_rows_by_id=_segments_by_id(args.segments),
        split=args.split,
    )
    _write_npz(
        args.output_npz,
        export_payload=export_payload,
        rows=enriched_rows,
        sequence_path=args.sequence_path,
        run_dir=args.run_dir,
    )
    _write_csv(args.output_csv, enriched_rows)
    print(
        f"wrote {len(enriched_rows)} {args.embedding_kind} embedding rows "
        f"for {len(selected_examples)} flies to {args.output_npz} and {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
