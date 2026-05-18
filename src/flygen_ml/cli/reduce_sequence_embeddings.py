from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _embedding_matrix(payload, embedding_key: str) -> np.ndarray:
    if embedding_key not in payload:
        available = ", ".join(payload.files)
        raise KeyError(f"embedding key {embedding_key!r} is not present in artifact; available keys: {available}")
    embeddings = np.asarray(payload[embedding_key], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"expected {embedding_key!r} to be a 2D array, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError("cannot reduce an empty embedding matrix")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"embedding matrix {embedding_key!r} contains non-finite values")
    return embeddings


def _fit_pca(embeddings: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    max_components = min(embeddings.shape)
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds max PCA rank {max_components} "
            f"for embedding matrix shape {embeddings.shape}"
        )
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    selected_singular_values = singular_values[:n_components]
    coordinates = u[:, :n_components] * selected_singular_values.reshape(1, -1)
    if embeddings.shape[0] > 1:
        all_explained_variance = (singular_values**2) / (embeddings.shape[0] - 1)
    else:
        all_explained_variance = np.zeros_like(singular_values)
    total_variance = float(all_explained_variance.sum())
    explained_variance = all_explained_variance[:n_components]
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)
    reconstruction = coordinates @ components + mean
    reconstruction_error = np.sqrt(((embeddings - reconstruction) ** 2).mean(axis=1))
    return {
        "coordinates": coordinates.astype(np.float32),
        "components": components.astype(np.float32),
        "mean": mean.astype(np.float32),
        "singular_values": selected_singular_values.astype(np.float32),
        "explained_variance": explained_variance.astype(np.float32),
        "explained_variance_ratio": explained_variance_ratio.astype(np.float32),
        "reconstruction_error": reconstruction_error.astype(np.float32),
        "total_variance": np.asarray(total_variance, dtype=np.float32),
    }


def _enrich_rows(
    rows: list[dict[str, str]],
    coordinates: np.ndarray,
    reconstruction_error: np.ndarray,
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for row, coordinate_row, error in zip(rows, coordinates, reconstruction_error, strict=True):
        pca_values = {
            f"pc{idx + 1}": f"{float(value):.9g}"
            for idx, value in enumerate(coordinate_row)
        }
        enriched.append(
            {
                **row,
                **pca_values,
                "pca_reconstruction_error": f"{float(error):.9g}",
            }
        )
    return enriched


def _write_csv(
    path: str | Path,
    rows: list[dict[str, str]],
    *,
    input_fieldnames: list[str],
    n_components: int,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pca_fieldnames = [f"pc{idx + 1}" for idx in range(n_components)] + ["pca_reconstruction_error"]
    fieldnames = input_fieldnames + [name for name in pca_fieldnames if name not in input_fieldnames]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metadata_arrays(payload, n_rows: int) -> dict[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for key in [
        "segment_index",
        "segment_id",
        "fly_id",
        "sample_key",
        "genotype",
        "cohort",
        "split",
        "predicted_genotype",
        "predicted_cohort",
        "both_correct",
        "qc_flags",
        "terminated_by_training_end",
    ]:
        if key in payload and np.asarray(payload[key]).shape[:1] == (n_rows,):
            copied[key] = np.asarray(payload[key])
    return copied


def _write_npz(
    path: str | Path,
    *,
    source_payload,
    pca: dict[str, np.ndarray],
    embedding_key: str,
    source_embeddings_path: str,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = pca["coordinates"]
    arrays: dict[str, object] = {
        **_metadata_arrays(source_payload, coordinates.shape[0]),
        "coordinates": coordinates,
        "pca_coordinates": coordinates,
        "pca_components": pca["components"],
        "pca_mean": pca["mean"],
        "pca_singular_values": pca["singular_values"],
        "pca_explained_variance": pca["explained_variance"],
        "pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "pca_reconstruction_error": pca["reconstruction_error"],
        "pca_total_variance": pca["total_variance"],
        "embedding_key": np.asarray(embedding_key),
        "method": np.asarray("pca"),
        "source_embeddings_path": np.asarray(source_embeddings_path),
    }
    np.savez_compressed(out_path, **arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce exported sequence embeddings for visualization and review.",
    )
    parser.add_argument("--embeddings-npz", required=True, help="Input embedding artifact from export_sequence_embeddings.")
    parser.add_argument("--metadata-csv", required=True, help="Input metadata CSV from export_sequence_embeddings.")
    parser.add_argument("--output-npz", required=True, help="Output reduced-coordinate .npz artifact.")
    parser.add_argument("--output-csv", required=True, help="Output metadata CSV with reduction coordinates appended.")
    parser.add_argument(
        "--embedding-key",
        default="embeddings",
        help="Embedding array key to reduce, e.g. embeddings, segment_embeddings, or unit_embeddings.",
    )
    parser.add_argument(
        "--method",
        default="pca",
        choices=("pca",),
        help="Reduction method. Currently only PCA is supported.",
    )
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components to retain.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = np.load(args.embeddings_npz)
    embeddings = _embedding_matrix(payload, args.embedding_key)
    metadata_rows, metadata_fieldnames = _load_csv_rows(args.metadata_csv)
    if len(metadata_rows) != embeddings.shape[0]:
        raise ValueError(
            f"metadata row count ({len(metadata_rows)}) does not match embedding row count ({embeddings.shape[0]})"
        )
    pca = _fit_pca(embeddings, args.n_components)
    enriched_rows = _enrich_rows(metadata_rows, pca["coordinates"], pca["reconstruction_error"])
    _write_npz(
        args.output_npz,
        source_payload=payload,
        pca=pca,
        embedding_key=args.embedding_key,
        source_embeddings_path=args.embeddings_npz,
    )
    _write_csv(
        args.output_csv,
        enriched_rows,
        input_fieldnames=metadata_fieldnames,
        n_components=args.n_components,
    )
    variance = ", ".join(f"{value:.3f}" for value in pca["explained_variance_ratio"][:3])
    print(
        f"wrote PCA coordinates for {embeddings.shape[0]} embedding rows "
        f"to {args.output_npz} and {args.output_csv}; "
        f"first explained variance ratios: {variance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
