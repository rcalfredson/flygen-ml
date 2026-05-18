from __future__ import annotations

import csv

import numpy as np
import pytest

from flygen_ml.cli import reduce_sequence_embeddings


def _write_embeddings_npz(path):
    embeddings = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        embeddings=embeddings,
        segment_embeddings=embeddings,
        segment_index=np.asarray([0, 1, 2, 3], dtype=np.int64),
        segment_id=np.asarray(["seg0", "seg1", "seg2", "seg3"]),
        fly_id=np.asarray(["a0", "a0", "b0", "b0"]),
        sample_key=np.asarray(["s_a0", "s_a0", "s_b0", "s_b0"]),
        genotype=np.asarray(["A", "A", "B", "B"]),
        cohort=np.asarray(["intact", "intact", "removed", "removed"]),
        split=np.asarray(["train", "train", "valid", "valid"]),
    )


def _write_metadata_csv(path, *, n_rows: int = 4):
    rows = [
        {"segment_index": "0", "segment_id": "seg0", "fly_id": "a0", "genotype": "A"},
        {"segment_index": "1", "segment_id": "seg1", "fly_id": "a0", "genotype": "A"},
        {"segment_index": "2", "segment_id": "seg2", "fly_id": "b0", "genotype": "B"},
        {"segment_index": "3", "segment_id": "seg3", "fly_id": "b0", "genotype": "B"},
    ][:n_rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment_index", "segment_id", "fly_id", "genotype"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_reduce_sequence_embeddings_cli_writes_pca_artifacts(tmp_path, monkeypatch):
    embeddings_path = tmp_path / "embeddings.npz"
    metadata_path = tmp_path / "embeddings.csv"
    output_npz = tmp_path / "embeddings_pca.npz"
    output_csv = tmp_path / "embeddings_pca.csv"
    _write_embeddings_npz(embeddings_path)
    _write_metadata_csv(metadata_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "reduce_sequence_embeddings",
            "--embeddings-npz",
            str(embeddings_path),
            "--metadata-csv",
            str(metadata_path),
            "--output-npz",
            str(output_npz),
            "--output-csv",
            str(output_csv),
            "--embedding-key",
            "segment_embeddings",
            "--n-components",
            "2",
        ],
    )

    assert reduce_sequence_embeddings.main() == 0

    rows = _read_csv(output_csv)
    assert [row["segment_id"] for row in rows] == ["seg0", "seg1", "seg2", "seg3"]
    assert set(rows[0]) == {
        "segment_index",
        "segment_id",
        "fly_id",
        "genotype",
        "pc1",
        "pc2",
        "pca_reconstruction_error",
    }
    assert all(row["pc1"] for row in rows)
    assert all(row["pc2"] for row in rows)

    payload = np.load(output_npz)
    assert payload["coordinates"].shape == (4, 2)
    assert payload["pca_coordinates"].shape == (4, 2)
    assert payload["pca_components"].shape == (2, 3)
    assert payload["pca_mean"].shape == (3,)
    assert payload["segment_id"].tolist() == ["seg0", "seg1", "seg2", "seg3"]
    assert str(payload["embedding_key"]) == "segment_embeddings"
    assert str(payload["method"]) == "pca"
    assert payload["pca_explained_variance_ratio"].shape == (2,)
    assert 0.0 < float(payload["pca_explained_variance_ratio"].sum()) <= 1.0
    assert payload["pca_reconstruction_error"].shape == (4,)


def test_reduce_sequence_embeddings_cli_requires_metadata_row_alignment(tmp_path, monkeypatch):
    embeddings_path = tmp_path / "embeddings.npz"
    metadata_path = tmp_path / "embeddings.csv"
    _write_embeddings_npz(embeddings_path)
    _write_metadata_csv(metadata_path, n_rows=3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "reduce_sequence_embeddings",
            "--embeddings-npz",
            str(embeddings_path),
            "--metadata-csv",
            str(metadata_path),
            "--output-npz",
            str(tmp_path / "embeddings_pca.npz"),
            "--output-csv",
            str(tmp_path / "embeddings_pca.csv"),
            "--n-components",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="metadata row count"):
        reduce_sequence_embeddings.main()
