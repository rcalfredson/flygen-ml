from __future__ import annotations

import csv
import json

import numpy as np

from flygen_ml.cli import export_sequence_embeddings
from flygen_ml.modeling.sequence_models import FlySequenceExample


def _write_sequence_npz(path):
    np.savez_compressed(
        path,
        x=np.zeros((3, 8, 2), dtype=np.float32),
        mask=np.ones((3, 8), dtype=bool),
        segment_id=np.asarray(["a0_seg0", "a0_seg1", "b0_seg0"]),
        sample_key=np.asarray(["s_a0", "s_a0", "s_b0"]),
        fly_id=np.asarray(["a0", "a0", "b0"]),
        genotype=np.asarray(["A", "A", "B"]),
        cohort=np.asarray(["intact", "intact", "removed"]),
        qc_flags=np.asarray(["", "has_missing_frames", ""]),
        terminated_by_training_end=np.asarray([False, True, False]),
        channels=np.asarray(["x_rel", "y_rel"]),
        target_length=np.asarray(8),
    )


def _write_run_files(run_dir):
    run_dir.mkdir()
    (run_dir / "model_artifact.json").write_text(
        json.dumps(
            {
                "model_kind": "sequence_conv1d_meanpool_torch_v1",
                "side_feature_names": [],
            }
        )
    )
    (run_dir / "predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,actual_genotype,predicted_genotype,genotype_probability,actual_cohort,predicted_cohort,cohort_probability,both_correct,n_segments,n_segments_with_qc_flags,evidence_bin",
                ",valid,a0,s_a0,A,A,0.9,intact,intact,0.8,True,2,0,medium",
                ",train,b0,s_b0,B,B,0.7,removed,removed,0.6,True,1,0,low",
            ]
        )
    )


def _write_segments_csv(path):
    path.write_text(
        "\n".join(
            [
                "segment_id,start_frame,stop_frame,anchor_reward_kind",
                "a0_seg0,10,20,calculated",
                "a0_seg1,21,40,calculated",
                "b0_seg0,50,60,calculated",
            ]
        )
    )


def _fake_examples():
    return [
        FlySequenceExample(
            fly_id="a0",
            sample_key="s_a0",
            genotype="A",
            cohort="intact",
            segment_indices=np.asarray([0, 1]),
            n_segments=2,
            n_segments_with_qc_flags=1,
        ),
        FlySequenceExample(
            fly_id="b0",
            sample_key="s_b0",
            genotype="B",
            cohort="removed",
            segment_indices=np.asarray([2]),
            n_segments=1,
            n_segments_with_qc_flags=0,
        ),
    ]


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_sequence_embeddings_cli_filters_split_and_writes_artifacts(tmp_path, monkeypatch):
    sequence_path = tmp_path / "sequences.npz"
    run_dir = tmp_path / "run"
    segments_path = tmp_path / "segments.csv"
    output_npz = tmp_path / "embeddings.npz"
    output_csv = tmp_path / "embeddings.csv"
    _write_sequence_npz(sequence_path)
    _write_run_files(run_dir)
    _write_segments_csv(segments_path)
    calls = []

    def fake_load_model(model_artifact, *, device=None):
        calls.append(("load_model", model_artifact, device))
        return {"model": "loaded"}

    def fake_load_sequence_npz(path):
        calls.append(("load_sequence", path))
        return np.zeros((3, 8, 2), dtype=np.float32), _fake_examples(), {}

    def fake_export(x, examples, *, model, embedding_kind):
        calls.append(("export", [example.fly_id for example in examples], model, embedding_kind))
        return {
            "rows": [
                {
                    "segment_index": 0,
                    "segment_position_in_fly": 0,
                    "eval_position": 0,
                    "fly_id": "a0",
                    "sample_key": "s_a0",
                    "genotype": "A",
                    "cohort": "intact",
                    "n_segments": 2,
                    "n_segments_with_qc_flags": 1,
                    "selected_for_model_eval": True,
                },
                {
                    "segment_index": 1,
                    "segment_position_in_fly": 1,
                    "eval_position": 1,
                    "fly_id": "a0",
                    "sample_key": "s_a0",
                    "genotype": "A",
                    "cohort": "intact",
                    "n_segments": 2,
                    "n_segments_with_qc_flags": 1,
                    "selected_for_model_eval": True,
                },
            ],
            "embedding_kind": "segment",
            "sequence_unit": "segment",
            "segment_embeddings": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }

    monkeypatch.setattr(export_sequence_embeddings, "load_torch_sequence_model_artifact", fake_load_model)
    monkeypatch.setattr(export_sequence_embeddings, "load_sequence_npz", fake_load_sequence_npz)
    monkeypatch.setattr(export_sequence_embeddings, "export_torch_sequence_embeddings", fake_export)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_sequence_embeddings",
            "--run-dir",
            str(run_dir),
            "--sequence-path",
            str(sequence_path),
            "--output-npz",
            str(output_npz),
            "--output-csv",
            str(output_csv),
            "--split",
            "valid",
            "--embedding-kind",
            "segment",
            "--segments",
            str(segments_path),
            "--device",
            "cpu",
        ],
    )

    assert export_sequence_embeddings.main() == 0

    rows = _read_csv(output_csv)
    assert [row["segment_id"] for row in rows] == ["a0_seg0", "a0_seg1"]
    assert {row["split"] for row in rows} == {"valid"}
    assert [row["predicted_genotype"] for row in rows] == ["A", "A"]
    assert [row["qc_flags"] for row in rows] == ["", "has_missing_frames"]
    assert [row["terminated_by_training_end"] for row in rows] == ["False", "True"]
    assert [row["source_start_frame"] for row in rows] == ["10", "21"]
    assert ("export", ["a0"], {"model": "loaded"}, "segment") in calls
    assert calls[0][0] == "load_model"
    assert calls[0][2] == "cpu"

    payload = np.load(output_npz)
    assert payload["embeddings"].shape == (2, 2)
    assert payload["segment_embeddings"].shape == (2, 2)
    assert payload["segment_id"].tolist() == ["a0_seg0", "a0_seg1"]
    assert payload["split"].tolist() == ["valid", "valid"]
    assert payload["both_correct"].tolist() == [True, True]
    assert str(payload["embedding_kind"]) == "segment"
