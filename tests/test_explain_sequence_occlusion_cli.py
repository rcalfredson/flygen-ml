from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from flygen_ml.cli import explain_sequence_occlusion
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
        qc_flags=np.asarray(["", "", ""]),
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


def _fake_examples():
    return [
        FlySequenceExample(
            fly_id="a0",
            sample_key="s_a0",
            genotype="A",
            cohort="intact",
            segment_indices=np.asarray([0, 1]),
            n_segments=2,
            n_segments_with_qc_flags=0,
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


def _write_training_sequence_npz(path):
    x_rows = []
    fly_ids = []
    sample_keys = []
    segment_ids = []
    genotypes = []
    cohorts = []
    qc_flags = []
    specs = [
        ("a0", "A", "intact", 0.0),
        ("a1", "A", "intact", 0.1),
        ("a2", "A", "intact", -0.1),
        ("b0", "B", "removed", 1.0),
        ("b1", "B", "removed", 1.1),
        ("b2", "B", "removed", 0.9),
    ]
    for fly_id, genotype, cohort, value in specs:
        for segment_idx in range(2):
            base = np.linspace(value, value + 0.1, num=8, dtype=np.float32)
            x_rows.append(np.stack([base, base + segment_idx * 0.01], axis=1))
            fly_ids.append(fly_id)
            sample_keys.append(f"s_{fly_id}")
            segment_ids.append(f"{fly_id}_seg{segment_idx}")
            genotypes.append(genotype)
            cohorts.append(cohort)
            qc_flags.append("")
    np.savez_compressed(
        path,
        x=np.stack(x_rows),
        mask=np.ones((len(x_rows), 8), dtype=bool),
        segment_id=np.asarray(segment_ids),
        sample_key=np.asarray(sample_keys),
        fly_id=np.asarray(fly_ids),
        genotype=np.asarray(genotypes),
        cohort=np.asarray(cohorts),
        qc_flags=np.asarray(qc_flags),
        channels=np.asarray(["x_rel", "y_rel"]),
        target_length=np.asarray(8),
    )


def _write_training_config(path):
    path.write_text(
        "\n".join(
            [
                "model_name: segment_conv1d_meanpool_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "valid_fraction: 0.34",
                "conv_channels: 2",
                "embedding_dim: 4",
                "dropout: 0.0",
                "train_max_segments_per_fly: 1",
                "eval_max_segments_per_fly: 0",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "device: cpu",
            ]
        )
    )


def test_explain_sequence_occlusion_cli_filters_split_and_enriches_segment_ids(
    tmp_path,
    monkeypatch,
):
    sequence_path = tmp_path / "sequences.npz"
    run_dir = tmp_path / "run"
    output_path = tmp_path / "segment_occlusion.csv"
    _write_sequence_npz(sequence_path)
    _write_run_files(run_dir)
    calls = []

    def fake_load_model(model_artifact, *, device=None):
        calls.append(("load_model", model_artifact, device))
        return {"side_feature_names": []}

    def fake_load_sequence_npz(path):
        calls.append(("load_sequence", path))
        return np.zeros((3, 8, 2), dtype=np.float32), _fake_examples(), {}

    def fake_explain(x, examples, *, model, side_inputs=None):
        calls.append(("explain", [example.fly_id for example in examples], side_inputs))
        return [
            {
                "fly_id": "a0",
                "sample_key": "s_a0",
                "segment_index": 0,
                "actual_genotype": "A",
                "predicted_genotype": "A",
                "occluded_predicted_genotype": "A",
                "genotype_prediction_changed": False,
                "actual_cohort": "intact",
                "predicted_cohort": "intact",
                "occluded_predicted_cohort": "removed",
                "cohort_prediction_changed": True,
                "joint_prediction_changed": True,
                "occlusion_status": "ok",
                "predicted_cohort_probability_delta": 0.25,
            },
            {
                "fly_id": "a0",
                "sample_key": "s_a0",
                "segment_index": 1,
                "actual_genotype": "A",
                "predicted_genotype": "A",
                "occluded_predicted_genotype": "B",
                "genotype_prediction_changed": True,
                "actual_cohort": "intact",
                "predicted_cohort": "intact",
                "occluded_predicted_cohort": "intact",
                "cohort_prediction_changed": False,
                "joint_prediction_changed": True,
                "occlusion_status": "ok",
                "predicted_cohort_probability_delta": -0.1,
            },
        ]

    monkeypatch.setattr(explain_sequence_occlusion, "load_torch_sequence_model_artifact", fake_load_model)
    monkeypatch.setattr(explain_sequence_occlusion, "load_sequence_npz", fake_load_sequence_npz)
    monkeypatch.setattr(explain_sequence_occlusion, "explain_torch_sequence_segment_occlusion", fake_explain)
    monkeypatch.setattr(
        "sys.argv",
        [
            "explain_sequence_occlusion",
            "--run-dir",
            str(run_dir),
            "--sequence-path",
            str(sequence_path),
            "--output-csv",
            str(output_path),
            "--split",
            "valid",
            "--device",
            "cpu",
        ],
    )

    assert explain_sequence_occlusion.main() == 0

    rows = _read_csv(output_path)
    assert [row["fly_id"] for row in rows] == ["a0", "a0"]
    assert [row["segment_id"] for row in rows] == ["a0_seg0", "a0_seg1"]
    assert {row["split"] for row in rows} == {"valid"}
    assert [row["occluded_predicted_genotype"] for row in rows] == ["A", "B"]
    assert [row["genotype_prediction_changed"] for row in rows] == ["False", "True"]
    assert [row["occluded_predicted_cohort"] for row in rows] == ["removed", "intact"]
    assert [row["cohort_prediction_changed"] for row in rows] == ["True", "False"]
    assert [row["joint_prediction_changed"] for row in rows] == ["True", "True"]
    assert ("explain", ["a0"], None) in calls
    assert calls[0][0] == "load_model"
    assert calls[0][2] == "cpu"


def test_explain_sequence_occlusion_cli_requires_holdout_model_artifact(tmp_path, monkeypatch):
    sequence_path = tmp_path / "sequences.npz"
    run_dir = tmp_path / "run"
    output_path = tmp_path / "segment_occlusion.csv"
    _write_sequence_npz(sequence_path)
    run_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "explain_sequence_occlusion",
            "--run-dir",
            str(run_dir),
            "--sequence-path",
            str(sequence_path),
            "--output-csv",
            str(output_path),
        ],
    )

    with pytest.raises(FileNotFoundError, match="model_artifact.json"):
        explain_sequence_occlusion.main()


def test_explain_sequence_occlusion_cli_runs_from_saved_torch_artifact(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from flygen_ml.modeling.sequence_training import train_and_save_sequence_run

    sequence_path = tmp_path / "sequences.npz"
    config_path = tmp_path / "segment_conv1d.yaml"
    run_dir = tmp_path / "run"
    output_path = tmp_path / "segment_occlusion.csv"
    _write_training_sequence_npz(sequence_path)
    _write_training_config(config_path)
    train_and_save_sequence_run(
        config_path=config_path,
        sequence_path=sequence_path,
        output_dir=run_dir,
    )
    valid_predictions = [
        row for row in _read_csv(run_dir / "predictions.csv")
        if row["split"] == "valid"
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            "explain_sequence_occlusion",
            "--run-dir",
            str(run_dir),
            "--sequence-path",
            str(sequence_path),
            "--output-csv",
            str(output_path),
            "--split",
            "valid",
            "--device",
            "cpu",
        ],
    )

    assert explain_sequence_occlusion.main() == 0

    rows = _read_csv(output_path)
    assert len(rows) == len(valid_predictions) * 2
    assert {row["occlusion_status"] for row in rows} == {"ok"}
    assert all(row["segment_id"].endswith(("seg0", "seg1")) for row in rows)
    assert all(row["occluded_predicted_genotype"] for row in rows)
    assert all(row["occluded_predicted_cohort"] for row in rows)
    assert {row["joint_prediction_changed"] for row in rows} <= {"True", "False"}
    assert all(row["predicted_cohort_probability_delta"] not in {"", "nan"} for row in rows)
