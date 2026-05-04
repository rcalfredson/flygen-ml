from __future__ import annotations

import json

from flygen_ml.cli import evaluate_sequence_model


def test_evaluate_sequence_model_prints_holdout_summary(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "holdout"
    run_dir.mkdir()
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "model_kind": "sequence_meanpool_mlp_numpy_v1",
                "train": {
                    "n_examples": 10,
                    "joint_accuracy": 0.8,
                    "genotype": {"accuracy": 0.9, "balanced_accuracy": 0.85},
                    "cohort": {"accuracy": 0.7, "balanced_accuracy": 0.65},
                },
                "valid": {
                    "n_examples": 4,
                    "joint_accuracy": 0.5,
                    "genotype": {"accuracy": 0.75, "balanced_accuracy": 0.7},
                    "cohort": {"accuracy": 0.75, "balanced_accuracy": 0.8},
                },
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["evaluate_sequence_model", "--run-dir", str(run_dir)])

    assert evaluate_sequence_model.main() == 0

    out = capsys.readouterr().out
    assert "run_type: sequence_holdout" in out
    assert "valid: joint_accuracy=0.500" in out
    assert "genotype_balanced_accuracy=0.700" in out
    assert "cohort_accuracy=0.750" in out


def test_evaluate_sequence_model_prints_cv_summary(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "cv"
    run_dir.mkdir()
    (run_dir / "cv_metrics_summary.json").write_text(
        json.dumps(
            {
                "model_kind": "sequence_meanpool_mlp_numpy_v1",
                "n_folds": 2,
                "folds": [
                    {
                        "fold": 0,
                        "valid": {
                            "n_examples": 5,
                            "joint_accuracy": 0.4,
                            "genotype": {"accuracy": 0.6, "balanced_accuracy": 0.55},
                            "cohort": {"accuracy": 0.7, "balanced_accuracy": 0.65},
                        },
                    },
                    {
                        "fold": 1,
                        "valid": {
                            "n_examples": 5,
                            "joint_accuracy": 0.6,
                            "genotype": {"accuracy": 0.8, "balanced_accuracy": 0.75},
                            "cohort": {"accuracy": 0.9, "balanced_accuracy": 0.85},
                        },
                    },
                ],
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["evaluate_sequence_model", "--run-dir", str(run_dir)])

    assert evaluate_sequence_model.main() == 0

    out = capsys.readouterr().out
    assert "run_type: sequence_grouped_cv" in out
    assert "fold 0: valid_joint_accuracy=0.400" in out
    assert "valid_joint_accuracy: mean=0.500" in out
    assert "valid_genotype_balanced_accuracy: mean=0.650" in out
    assert "valid_cohort_accuracy: mean=0.800" in out


def test_evaluate_sequence_model_prints_confusion_and_misclassifications(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "cv"
    run_dir.mkdir()
    (run_dir / "cv_metrics_summary.json").write_text(
        json.dumps(
            {
                "model_kind": "sequence_meanpool_mlp_numpy_v1",
                "n_folds": 1,
                "folds": [
                    {
                        "fold": 0,
                        "valid": {
                            "n_examples": 2,
                            "joint_accuracy": 0.5,
                            "genotype": {"accuracy": 1.0, "balanced_accuracy": 1.0},
                            "cohort": {"accuracy": 0.5, "balanced_accuracy": 0.5},
                        },
                    }
                ],
            }
        )
    )
    (run_dir / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,actual_genotype,predicted_genotype,genotype_probability,actual_cohort,predicted_cohort,cohort_probability,both_correct,n_segments,n_segments_with_qc_flags,evidence_bin",
                "0,valid,fly0,s0,G0,G0,0.90,intact,intact,0.80,True,50,0,high_n_segments_ge50",
                "0,valid,fly1,s1,G1,G1,0.70,removed,intact,0.60,False,10,1,low_n_segments_lt20",
                "0,train,fly2,s2,G1,G0,0.55,removed,removed,0.80,False,10,1,low_n_segments_lt20",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_sequence_model",
            "--run-dir",
            str(run_dir),
            "--confusion",
            "--misclassifications",
        ],
    )

    assert evaluate_sequence_model.main() == 0

    out = capsys.readouterr().out
    assert "prediction_split: valid" in out
    assert "joint_confusion_matrix:" in out
    assert "genotype_confusion_matrix:" in out
    assert "cohort_confusion_matrix:" in out
    assert "misclassifications: 1" in out
    assert "fold=0 actual=G1|removed predicted=G1|intact" in out
