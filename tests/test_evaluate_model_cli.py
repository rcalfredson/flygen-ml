from __future__ import annotations

import json

from flygen_ml.cli import evaluate_model


def test_evaluate_model_prints_holdout_summary(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "holdout"
    run_dir.mkdir()
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "label_key": "cohort",
                "train": {
                    "accuracy": 0.9,
                    "balanced_accuracy": 0.8,
                    "n_examples": 10,
                    "label_recall": {"intact": 1.0, "removed": 0.6},
                },
                "valid": {
                    "accuracy": 0.75,
                    "balanced_accuracy": 0.7,
                    "n_examples": 4,
                    "label_recall": {"intact": 0.5, "removed": 0.9},
                },
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_model", "--run-dir", str(run_dir)],
    )

    assert evaluate_model.main() == 0

    out = capsys.readouterr().out
    assert "run_type: holdout" in out
    assert "label_key: cohort" in out
    assert "valid: accuracy=0.750" in out
    assert "removed=0.900" in out


def test_evaluate_model_prints_cv_summary(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "cv"
    run_dir.mkdir()
    (run_dir / "cv_metrics_summary.json").write_text(
        json.dumps(
            {
                "label_key": "cohort",
                "n_folds": 2,
                "folds": [
                    {
                        "fold": 0,
                        "valid": {
                            "accuracy": 0.8,
                            "balanced_accuracy": 0.75,
                            "n_examples": 5,
                            "label_recall": {"intact": 0.8, "removed": 0.7},
                        },
                    },
                    {
                        "fold": 1,
                        "valid": {
                            "accuracy": 0.9,
                            "balanced_accuracy": 0.85,
                            "n_examples": 5,
                            "label_recall": {"intact": 0.9, "removed": 0.8},
                        },
                    },
                ],
                "summary": {
                    "valid": {
                        "accuracy": {"n": 2, "mean": 0.85, "std": 0.05, "min": 0.8, "max": 0.9},
                        "balanced_accuracy": {"n": 2, "mean": 0.8, "std": 0.05, "min": 0.75, "max": 0.85},
                        "label_recall": {
                            "intact": {"n": 2, "mean": 0.85, "std": 0.05, "min": 0.8, "max": 0.9},
                            "removed": {"n": 2, "mean": 0.75, "std": 0.05, "min": 0.7, "max": 0.8},
                        },
                    }
                },
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_model", "--run-dir", str(run_dir)],
    )

    assert evaluate_model.main() == 0

    out = capsys.readouterr().out
    assert "run_type: grouped_cv" in out
    assert "fold 0: valid_accuracy=0.800" in out
    assert "valid_accuracy: mean=0.850" in out
    assert "valid_recall[removed]: mean=0.750" in out


def test_evaluate_model_prints_cv_confusion_and_misclassifications(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "cv"
    run_dir.mkdir()
    (run_dir / "cv_metrics_summary.json").write_text(
        json.dumps(
            {
                "label_key": "cohort",
                "n_folds": 1,
                "folds": [
                    {
                        "fold": 0,
                        "valid": {
                            "accuracy": 0.5,
                            "balanced_accuracy": 0.5,
                            "n_examples": 2,
                            "label_recall": {"intact": 1.0, "removed": 0.0},
                        },
                    }
                ],
                "summary": {
                    "valid": {
                        "accuracy": {"n": 1, "mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                        "balanced_accuracy": {"n": 1, "mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                        "label_recall": {
                            "intact": {"n": 1, "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                            "removed": {"n": 1, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                        },
                    }
                },
            }
        )
    )
    (run_dir / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,n_segments,evidence_bin",
                "0,valid,fly0,s0,cohort,intact,intact,0.1,50,high_n_segments_ge50",
                "0,valid,fly1,s1,cohort,removed,intact,0.4,10,low_n_segments_lt20",
                "0,train,fly2,s2,cohort,removed,removed,0.9,10,low_n_segments_lt20",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_model",
            "--run-dir",
            str(run_dir),
            "--confusion",
            "--misclassifications",
        ],
    )

    assert evaluate_model.main() == 0

    out = capsys.readouterr().out
    assert "prediction_split: valid" in out
    assert "confusion_matrix:" in out
    assert "removed\t1\t0" in out
    assert "misclassifications: 1" in out
    assert "fold=0 actual=removed predicted=intact" in out
