from __future__ import annotations

import csv

from flygen_ml.cli import evaluate_joint_predictions


def _read_csv(path):
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def test_evaluate_joint_predictions_joins_cv_runs_and_writes_summary(monkeypatch, tmp_path, capsys):
    genotype_run = tmp_path / "genotype_run"
    cohort_run = tmp_path / "cohort_run"
    genotype_run.mkdir()
    cohort_run.mkdir()
    (genotype_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,n_segments,evidence_bin",
                "0,valid,fly0,s0,genotype,G0,G0,0.90,55,high_n_segments_ge50",
                "0,valid,fly1,s1,genotype,G1,G0,0.65,10,low_n_segments_lt20",
                "0,train,fly2,s2,genotype,G0,G0,0.80,55,high_n_segments_ge50",
                "1,valid,fly0,s0,genotype,G0,G1,0.55,55,high_n_segments_ge50",
            ]
        )
    )
    (cohort_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,n_segments,evidence_bin",
                "0,valid,fly0,s0,cohort,intact,intact,0.85,55,high_n_segments_ge50",
                "0,valid,fly1,s1,cohort,removed,intact,0.60,10,low_n_segments_lt20",
                "1,valid,fly0,s0,cohort,intact,removed,0.52,55,high_n_segments_ge50",
            ]
        )
    )
    output_path = tmp_path / "joint_predictions.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_joint_predictions",
            "--axis-a-run",
            str(genotype_run),
            "--axis-b-run",
            str(cohort_run),
            "--axis-a-name",
            "genotype",
            "--axis-b-name",
            "cohort",
            "--split",
            "valid",
            "--output",
            str(output_path),
        ],
    )

    assert evaluate_joint_predictions.main() == 0

    out = capsys.readouterr().out
    assert "axis_a_predictions:" in out
    assert "cv_predictions.csv" in out
    assert "join_columns: fly_id, sample_key, split, fold" in out
    assert "n_joined_examples: 3" in out
    assert "joint_accuracy: 0.333" in out
    assert "genotype_accuracy: 0.333" in out
    assert "cohort_accuracy: 0.333" in out
    assert "correctness_counts: both_correct=1, genotype_only_wrong=0, cohort_only_wrong=0, both_wrong=2" in out
    assert "joint_confusion_matrix:" in out
    assert "G1|removed" in out
    assert "by_evidence_bin:" in out
    assert "high_n_segments_ge50: n=2, joint_accuracy=0.500" in out

    rows = _read_csv(output_path)
    assert len(rows) == 3
    assert rows[0]["fold"] == "0"
    assert rows[0]["split"] == "valid"
    assert rows[0]["genotype_actual_label"] == "G0"
    assert rows[0]["cohort_predicted_label"] == "intact"
    assert rows[0]["actual_joint_label"] == "G0|intact"
    assert rows[0]["predicted_joint_label"] == "G0|intact"
    assert rows[0]["joint_correct"] == "True"
    assert rows[0]["genotype_predicted_probability"] == "0.90"
    assert rows[0]["cohort_evidence_bin"] == "high_n_segments_ge50"


def test_evaluate_joint_predictions_accepts_direct_holdout_prediction_csvs(monkeypatch, tmp_path, capsys):
    axis_a_path = tmp_path / "axis_a_predictions.csv"
    axis_b_path = tmp_path / "axis_b_predictions.csv"
    axis_a_path.write_text(
        "\n".join(
            [
                "split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,evidence_bin",
                "valid,fly0,s0,a,A,A,0.9,high",
                "valid,fly1,s1,a,B,A,0.7,low",
            ]
        )
    )
    axis_b_path.write_text(
        "\n".join(
            [
                "split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,evidence_bin",
                "valid,fly0,s0,b,X,Y,0.6,high",
                "valid,fly1,s1,b,Y,Y,0.8,low",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_joint_predictions",
            "--axis-a-predictions",
            str(axis_a_path),
            "--axis-b-predictions",
            str(axis_b_path),
            "--axis-a-name",
            "axis_a",
            "--axis-b-name",
            "axis_b",
        ],
    )

    assert evaluate_joint_predictions.main() == 0

    out = capsys.readouterr().out
    assert "join_columns: fly_id, sample_key, split" in out
    assert "n_joined_examples: 2" in out
    assert "joint_accuracy: 0.000" in out
    assert "axis_a_accuracy: 0.500" in out
    assert "axis_b_accuracy: 0.500" in out
    assert "correctness_counts: both_correct=0, axis_a_only_wrong=1, axis_b_only_wrong=1, both_wrong=0" in out


def test_evaluate_joint_predictions_can_ignore_cv_fold_when_joining(monkeypatch, tmp_path, capsys):
    axis_a_run = tmp_path / "axis_a"
    axis_b_run = tmp_path / "axis_b"
    axis_a_run.mkdir()
    axis_b_run.mkdir()
    (axis_a_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability",
                "0,valid,fly0,s0,genotype,G0,G0,0.9",
                "1,valid,fly1,s1,genotype,G1,G1,0.8",
            ]
        )
    )
    (axis_b_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability",
                "4,valid,fly0,s0,cohort,intact,intact,0.7",
                "3,valid,fly1,s1,cohort,removed,intact,0.6",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_joint_predictions",
            "--axis-a-run",
            str(axis_a_run),
            "--axis-b-run",
            str(axis_b_run),
            "--axis-a-name",
            "genotype",
            "--axis-b-name",
            "cohort",
            "--join-without-fold",
        ],
    )

    assert evaluate_joint_predictions.main() == 0

    out = capsys.readouterr().out
    assert "join_columns: fly_id, sample_key, split" in out
    assert "n_joined_examples: 2" in out
    assert "joint_accuracy: 0.500" in out


def test_evaluate_joint_predictions_rejects_duplicate_keys_when_ignoring_fold(monkeypatch, tmp_path):
    axis_a_run = tmp_path / "axis_a"
    axis_b_run = tmp_path / "axis_b"
    axis_a_run.mkdir()
    axis_b_run.mkdir()
    (axis_a_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability",
                "0,valid,fly0,s0,genotype,G0,G0,0.9",
                "1,valid,fly0,s0,genotype,G0,G0,0.8",
            ]
        )
    )
    (axis_b_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability",
                "0,valid,fly0,s0,cohort,intact,intact,0.7",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_joint_predictions",
            "--axis-a-run",
            str(axis_a_run),
            "--axis-b-run",
            str(axis_b_run),
            "--axis-a-name",
            "genotype",
            "--axis-b-name",
            "cohort",
            "--join-without-fold",
        ],
    )

    try:
        evaluate_joint_predictions.main()
    except ValueError as error:
        assert "duplicate join keys" in str(error)
        assert "genotype" in str(error)
    else:
        raise AssertionError("expected duplicate join keys to fail")
