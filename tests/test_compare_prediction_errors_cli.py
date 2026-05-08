from __future__ import annotations

import csv

from flygen_ml.cli import compare_prediction_errors


def _read_csv(path):
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def test_compare_prediction_errors_compares_single_target_and_sequence_runs(monkeypatch, tmp_path, capsys):
    logreg_run = tmp_path / "logreg"
    sequence_run = tmp_path / "sequence"
    logreg_run.mkdir()
    sequence_run.mkdir()
    (logreg_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,n_segments,evidence_bin",
                "0,valid,fly0,s0,genotype,G0,G0,0.90,50,high",
                "1,valid,fly1,s1,genotype,G1,G0,0.60,20,moderate",
                "2,valid,fly2,s2,genotype,G1,G1,0.80,10,low",
                "0,train,fly3,s3,genotype,G0,G1,0.55,10,low",
            ]
        )
    )
    (sequence_run / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,actual_genotype,predicted_genotype,genotype_probability,actual_cohort,predicted_cohort,cohort_probability,both_correct,n_segments,n_segments_with_qc_flags,evidence_bin",
                "4,valid,fly0,s0,G0,G1,0.70,intact,intact,0.80,False,50,0,high",
                "3,valid,fly1,s1,G1,G1,0.85,removed,removed,0.90,True,20,0,moderate",
                "2,valid,fly2,s2,G1,G1,0.80,removed,intact,0.55,False,10,1,low",
            ]
        )
    )
    output_path = tmp_path / "joined.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_prediction_errors",
            "--run-a",
            str(logreg_run),
            "--run-b",
            str(sequence_run),
            "--axis",
            "genotype",
            "--run-a-name",
            "logreg",
            "--run-b-name",
            "conv1d",
            "--join-without-fold",
            "--output",
            str(output_path),
        ],
    )

    assert compare_prediction_errors.main() == 0

    out = capsys.readouterr().out
    assert "axis: genotype" in out
    assert "join_columns: fly_id, sample_key, split" in out
    assert "n_joined_examples: 3" in out
    assert "logreg_accuracy: 0.667" in out
    assert "conv1d_accuracy: 0.667" in out
    assert "correctness_counts: both_correct=1, logreg_only_correct=1, conv1d_only_correct=1, both_wrong=0" in out

    rows = _read_csv(output_path)
    assert len(rows) == 3
    assert {row["correctness_case"] for row in rows} == {
        "both_correct",
        "run_a_only_correct",
        "run_b_only_correct",
    }


def test_compare_prediction_errors_can_join_aligned_folds(monkeypatch, tmp_path, capsys):
    run_a = tmp_path / "a.csv"
    run_b = tmp_path / "b.csv"
    run_a.write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability",
                "0,valid,fly0,s0,cohort,intact,intact,0.9",
                "1,valid,fly0,s0,cohort,intact,removed,0.6",
            ]
        )
    )
    run_b.write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,actual_genotype,predicted_genotype,genotype_probability,actual_cohort,predicted_cohort,cohort_probability",
                "0,valid,fly0,s0,G0,G0,0.8,intact,removed,0.7",
                "1,valid,fly0,s0,G0,G0,0.8,intact,removed,0.7",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_prediction_errors",
            "--predictions-a",
            str(run_a),
            "--predictions-b",
            str(run_b),
            "--axis",
            "cohort",
        ],
    )

    assert compare_prediction_errors.main() == 0

    out = capsys.readouterr().out
    assert "join_columns: fly_id, sample_key, split, fold" in out
    assert "n_joined_examples: 2" in out
    assert "run_a_accuracy: 0.500" in out
    assert "run_b_accuracy: 0.000" in out


def test_compare_prediction_errors_rejects_duplicate_keys_when_ignoring_fold(monkeypatch, tmp_path):
    run_a = tmp_path / "a.csv"
    run_b = tmp_path / "b.csv"
    run_a.write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability",
                "0,valid,fly0,s0,genotype,G0,G0,0.9",
                "1,valid,fly0,s0,genotype,G0,G0,0.8",
            ]
        )
    )
    run_b.write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,actual_genotype,predicted_genotype,genotype_probability",
                "0,valid,fly0,s0,G0,G0,0.7",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_prediction_errors",
            "--predictions-a",
            str(run_a),
            "--predictions-b",
            str(run_b),
            "--axis",
            "genotype",
            "--join-without-fold",
        ],
    )

    try:
        compare_prediction_errors.main()
    except ValueError as error:
        assert "duplicate join keys" in str(error)
    else:
        raise AssertionError("expected duplicate join keys to fail")
