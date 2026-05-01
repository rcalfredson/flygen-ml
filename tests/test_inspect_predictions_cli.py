from __future__ import annotations

import csv
import json

from flygen_ml.cli import inspect_predictions


def test_build_prediction_review_rows_joins_predictions_to_metadata():
    prediction_rows = [
        {
            "fold": "0",
            "split": "valid",
            "fly_id": "fly0",
            "sample_key": "sample0",
            "label_key": "cohort",
            "actual_label": "intact",
            "predicted_label": "removed",
            "predicted_probability": "0.80",
            "n_segments": "25",
            "n_segments_with_qc_flags": "2",
            "evidence_bin": "moderate_n_segments_20_to_49",
        },
        {
            "fold": "0",
            "split": "valid",
            "fly_id": "fly1",
            "sample_key": "sample1",
            "label_key": "cohort",
            "actual_label": "removed",
            "predicted_label": "removed",
            "predicted_probability": "0.90",
            "n_segments": "60",
            "n_segments_with_qc_flags": "0",
            "evidence_bin": "high_n_segments_ge50",
        },
    ]
    feature_rows = [
        {
            "fly_id": "fly0",
            "sample_key": "sample0",
            "genotype": "G0",
            "cohort": "intact",
            "chamber_type": "large",
            "training_idx": 1,
            "n_segments": 25,
            "duration_frames_mean": 12.0,
        },
        {
            "fly_id": "fly1",
            "sample_key": "sample1",
            "genotype": "G1",
            "cohort": "removed",
            "chamber_type": "large",
            "training_idx": 1,
            "n_segments": 60,
            "duration_frames_mean": 20.0,
        },
    ]

    rows = inspect_predictions.build_prediction_review_rows(
        prediction_rows=prediction_rows,
        feature_rows=feature_rows,
        errors_only=True,
        include_features=True,
    )

    assert len(rows) == 1
    assert rows[0]["fly_id"] == "fly0"
    assert rows[0]["correct"] is False
    assert rows[0]["decision_margin"] == 0.30000000000000004
    assert rows[0]["genotype"] == "G0"
    assert rows[0]["cohort"] == "intact"
    assert rows[0]["duration_frames_mean"] == 12.0


def test_inspect_predictions_cli_writes_cv_error_review(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    features_path = tmp_path / "features.csv"
    output_path = tmp_path / "review.csv"

    (run_dir / "run_metadata.json").write_text(json.dumps({"features_path": str(features_path)}))
    (run_dir / "cv_predictions.csv").write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,predicted_probability,n_segments,n_segments_with_qc_flags,evidence_bin",
                "0,valid,fly0,sample0,cohort,intact,removed,0.80,25,2,moderate_n_segments_20_to_49",
                "0,valid,fly1,sample1,cohort,removed,removed,0.90,60,0,high_n_segments_ge50",
                "0,train,fly2,sample2,cohort,intact,removed,0.70,10,0,low_n_segments_lt20",
            ]
        )
    )
    features_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,cohort,chamber_type,training_idx,n_segments,n_segments_with_qc_flags,duration_frames_mean",
                "fly0,sample0,G0,intact,large,1,25,2,12.0",
                "fly1,sample1,G1,removed,large,1,60,0,20.0",
                "fly2,sample2,G0,intact,large,1,10,0,8.0",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect_predictions",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
            "--errors-only",
            "--include-features",
        ],
    )

    assert inspect_predictions.main() == 0

    assert "wrote 1 prediction review rows" in capsys.readouterr().out
    rows = list(csv.DictReader(output_path.open()))
    assert len(rows) == 1
    assert rows[0]["fold"] == "0"
    assert rows[0]["actual_label"] == "intact"
    assert rows[0]["predicted_label"] == "removed"
    assert rows[0]["correct"] == "False"
    assert rows[0]["genotype"] == "G0"
    assert rows[0]["duration_frames_mean"] == "12.0"
