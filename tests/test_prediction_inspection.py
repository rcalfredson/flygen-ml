from __future__ import annotations

import csv
import json

from flygen_ml.modeling.inspection import build_prediction_inspection_rows, write_prediction_inspection_rows


def test_build_prediction_inspection_rows_reports_misclassified_validation_rows(tmp_path):
    predictions = [
        {
            "split": "valid",
            "fly_id": "fly0",
            "sample_key": "sample0",
            "actual_genotype": "A",
            "predicted_genotype": "B",
            "predicted_probability": 0.55,
        },
        {
            "split": "valid",
            "fly_id": "fly1",
            "sample_key": "sample1",
            "actual_genotype": "B",
            "predicted_genotype": "A",
            "predicted_probability": 0.20,
        },
        {
            "split": "train",
            "fly_id": "fly2",
            "sample_key": "sample2",
            "actual_genotype": "A",
            "predicted_genotype": "B",
            "predicted_probability": 0.90,
        },
    ]
    feature_rows = [
        {"fly_id": "fly0", "sample_key": "sample0", "genotype": "A", "wide_mean": 3.0, "count": 1.0},
        {"fly_id": "fly1", "sample_key": "sample1", "genotype": "B", "wide_mean": 0.0, "count": 4.0},
        {"fly_id": "fly2", "sample_key": "sample2", "genotype": "A", "wide_mean": 3.0, "count": 1.0},
    ]
    model = {
        "labels": ["A", "B"],
        "feature_names": ["wide_mean", "count"],
        "feature_means": [1.0, 2.0],
        "feature_stds": [1.0, 1.0],
        "weights": [0.5, 0.25],
    }

    rows = build_prediction_inspection_rows(
        predictions=predictions,
        feature_rows=feature_rows,
        model=model,
        top_n=1,
    )

    assert len(rows) == 2
    assert rows[0]["fly_id"] == "fly0"
    assert rows[0]["decision_margin"] == 0.050000000000000044
    assert rows[0]["top_toward_predicted"] == "wide_mean:1"
    assert rows[0]["top_against_predicted"] == "count:-0.25"
    assert rows[1]["fly_id"] == "fly1"
    assert rows[1]["top_toward_predicted"] == "wide_mean:-0.5"
    assert rows[1]["top_against_predicted"] == "count:0.5"

    output_path = tmp_path / "inspection.csv"
    with output_path.open("w", newline="") as handle:
        write_prediction_inspection_rows(rows, handle)

    written = list(csv.DictReader(output_path.open()))
    assert written[0]["fly_id"] == "fly0"
    assert "wide_mean" in written[0]


def test_inspect_misclassifications_cli_uses_metadata_features_path(tmp_path, monkeypatch, capsys):
    from flygen_ml.cli import inspect_misclassifications

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    features_path = tmp_path / "features.csv"
    output_path = tmp_path / "inspection.csv"

    (run_dir / "run_metadata.json").write_text(json.dumps({"features_path": str(features_path)}))
    (run_dir / "model_artifact.json").write_text(
        json.dumps(
            {
                "labels": ["A", "B"],
                "feature_names": ["wide_mean"],
                "feature_means": [1.0],
                "feature_stds": [1.0],
                "weights": [0.5],
            }
        )
    )
    (run_dir / "predictions.csv").write_text(
        "\n".join(
            [
                "split,fly_id,sample_key,actual_genotype,predicted_genotype,predicted_probability",
                "valid,fly0,sample0,A,B,0.55",
            ]
        )
    )
    features_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,chamber_type,training_idx,wide_mean",
                "fly0,sample0,A,large,1,3.0",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect_misclassifications",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
        ],
    )

    exit_code = inspect_misclassifications.main()

    assert exit_code == 0
    assert output_path.exists()
    assert "wrote 1 inspected rows" in capsys.readouterr().out
