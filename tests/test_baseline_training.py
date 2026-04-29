from __future__ import annotations

import json

from flygen_ml.modeling.train import train_and_save_cross_validation_run, train_and_save_run


def test_train_and_save_run_writes_artifacts(tmp_path):
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,cohort,chamber_type,training_idx,n_segments,n_segments_with_qc_flags,duration_frames_mean,path_length_px_mean,straightness_mean",
                "a0,s0,A,intact,large,1,3,0,10.0,1.0,0.9",
                "a1,s1,A,intact,large,1,3,0,11.0,1.2,0.8",
                "a2,s2,A,intact,large,1,3,0,9.5,0.8,0.85",
                "b0,s3,B,removed,large,1,3,0,20.0,5.0,0.2",
                "b1,s4,B,removed,large,1,3,0,19.0,4.8,0.3",
                "b2,s5,B,removed,large,1,3,0,21.0,5.2,0.25",
            ]
        )
    )
    config_path = tmp_path / "logreg.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: logreg_v1",
                "model_kind: baseline",
                "group_key: fly_id",
                "random_seed: 7",
                "valid_fraction: 0.34",
                "learning_rate: 0.1",
                "max_iter: 300",
                "l2_reg: 0.01",
                "exclude_feature_names: n_segments,n_segments_with_qc_flags",
            ]
        )
    )

    output_dir = tmp_path / "run"
    metadata = train_and_save_run(
        config_path=config_path,
        features_path=features_path,
        output_dir=output_dir,
    )

    assert metadata["status"] == "completed"
    assert metadata["train_rows"] == 4
    assert metadata["valid_rows"] == 2
    assert metadata["excluded_feature_names"] == ["n_segments", "n_segments_with_qc_flags"]
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "model_artifact.json").exists()
    assert (output_dir / "predictions.csv").exists()

    metrics = json.loads((output_dir / "metrics_summary.json").read_text())
    assert metrics["train"]["n_examples"] == 4
    assert metrics["valid"]["n_examples"] == 2
    assert "valid_by_evidence_bin" in metrics

    model = json.loads((output_dir / "model_artifact.json").read_text())
    assert model["label_key"] == "genotype"
    assert "n_segments" not in model["feature_names"]
    assert "n_segments_with_qc_flags" not in model["feature_names"]
    assert "cohort" not in model["feature_names"]

    predictions = (output_dir / "predictions.csv").read_text()
    assert "n_segments,n_segments_with_qc_flags,evidence_bin" in predictions


def test_train_and_save_run_supports_cohort_label_key(tmp_path):
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,cohort,chamber_type,training_idx,n_segments,n_segments_with_qc_flags,duration_frames_mean,path_length_px_mean,straightness_mean",
                "a0,s0,A,intact,large,1,3,0,10.0,1.0,0.9",
                "a1,s1,B,intact,large,1,3,0,11.0,1.2,0.8",
                "a2,s2,A,intact,large,1,3,0,9.5,0.8,0.85",
                "b0,s3,A,removed,large,1,3,0,20.0,5.0,0.2",
                "b1,s4,B,removed,large,1,3,0,19.0,4.8,0.3",
                "b2,s5,A,removed,large,1,3,0,21.0,5.2,0.25",
            ]
        )
    )
    config_path = tmp_path / "logreg.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: logreg_v1",
                "model_kind: baseline",
                "label_key: cohort",
                "group_key: fly_id",
                "random_seed: 7",
                "valid_fraction: 0.34",
                "learning_rate: 0.1",
                "max_iter: 300",
                "l2_reg: 0.01",
            ]
        )
    )

    output_dir = tmp_path / "cohort_run"
    metadata = train_and_save_run(
        config_path=config_path,
        features_path=features_path,
        output_dir=output_dir,
    )

    assert metadata["label_key"] == "cohort"
    metrics = json.loads((output_dir / "metrics_summary.json").read_text())
    assert metrics["label_key"] == "cohort"

    model = json.loads((output_dir / "model_artifact.json").read_text())
    assert model["label_key"] == "cohort"
    assert model["labels"] == ["intact", "removed"]
    assert "cohort" not in model["feature_names"]
    assert "genotype" not in model["feature_names"]

    predictions = (output_dir / "predictions.csv").read_text()
    assert predictions.startswith("split,fly_id,sample_key,label_key,actual_label,predicted_label,")
    assert "actual_genotype" not in predictions.splitlines()[0]


def test_train_and_save_cross_validation_run_writes_fold_artifacts(tmp_path):
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,cohort,chamber_type,training_idx,n_segments,n_segments_with_qc_flags,duration_frames_mean,path_length_px_mean,straightness_mean",
                "a0,s0,A,intact,large,1,60,0,10.0,1.0,0.9",
                "a1,s1,A,intact,large,1,30,0,11.0,1.2,0.8",
                "a2,s2,A,intact,large,1,5,0,9.5,0.8,0.85",
                "b0,s3,B,removed,large,1,60,0,20.0,5.0,0.2",
                "b1,s4,B,removed,large,1,30,0,19.0,4.8,0.3",
                "b2,s5,B,removed,large,1,5,0,21.0,5.2,0.25",
            ]
        )
    )
    config_path = tmp_path / "logreg.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: logreg_v1",
                "model_kind: baseline",
                "group_key: fly_id",
                "random_seed: 7",
                "learning_rate: 0.1",
                "max_iter: 300",
                "l2_reg: 0.01",
                "exclude_feature_names: n_segments,n_segments_with_qc_flags",
            ]
        )
    )

    output_dir = tmp_path / "cv_run"
    metadata = train_and_save_cross_validation_run(
        config_path=config_path,
        features_path=features_path,
        output_dir=output_dir,
        n_splits=3,
    )

    assert metadata["status"] == "completed"
    assert metadata["evaluation_kind"] == "grouped_stratified_k_fold_cv"
    assert metadata["n_folds"] == 3
    assert (output_dir / "cv_metrics_summary.json").exists()
    assert (output_dir / "cv_predictions.csv").exists()
    assert not (output_dir / "model_artifact.json").exists()

    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["n_folds"] == 3
    assert len(metrics["folds"]) == 3
    assert "valid_by_evidence_bin" in metrics["summary"]
    assert "high_n_segments_ge50" in metrics["summary"]["valid_by_evidence_bin"]

    predictions = (output_dir / "cv_predictions.csv").read_text()
    assert predictions.startswith("fold,split,fly_id")
