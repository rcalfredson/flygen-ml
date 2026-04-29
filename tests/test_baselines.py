from __future__ import annotations

from flygen_ml.modeling.baselines import predict_fly_level_baseline, train_fly_level_baseline


def test_train_fly_level_baseline_supports_multiclass_labels():
    rows = [
        {"fly_id": "a0", "sample_key": "s0", "genotype": "A", "feature": 0.0},
        {"fly_id": "a1", "sample_key": "s1", "genotype": "A", "feature": 0.2},
        {"fly_id": "b0", "sample_key": "s2", "genotype": "B", "feature": 4.0},
        {"fly_id": "b1", "sample_key": "s3", "genotype": "B", "feature": 4.2},
        {"fly_id": "c0", "sample_key": "s4", "genotype": "C", "feature": 8.0},
        {"fly_id": "c1", "sample_key": "s5", "genotype": "C", "feature": 8.2},
    ]

    model = train_fly_level_baseline(
        rows,
        config={"learning_rate": 0.2, "max_iter": 1000, "l2_reg": 0.0},
    )
    predictions = predict_fly_level_baseline(rows, model=model)

    assert model["model_kind"] == "softmax_logreg_numpy_v1"
    assert model["label_key"] == "genotype"
    assert model["labels"] == ["A", "B", "C"]
    assert [row["predicted_label"] for row in predictions] == ["A", "A", "B", "B", "C", "C"]
    assert [row["predicted_genotype"] for row in predictions] == ["A", "A", "B", "B", "C", "C"]
    assert all(0.0 <= float(row["predicted_probability"]) <= 1.0 for row in predictions)


def test_train_fly_level_baseline_supports_non_genotype_label_key():
    rows = [
        {"fly_id": "a0", "sample_key": "s0", "genotype": "G0", "cohort": "intact", "feature": 0.0},
        {"fly_id": "a1", "sample_key": "s1", "genotype": "G1", "cohort": "intact", "feature": 0.2},
        {"fly_id": "b0", "sample_key": "s2", "genotype": "G0", "cohort": "removed", "feature": 4.0},
        {"fly_id": "b1", "sample_key": "s3", "genotype": "G1", "cohort": "removed", "feature": 4.2},
    ]

    model = train_fly_level_baseline(
        rows,
        config={"label_key": "cohort", "learning_rate": 0.2, "max_iter": 1000, "l2_reg": 0.0},
    )
    predictions = predict_fly_level_baseline(rows, model=model)

    assert model["label_key"] == "cohort"
    assert model["labels"] == ["intact", "removed"]
    assert "cohort" not in model["feature_names"]
    assert "genotype" not in model["feature_names"]
    assert [row["label_key"] for row in predictions] == ["cohort"] * 4
    assert [row["actual_label"] for row in predictions] == ["intact", "intact", "removed", "removed"]
    assert [row["predicted_label"] for row in predictions] == ["intact", "intact", "removed", "removed"]
    assert all("actual_genotype" not in row for row in predictions)
