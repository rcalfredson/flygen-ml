from __future__ import annotations

import csv
import json

import numpy as np

from flygen_ml.modeling.sequence_training import train_and_save_sequence_cross_validation_run


def _write_sequence_fixture(path):
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
            x_rows.append(np.full((4, 2), value + segment_idx * 0.01, dtype=np.float32))
            fly_ids.append(fly_id)
            sample_keys.append(f"s_{fly_id}")
            segment_ids.append(f"{fly_id}_seg{segment_idx}")
            genotypes.append(genotype)
            cohorts.append(cohort)
            qc_flags.append("" if segment_idx == 0 else "has_missing_frames")
    np.savez_compressed(
        path,
        x=np.stack(x_rows),
        mask=np.ones((len(x_rows), 4), dtype=bool),
        segment_id=np.asarray(segment_ids),
        sample_key=np.asarray(sample_keys),
        fly_id=np.asarray(fly_ids),
        genotype=np.asarray(genotypes),
        cohort=np.asarray(cohorts),
        qc_flags=np.asarray(qc_flags),
        channels=np.asarray(["x_rel", "y_rel"]),
        target_length=np.asarray(4),
    )


def test_train_and_save_sequence_cross_validation_run_writes_fly_level_outputs(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    _write_sequence_fixture(sequence_path)
    config_path = tmp_path / "segment_meanpool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_meanpool_v1",
                "model_kind: sequence_meanpool_mlp_numpy_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "hidden_dim: 4",
                "max_segments_per_fly: 2",
                "learning_rate: 0.05",
                "max_iter: 5",
                "l2_reg: 0.0",
            ]
        )
    )
    output_dir = tmp_path / "run"

    metadata = train_and_save_sequence_cross_validation_run(
        config_path=config_path,
        sequence_path=sequence_path,
        output_dir=output_dir,
        n_splits=3,
    )

    assert metadata["status"] == "completed"
    assert metadata["model_kind"] == "sequence_meanpool_mlp_numpy_v1"
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["n_folds"] == 3
    assert metrics["sequence_metadata"]["n_flies"] == 6

    with (output_dir / "cv_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    valid_predictions = [row for row in predictions if row["split"] == "valid"]
    assert len(valid_predictions) == 6
    assert {row["fly_id"] for row in valid_predictions} == {"a0", "a1", "a2", "b0", "b1", "b2"}
    assert "actual_genotype" in predictions[0]
    assert "actual_cohort" in predictions[0]
