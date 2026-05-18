from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from flygen_ml.modeling.sequence_training import train_and_save_sequence_cross_validation_run

torch = pytest.importorskip("torch")


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


def test_torch_segment_chain_encoder_batches_multiple_windows():
    from flygen_ml.modeling.torch_sequence_models import _build_module

    module = _build_module(
        n_channels=2,
        conv_channels=2,
        embedding_dim=4,
        n_side_features=0,
        fusion_hidden_dim=5,
        pooling="mean",
        genotype_pooling=None,
        cohort_pooling=None,
        sequence_unit="segment_chain",
        chain_length=2,
        chain_stride=1,
        gru_hidden_dim=4,
        gru_layers=1,
        gru_bidirectional=False,
        attention_hidden_dim=3,
        n_genotype_classes=2,
        n_cohort_classes=2,
        dropout=0.0,
    )
    segment_embeddings = torch.randn(4, 4)

    chain_embeddings = module.encode_chains(segment_embeddings)

    assert chain_embeddings.shape == (3, 4)


def test_torch_segment_gru_encoder_preserves_ordered_segment_count():
    from flygen_ml.modeling.torch_sequence_models import _build_module

    module = _build_module(
        n_channels=2,
        conv_channels=2,
        embedding_dim=4,
        n_side_features=0,
        fusion_hidden_dim=5,
        pooling="mean",
        genotype_pooling=None,
        cohort_pooling=None,
        sequence_unit="segment_gru",
        chain_length=1,
        chain_stride=1,
        gru_hidden_dim=5,
        gru_layers=1,
        gru_bidirectional=False,
        attention_hidden_dim=3,
        n_genotype_classes=2,
        n_cohort_classes=2,
        dropout=0.0,
    )
    segment_embeddings = torch.randn(4, 4)

    gru_embeddings = module.encode_gru(segment_embeddings)

    assert gru_embeddings.shape == (4, 5)


def test_export_torch_sequence_embeddings_returns_segment_and_gru_unit_views():
    from flygen_ml.modeling.torch_sequence_models import _build_module, export_torch_sequence_embeddings
    from flygen_ml.modeling.sequence_models import FlySequenceExample

    module = _build_module(
        n_channels=2,
        conv_channels=2,
        embedding_dim=4,
        n_side_features=0,
        fusion_hidden_dim=5,
        pooling="mean",
        genotype_pooling=None,
        cohort_pooling=None,
        sequence_unit="segment_gru",
        chain_length=1,
        chain_stride=1,
        gru_hidden_dim=5,
        gru_layers=1,
        gru_bidirectional=False,
        attention_hidden_dim=3,
        n_genotype_classes=2,
        n_cohort_classes=2,
        dropout=0.0,
    )
    x = np.random.default_rng(4).normal(size=(3, 8, 2)).astype(np.float32)
    examples = [
        FlySequenceExample(
            fly_id="a0",
            sample_key="s_a0",
            genotype="A",
            cohort="intact",
            segment_indices=np.asarray([0, 1, 2]),
            n_segments=3,
            n_segments_with_qc_flags=0,
        )
    ]
    model = {
        "module": module,
        "device": "cpu",
        "input_means": np.zeros(2, dtype=np.float32),
        "input_stds": np.ones(2, dtype=np.float32),
        "embedding_dim": 4,
        "sequence_unit": "segment_gru",
        "gru_hidden_dim": 5,
        "gru_bidirectional": False,
        "eval_max_segments_per_fly": 0,
    }

    payload = export_torch_sequence_embeddings(x, examples, model=model, embedding_kind="both")

    assert payload["segment_embeddings"].shape == (3, 4)
    assert payload["unit_embeddings"].shape == (3, 5)
    assert [row["segment_index"] for row in payload["rows"]] == [0, 1, 2]
    assert [row["segment_position_in_fly"] for row in payload["rows"]] == [0, 1, 2]


def _write_feature_fixture(path):
    path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,cohort,chamber_type,training_idx,n_segments,n_segments_with_qc_flags,path_length_px_mean,straightness_mean",
                "a0,s_a0,A,intact,large,1,2,0,1.0,0.1",
                "a1,s_a1,A,intact,large,1,2,0,1.1,0.2",
                "a2,s_a2,A,intact,large,1,2,0,0.9,0.3",
                "b0,s_b0,B,removed,large,1,2,1,3.0,0.8",
                "b1,s_b1,B,removed,large,1,2,1,3.1,0.9",
                "b2,s_b2,B,removed,large,1,2,1,2.9,0.7",
            ]
        )
    )


def test_torch_sequence_cross_validation_run_writes_fly_level_outputs(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    _write_sequence_fixture(sequence_path)
    config_path = tmp_path / "segment_conv1d.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_conv1d_meanpool_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
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
    output_dir = tmp_path / "run"

    metadata = train_and_save_sequence_cross_validation_run(
        config_path=config_path,
        sequence_path=sequence_path,
        output_dir=output_dir,
        n_splits=3,
    )

    assert metadata["status"] == "completed"
    assert metadata["model_kind"] == "sequence_conv1d_meanpool_torch_v1"
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["model_kind"] == "sequence_conv1d_meanpool_torch_v1"
    assert metrics["training"]["conv_channels"] == 2
    assert metrics["training"]["embedding_dim"] == 4

    with (output_dir / "cv_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    valid_predictions = [row for row in predictions if row["split"] == "valid"]
    assert len(valid_predictions) == 6
    assert {row["fly_id"] for row in valid_predictions} == {"a0", "a1", "a2", "b0", "b1", "b2"}


def test_torch_sequence_cross_validation_supports_side_features(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    features_path = tmp_path / "features.csv"
    _write_sequence_fixture(sequence_path)
    _write_feature_fixture(features_path)
    config_path = tmp_path / "segment_conv1d_fused.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_conv1d_meanpool_fused_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "conv_channels: 2",
                "embedding_dim: 4",
                "fusion_hidden_dim: 5",
                "dropout: 0.0",
                "train_max_segments_per_fly: 1",
                "eval_max_segments_per_fly: 0",
                f"side_features_path: {features_path}",
                "side_feature_names: path_length_px_mean,straightness_mean",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "device: cpu",
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

    assert metadata["n_side_features"] == 2
    assert metadata["side_feature_names"] == ["path_length_px_mean", "straightness_mean"]
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["training"]["fusion_hidden_dim"] == 5
    assert metrics["training"]["side_feature_names"] == ["path_length_px_mean", "straightness_mean"]


def test_torch_sequence_cross_validation_supports_attention_pooling(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    features_path = tmp_path / "features.csv"
    _write_sequence_fixture(sequence_path)
    _write_feature_fixture(features_path)
    config_path = tmp_path / "segment_conv1d_attention.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_conv1d_attnpool_fused_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "conv_channels: 2",
                "embedding_dim: 4",
                "fusion_hidden_dim: 5",
                "pooling: attention",
                "attention_hidden_dim: 3",
                "dropout: 0.0",
                "train_max_segments_per_fly: 1",
                "eval_max_segments_per_fly: 0",
                f"side_features_path: {features_path}",
                "side_feature_names: path_length_px_mean,straightness_mean",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "device: cpu",
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

    assert metadata["pooling"] == "attention"
    assert metadata["attention_hidden_dim"] == 3
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["training"]["pooling"] == "attention"
    assert metrics["training"]["attention_hidden_dim"] == 3


def test_torch_sequence_cross_validation_supports_mean_attention_concat_pooling(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    features_path = tmp_path / "features.csv"
    _write_sequence_fixture(sequence_path)
    _write_feature_fixture(features_path)
    config_path = tmp_path / "segment_conv1d_mean_attention.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_conv1d_mean_attnpool_fused_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "conv_channels: 2",
                "embedding_dim: 4",
                "fusion_hidden_dim: 5",
                "pooling: mean_attention_concat",
                "attention_hidden_dim: 3",
                "dropout: 0.0",
                "train_max_segments_per_fly: 1",
                "eval_max_segments_per_fly: 0",
                f"side_features_path: {features_path}",
                "side_feature_names: path_length_px_mean,straightness_mean",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "device: cpu",
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

    assert metadata["pooling"] == "mean_attention_concat"
    assert metadata["attention_hidden_dim"] == 3
    assert metadata["pooled_embedding_dim"] == 8
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["training"]["pooling"] == "mean_attention_concat"
    assert metrics["training"]["attention_hidden_dim"] == 3
    assert metrics["training"]["pooled_embedding_dim"] == 8


def test_torch_sequence_cross_validation_supports_head_specific_pooling(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    features_path = tmp_path / "features.csv"
    _write_sequence_fixture(sequence_path)
    _write_feature_fixture(features_path)
    config_path = tmp_path / "segment_conv1d_headpool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_conv1d_headpool_fused_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "conv_channels: 2",
                "embedding_dim: 4",
                "fusion_hidden_dim: 5",
                "genotype_pooling: mean",
                "cohort_pooling: mean_attention_concat",
                "attention_hidden_dim: 3",
                "dropout: 0.0",
                "train_max_segments_per_fly: 1",
                "eval_max_segments_per_fly: 0",
                f"side_features_path: {features_path}",
                "side_feature_names: path_length_px_mean,straightness_mean",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "device: cpu",
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

    assert metadata["pooling"] == "mean"
    assert metadata["genotype_pooling"] == "mean"
    assert metadata["cohort_pooling"] == "mean_attention_concat"
    assert metadata["genotype_pooled_embedding_dim"] == 4
    assert metadata["cohort_pooled_embedding_dim"] == 8
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["training"]["genotype_pooling"] == "mean"
    assert metrics["training"]["cohort_pooling"] == "mean_attention_concat"
    assert metrics["training"]["genotype_pooled_embedding_dim"] == 4
    assert metrics["training"]["cohort_pooled_embedding_dim"] == 8


def test_torch_sequence_cross_validation_supports_segment_chain_units(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    features_path = tmp_path / "features.csv"
    _write_sequence_fixture(sequence_path)
    _write_feature_fixture(features_path)
    config_path = tmp_path / "segment_chain_conv1d_headpool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_chain_conv1d_headpool_fused_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "conv_channels: 2",
                "embedding_dim: 4",
                "fusion_hidden_dim: 5",
                "sequence_unit: segment_chain",
                "chain_length: 2",
                "chain_stride: 1",
                "genotype_pooling: mean",
                "cohort_pooling: mean_attention_concat",
                "attention_hidden_dim: 3",
                "dropout: 0.0",
                "train_max_segments_per_fly: 2",
                "eval_max_segments_per_fly: 0",
                f"side_features_path: {features_path}",
                "side_feature_names: path_length_px_mean,straightness_mean",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "progress_interval: 1",
                "validation_interval: 1",
                "select_best_epoch: true",
                "validation_monitor_metric: joint_accuracy",
                "device: cpu",
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

    assert metadata["sequence_unit"] == "segment_chain"
    assert metadata["chain_length"] == 2
    assert metadata["chain_stride"] == 1
    assert metadata["segment_sampling"] == "random_contiguous_span_per_epoch"
    assert metadata["progress_interval"] == 1
    assert metadata["validation_interval"] == 1
    assert metadata["select_best_epoch"] is True
    assert metadata["best_epoch"] == 1
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["training"]["sequence_unit"] == "segment_chain"
    assert metrics["training"]["chain_length"] == 2
    assert metrics["training"]["chain_stride"] == 1
    assert metrics["training"]["segment_sampling"] == "random_contiguous_span_per_epoch"
    assert metrics["training"]["progress_interval"] == 1
    assert metrics["training"]["validation_interval"] == 1
    assert metrics["training"]["select_best_epoch"] is True
    assert metrics["training"]["best_epoch"] == 1
    assert metrics["folds"][0]["training"]["best_epoch"] == 1
    assert metrics["folds"][0]["training"]["validation_history"][0]["epoch"] == 1


def test_torch_sequence_cross_validation_supports_segment_gru_units(tmp_path):
    sequence_path = tmp_path / "sequences.npz"
    features_path = tmp_path / "features.csv"
    _write_sequence_fixture(sequence_path)
    _write_feature_fixture(features_path)
    config_path = tmp_path / "segment_gru_conv1d_headpool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: segment_gru_conv1d_headpool_fused_v1",
                "model_kind: sequence_conv1d_meanpool_torch_v1",
                "split_label_key: genotype",
                "random_seed: 3",
                "conv_channels: 2",
                "embedding_dim: 4",
                "fusion_hidden_dim: 5",
                "sequence_unit: segment_gru",
                "gru_hidden_dim: 3",
                "gru_layers: 1",
                "gru_bidirectional: false",
                "genotype_pooling: mean",
                "cohort_pooling: mean_attention_concat",
                "attention_hidden_dim: 3",
                "dropout: 0.0",
                "train_max_segments_per_fly: 2",
                "eval_max_segments_per_fly: 0",
                f"side_features_path: {features_path}",
                "side_feature_names: path_length_px_mean,straightness_mean",
                "learning_rate: 0.001",
                "max_iter: 1",
                "weight_decay: 0.0",
                "device: cpu",
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

    assert metadata["sequence_unit"] == "segment_gru"
    assert metadata["gru_hidden_dim"] == 3
    assert metadata["gru_layers"] == 1
    assert metadata["gru_bidirectional"] is False
    assert metadata["genotype_pooled_embedding_dim"] == 3
    assert metadata["cohort_pooled_embedding_dim"] == 6
    assert metadata["segment_sampling"] == "random_contiguous_span_per_epoch"
    metrics = json.loads((output_dir / "cv_metrics_summary.json").read_text())
    assert metrics["training"]["sequence_unit"] == "segment_gru"
    assert metrics["training"]["gru_hidden_dim"] == 3
    assert metrics["training"]["gru_layers"] == 1
    assert metrics["training"]["gru_bidirectional"] is False
    assert metrics["training"]["genotype_pooled_embedding_dim"] == 3
    assert metrics["training"]["cohort_pooled_embedding_dim"] == 6
    assert metrics["training"]["segment_sampling"] == "random_contiguous_span_per_epoch"
