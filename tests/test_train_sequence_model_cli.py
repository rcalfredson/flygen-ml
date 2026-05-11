from __future__ import annotations

from flygen_ml.cli import train_sequence_model


def test_train_sequence_model_cli_passes_seed_to_cv_training(monkeypatch):
    calls = []

    def fake_train_and_save_sequence_cross_validation_run(**kwargs):
        calls.append(kwargs)
        return {
            "model_kind": "sequence_conv1d_meanpool_torch_v1",
            "n_folds": 5,
        }

    monkeypatch.setattr(
        train_sequence_model,
        "train_and_save_sequence_cross_validation_run",
        fake_train_and_save_sequence_cross_validation_run,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_sequence_model",
            "--config",
            "config.yaml",
            "--sequences",
            "sequences.npz",
            "--output",
            "run",
            "--cv-folds",
            "5",
            "--seed",
            "7",
        ],
    )

    assert train_sequence_model.main() == 0

    assert calls == [
        {
            "config_path": "config.yaml",
            "sequence_path": "sequences.npz",
            "output_dir": "run",
            "n_splits": 5,
            "random_seed": 7,
        }
    ]


def test_train_sequence_model_cli_passes_seed_to_holdout_training(monkeypatch):
    calls = []

    def fake_train_and_save_sequence_run(**kwargs):
        calls.append(kwargs)
        return {
            "model_kind": "sequence_conv1d_meanpool_torch_v1",
            "train_flies": 10,
            "valid_flies": 4,
        }

    monkeypatch.setattr(
        train_sequence_model,
        "train_and_save_sequence_run",
        fake_train_and_save_sequence_run,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_sequence_model",
            "--config",
            "config.yaml",
            "--sequences",
            "sequences.npz",
            "--output",
            "run",
            "--seed",
            "9",
        ],
    )

    assert train_sequence_model.main() == 0

    assert calls == [
        {
            "config_path": "config.yaml",
            "sequence_path": "sequences.npz",
            "output_dir": "run",
            "random_seed": 9,
        }
    ]
