from __future__ import annotations

import json

from flygen_ml.cli import summarize_sequence_runs


def _write_cv_run(path, *, seed: int, joint_values: tuple[float, float]) -> None:
    path.mkdir()
    (path / "run_metadata.json").write_text(json.dumps({"random_seed": seed}))
    (path / "cv_metrics_summary.json").write_text(
        json.dumps(
            {
                "model_kind": "sequence_conv1d_meanpool_torch_v1",
                "random_seed": seed,
                "n_folds": 2,
                "folds": [
                    {
                        "fold": 0,
                        "valid": {
                            "n_examples": 5,
                            "joint_accuracy": joint_values[0],
                            "genotype": {"accuracy": 0.6, "balanced_accuracy": 0.55},
                            "cohort": {"accuracy": 0.7, "balanced_accuracy": 0.65},
                        },
                    },
                    {
                        "fold": 1,
                        "valid": {
                            "n_examples": 5,
                            "joint_accuracy": joint_values[1],
                            "genotype": {"accuracy": 0.8, "balanced_accuracy": 0.75},
                            "cohort": {"accuracy": 0.9, "balanced_accuracy": 0.85},
                        },
                    },
                ],
            }
        )
    )


def test_summarize_sequence_runs_prints_across_run_summary(monkeypatch, tmp_path, capsys):
    run0 = tmp_path / "run0"
    run1 = tmp_path / "run1"
    _write_cv_run(run0, seed=0, joint_values=(0.4, 0.6))
    _write_cv_run(run1, seed=1, joint_values=(0.6, 0.8))
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_sequence_runs",
            "--run-dir",
            str(run0),
            "--run-dir",
            str(run1),
        ],
    )

    assert summarize_sequence_runs.main() == 0

    out = capsys.readouterr().out
    assert "n_runs: 2" in out
    assert "run 0: seed=0 valid_joint_accuracy=0.500" in out
    assert "run 1: seed=1 valid_joint_accuracy=0.700" in out
    assert "valid_joint_accuracy: mean=0.600, std=0.100, min=0.500, max=0.700, n=2" in out
    assert "valid_genotype_balanced_accuracy: mean=0.650" in out


def test_summarize_sequence_runs_writes_json(monkeypatch, tmp_path, capsys):
    run0 = tmp_path / "run0"
    _write_cv_run(run0, seed=3, joint_values=(0.4, 0.6))
    output_json = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_sequence_runs",
            "--run-dir",
            str(run0),
            "--output-json",
            str(output_json),
            "--json",
        ],
    )

    assert summarize_sequence_runs.main() == 0

    payload = json.loads(output_json.read_text())
    assert payload["n_runs"] == 1
    assert payload["runs"][0]["random_seed"] == 3
    assert payload["runs"][0]["valid_joint_accuracy"] == 0.5
    assert json.loads(capsys.readouterr().out)["summary"]["valid_joint_accuracy"]["mean"] == 0.5
