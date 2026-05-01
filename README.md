# flygen-ml

Machine-learning tooling for asking whether a fly's trajectory carries enough
information to identify labels such as genotype, antenna condition, cohort, or
other manifest metadata.

The current pipeline focuses on between-reward trajectories from paired `.data`
and `.trx` files. It extracts canonical trajectory segments, builds engineered
movement features, aggregates those features to the fly level, and trains a
simple baseline classifier with grouped evaluation.

The package is organized as a self-contained pipeline for this task: it reads
paired `.data` and `.trx` recording files, extracts trajectory segments, builds
feature tables, and writes model artifacts without requiring notebook-specific
analysis state.

## Input Data

The pipeline expects paired `.data` and `.trx` files from the lab's trajectory
recording workflow. It does not depend on notebook state or external analysis
objects, but it is format-tethered to these files: loaders assume the recording
metadata, protocol fields, timestamps, and per-fly trajectory arrays used by that
workflow.

Each manifest row points to one `.data` / `.trx` pair and one experimental fly
within that recording. The loader uses those files to recover:

- selected training bounds
- experimental fly identity
- reward geometry and reward-entry / reward-exit events
- per-frame `x` / `y` trajectories for the selected fly
- manifest labels such as `genotype`, `cohort`, `chamber`, and `training_idx`

For the expected input contract, see
[docs/source-data-contract.md](docs/source-data-contract.md).

## Current Scope

The main supported path is a fly-level logistic-regression baseline:

- input: manifest rows pointing at paired `.data` / `.trx` recordings
- segmentation: canonical between-reward trajectory intervals
- features: engineered per-segment movement summaries aggregated per fly
- model: NumPy logistic regression / softmax regression baseline
- evaluation: grouped train/validation split or grouped K-fold CV
- target label: configurable with `label_key`, defaulting to `genotype`

The first target was genotype classification. The modeling path now also supports
other manifest columns, for example `cohort` for antennae-intact vs
antennae-removed classification.

## Installation

Use any Python environment you prefer: Conda, `venv`, `uv`, or another local
environment manager. From the repository root, install the package in editable
mode:

```bash
pip install -e .[dev]
```

This installs the runtime dependencies plus development and test dependencies
from `pyproject.toml`.

For a runtime-only install:

```bash
pip install -e .
```

After installation, CLI commands should work as `python -m flygen_ml...`. For a
temporary shell-only alternative without installing the package:

```bash
export PYTHONPATH="$PWD/src"
```

## End-to-End Pipeline

The typical workflow writes intermediate artifacts under `artifacts/` and model
runs under `runs/`.

### 1. Build A Manifest

If a manifest already exists, this step can be skipped. A manifest row identifies
one experimental fly in one recording and includes labels such as `genotype` and
`cohort`.

```bash
python -m flygen_ml.cli.build_manifest_from_globs \
  --spec configs/manifest_globs/yang_2025_antennae_kir_v1.csv \
  --output artifacts/manifest.csv \
  --repeat-fly-indices 0,1
```

Expected manifest columns include:

- `sample_key`
- `data_path`
- `trx_path`
- `genotype`
- `cohort`
- `date`
- `chamber`
- `training_idx`
- `fly_idx`

### 2. Extract Between-Reward Segments

```bash
python -m flygen_ml.cli.extract_segments \
  --config configs/dataset/v1_binary.yaml \
  --manifest artifacts/manifest.csv \
  --output artifacts/segments_with_cohort.csv
```

The segment table keeps source provenance and frame boundaries so raw trajectory
slices can be recovered later. It also carries manifest metadata such as
`genotype` and `cohort`.

### 3. Build Fly-Level Features

```bash
python -m flygen_ml.cli.build_features \
  --feature-set engineered_v1 \
  --segments artifacts/segments_with_cohort.csv \
  --output artifacts/features_antennae_no_training_end.csv
```

By default, feature building omits segments that ended only because the selected
training ended. To include those segments:

```bash
python -m flygen_ml.cli.build_features \
  --feature-set engineered_v1 \
  --segments artifacts/segments_with_cohort.csv \
  --output artifacts/features_with_training_end.csv \
  --include-training-end-segments
```

The feature table is one row per fly. Metadata columns such as `genotype`,
`cohort`, `chamber_type`, and `training_idx` are preserved for grouping and
labeling, but are excluded from numeric model features.

### 4. Train A Genotype Classifier

Genotype is the default target label.

```bash
python -m flygen_ml.cli.train_model \
  --config configs/model/logreg.yaml \
  --features artifacts/features_antennae_no_training_end.csv \
  --output runs/logreg_v1_movement_only
```

### 5. Train An Antenna-Condition Classifier

Use `--label-key cohort` to classify antennae-intact vs antennae-removed.

```bash
python -m flygen_ml.cli.train_model \
  --config configs/model/logreg.yaml \
  --features artifacts/features_antennae_no_training_end.csv \
  --output runs/logreg_v1_movement_only_antennae_condition \
  --label-key cohort
```

The same target can also be set in a config file:

```yaml
label_key: cohort
```

`target_key` is accepted as a backward-compatible alias, but `label_key` is the
preferred name.

### 6. Evaluate A Saved Run

```bash
python -m flygen_ml.cli.evaluate_model \
  --run-dir runs/logreg_v1_movement_only_antennae_condition
```

This prints a compact text summary. Use `--json` to print the raw saved metrics
payload. The command auto-detects holdout runs and grouped CV runs.

To include a confusion matrix and misclassified validation rows:

```bash
python -m flygen_ml.cli.evaluate_model \
  --run-dir runs/logreg_v1_movement_only_antennae_condition \
  --confusion \
  --misclassifications
```

### 7. Run Grouped Cross-Validation

```bash
python -m flygen_ml.cli.train_model \
  --config configs/model/logreg.yaml \
  --features artifacts/features_antennae_no_training_end.csv \
  --output runs/logreg_v1_movement_only_antennae_condition_cv \
  --label-key cohort \
  --cv-folds 5
```

The splitter groups by `group_key` from the model config, currently `fly_id`, and
stratifies using the active `label_key`.

The same evaluation command works for CV runs:

```bash
python -m flygen_ml.cli.evaluate_model \
  --run-dir runs/logreg_v1_movement_only_antennae_condition_cv \
  --confusion \
  --misclassifications
```

## Output Artifacts

A standard holdout run writes:

- `run_metadata.json`: paths, split sizes, labels, model kind, and active
  `label_key`
- `model_artifact.json`: trained weights, feature names, feature scaling values,
  labels, and active `label_key`
- `metrics_summary.json`: train/validation accuracy, balanced accuracy, per-label
  recall, support, and evidence-bin summaries
- `predictions.csv`: per-fly predictions with generic target fields

Prediction rows use:

- `label_key`
- `actual_label`
- `predicted_label`
- `predicted_probability`

For genotype-mode runs, the prediction output also includes
`actual_genotype` and `predicted_genotype` as backward-compatible aliases.

Cross-validation writes:

- `cv_metrics_summary.json`
- `cv_predictions.csv`
- `run_metadata.json`

## Inspection Helpers

Rank extracted segments by an engineered metric:

```bash
python -m flygen_ml.cli.inspect_segments \
  --segments artifacts/segments_with_cohort.csv \
  --metric end_radius_px \
  --limit 20 \
  --output artifacts/top_end_radius_segments.csv
```

Inspect misclassified validation flies and their largest feature contributors:

```bash
python -m flygen_ml.cli.inspect_misclassifications \
  --run-dir runs/logreg_v1_movement_only_antennae_condition \
  --output runs/logreg_v1_movement_only_antennae_condition/misclassified_valid_fly_inspection.csv
```

Use `--include-correct` to inspect all rows in the selected split.

## Modeling Notes

The baseline model automatically ignores known metadata columns, including
`fly_id`, `sample_key`, `genotype`, `cohort`, `chamber`, `chamber_type`,
`training_idx`, `date`, `fly_idx`, and prediction-label fields. Additional
numeric features can be excluded in `configs/model/logreg.yaml`:

```yaml
exclude_feature_names: n_segments,n_segments_with_qc_flags
```

The current split protects against fly-level leakage. It does not, by itself,
guarantee independence across recording date, experimental batch, chamber, or
other possible nuisance structure. For stronger claims, inspect those covariates
and prefer grouped cross-validation or stricter grouping strategies where
appropriate.

## Development

Run tests from the repository root:

```bash
pytest
```

Useful CLI entry points:

```bash
python -m flygen_ml.cli.build_manifest_from_globs --help
python -m flygen_ml.cli.extract_segments --help
python -m flygen_ml.cli.build_features --help
python -m flygen_ml.cli.train_model --help
python -m flygen_ml.cli.evaluate_model --help
python -m flygen_ml.cli.inspect_segments --help
python -m flygen_ml.cli.inspect_misclassifications --help
```

## Reference Docs

- [docs/source-data-contract.md](docs/source-data-contract.md)
- [docs/upstream-notes.md](docs/upstream-notes.md)
- [docs/implementation-spec.md](docs/implementation-spec.md)
- [docs/experiment-plan.md](docs/experiment-plan.md)
