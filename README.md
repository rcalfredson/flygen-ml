# flygen-ml

Machine-learning tooling for asking whether a fly's trajectory carries enough
information to identify labels such as genotype, antenna condition, cohort, or
other manifest metadata.

The current pipeline focuses on between-reward trajectories from paired `.data`
and `.trx` files. It extracts canonical trajectory segments, builds engineered
movement features, exports low-level trajectory tensors, and trains fly-level
classifiers with grouped evaluation.

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

The project supports two complementary modeling paths:

- engineered fly-level baselines: per-segment movement summaries aggregated per
  fly, then classified with NumPy logistic regression / softmax regression
- sequence models: fixed-length, reward-normalized trajectory tensors encoded at
  the segment level, aggregated at the fly level, and classified with dual
  genotype/cohort output heads

The first target was genotype classification. The modeling path now also
supports `cohort` for antennae-intact vs antennae-removed classification, and the
strongest sequence models predict both axes jointly.

The current strongest model family is GRU-128 over ordered Conv1D segment
embeddings, with head-specific fly-level pooling and behavior-derived side
features. Across grouped CV seed sweeps, the current results are:

| trajectory evidence | model | joint | genotype | genotype bal | cohort | cohort bal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Training 2 only | engineered logreg | 0.664 | 0.738 | 0.724 | 0.900 | 0.897 |
| Training 2 only | GRU-128 | 0.764 | 0.788 | 0.779 | 0.948 | 0.949 |
| Training 1 only | engineered logreg | 0.659 | 0.794 | 0.783 | 0.857 | 0.854 |
| Training 1 only | GRU-128 | **0.809** | **0.861** | **0.856** | 0.929 | 0.928 |
| Training 1 + Training 2 | engineered logreg | 0.677 | 0.774 | 0.757 | 0.882 | 0.881 |
| Training 1 + Training 2 | GRU-128 | 0.803 | 0.833 | 0.827 | **0.954** | **0.954** |

Training 1-only is currently strongest for genotype and joint classification.
Combined Training 1 + Training 2 evidence is currently strongest for
antenna-condition/cohort classification. In all tested regimes, GRU-128
outperforms the engineered logistic-regression baseline.

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

For Training 1 runs, use the Training 1 glob spec and a separate manifest path:

```bash
python -m flygen_ml.cli.build_manifest_from_globs \
  --spec configs/manifest_globs/yang_2025_antennae_kir_t1.csv \
  --output artifacts/manifest_t1.csv \
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

For Training 1:

```bash
python -m flygen_ml.cli.extract_segments \
  --config configs/dataset/v1_binary.yaml \
  --manifest artifacts/manifest_t1.csv \
  --output artifacts/segments_t1_with_cohort.csv
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

For Training 1:

```bash
python -m flygen_ml.cli.build_features \
  --feature-set engineered_v1 \
  --segments artifacts/segments_t1_with_cohort.csv \
  --output artifacts/features_t1_antennae_no_training_end.csv
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

The combined Training 1 + Training 2 feature table used in the current
experiments is one row per fly in `artifacts/features_t1_t2_antennae_no_training_end.csv`.
It should be built by combining the T1 and T2 fly-level feature rows into a
single fly-level row, not by treating T1 and T2 as independent labeled examples.

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

### 8. Export Trajectory Tensors

The sequence-model path uses the segment table as an index back into the raw
`.trx` trajectories. It exports fixed-length, reward-normalized per-segment
tensors while preserving fly labels for fly-level aggregation.

```bash
python -m flygen_ml.cli.export_sequence_tensors \
  --segments artifacts/segments_with_cohort.csv \
  --output artifacts/sequences_v1.npz \
  --target-length 128
```

For Training 1:

```bash
python -m flygen_ml.cli.export_sequence_tensors \
  --segments artifacts/segments_t1_with_cohort.csv \
  --output artifacts/sequences_t1_v1.npz \
  --target-length 128
```

The default channels are `x_rel`, `y_rel`, `dx_rel`, `dy_rel`, `speed_rel`, and
`r_rel`, where positions are centered on the reward location and scaled by the
reward radius.

The combined Training 1 + Training 2 sequence artifact used in the current
experiments is `artifacts/sequences_t1_t2_v1.npz`. It combines existing T1 and
T2 segment tensors while preserving a single fly-level prediction unit.

To export a richer per-timestep motion representation, use `--channel-set rich`.
This keeps the same fly-level training path but adds channels for acceleration,
radial/tangential motion, heading, and turning:

```bash
python -m flygen_ml.cli.export_sequence_tensors \
  --segments artifacts/segments_with_cohort.csv \
  --output artifacts/sequences_rich_v1.npz \
  --target-length 128 \
  --channel-set rich
```

### 9. Train A Fly-Level Sequence Model

The first sequence model is a small NumPy baseline: flatten each segment tensor,
encode it with a one-hidden-layer MLP, mean-pool segment embeddings per fly, and
predict both `genotype` and `cohort` with separate softmax heads.
During training, it randomly samples up to `train_max_segments_per_fly` segments
per fly each epoch. During evaluation, `eval_max_segments_per_fly: 0` means use
all available segments for each fly.

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_meanpool.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_meanpool_v1_cv \
  --cv-folds 5
```

Sequence predictions are one row per fly, not one row per segment, and include
both genotype and cohort predictions plus joint correctness.

To print a compact summary of a sequence run:

```bash
python -m flygen_ml.cli.evaluate_sequence_model \
  --run-dir runs/segment_meanpool_v1_cv
```

For a stronger segment encoder, install PyTorch in the active environment and
train the Conv1D sequence model:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_conv1d_meanpool.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_conv1d_meanpool_v1_cv \
  --cv-folds 5
```

To concatenate engineered fly-level movement summaries onto the learned
trajectory embedding before the prediction heads:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_conv1d_meanpool_fused.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_conv1d_meanpool_fused_v1_cv \
  --cv-folds 5
```

Use `--seed` to override `random_seed` from the config without creating a
seed-specific config file:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_conv1d_headpool_fused_wide.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_conv1d_headpool_fused_wide_v1_seed1_cv \
  --cv-folds 5 \
  --seed 1
```

To test short ordered chains of consecutive between-reward trajectories instead
of pooling individual segment embeddings directly. The chain config prints
fold/epoch progress every five epochs because these runs are heavier than the
single-segment model:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_chain_conv1d_headpool_fused_wide.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_chain_conv1d_headpool_fused_wide_v1_cv \
  --cv-folds 5
```

To run a longer chain experiment with validation monitoring and best-epoch
restoration:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_chain_conv1d_headpool_fused_wide_long.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_chain_conv1d_headpool_fused_wide_long_v1_cv \
  --cv-folds 5
```

To run an ordered-segment GRU experiment, which encodes individual trajectories
with the Conv1D segment encoder and then passes the ordered segment embeddings
through a GRU before fly-level pooling:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_gru_conv1d_headpool_fused_wide_long.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_gru_conv1d_headpool_fused_wide_long_v1_cv \
  --cv-folds 5
```

The current GRU-128 configs use the same ordered-segment architecture with a
larger recurrent hidden state. Training 1-only is the strongest current setup
for genotype and joint accuracy:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_gru128_t1_conv1d_headpool_fused_wide_long.yaml \
  --sequences artifacts/sequences_t1_v1.npz \
  --output runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_cv \
  --cv-folds 5
```

Combined Training 1 + Training 2 is the strongest current setup for cohort:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_gru128_t1_t2_conv1d_headpool_fused_wide_long.yaml \
  --sequences artifacts/sequences_t1_t2_v1.npz \
  --output runs/segment_gru128_t1_t2_conv1d_headpool_fused_wide_long_v1_cv \
  --cv-folds 5
```

To repeat seed sweeps without creating seed-specific config files:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_gru128_t1_conv1d_headpool_fused_wide_long.yaml \
  --sequences artifacts/sequences_t1_v1.npz \
  --output runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_seed1_cv \
  --cv-folds 5 \
  --seed 1
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

Sequence CV runs can be summarized one at a time:

```bash
python -m flygen_ml.cli.evaluate_sequence_model \
  --run-dir runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_cv
```

To summarize repeated sequence CV runs across seeds:

```bash
python -m flygen_ml.cli.summarize_sequence_runs \
  --run-dir runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_cv \
  --run-dir runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_seed1_cv \
  --run-dir runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_seed2_cv \
  --run-dir runs/segment_gru128_t1_conv1d_headpool_fused_wide_long_v1_seed3_cv
```

Use `--output-json` to save the across-seed summary as JSON.

Evaluate whether two independently trained fly-level classifiers jointly
identify each fly:

```bash
python -m flygen_ml.cli.evaluate_joint_predictions \
  --axis-a-run runs/logreg_v1_movement_only_genotype_cv \
  --axis-b-run runs/logreg_v1_movement_only_antennae_condition_cv \
  --axis-a-name genotype \
  --axis-b-name cohort \
  --split valid \
  --join-without-fold \
  --output runs/joint_genotype_cohort_eval/joint_predictions.csv
```

The joint evaluator auto-detects `cv_predictions.csv` or `predictions.csv`
inside run directories, or accepts explicit prediction CSVs with
`--axis-a-predictions` and `--axis-b-predictions`. Rows are joined by
`fly_id`, `sample_key`, `split` when present on both axes, and `fold` when
present on both axes. Use `--join-without-fold` for out-of-fold validation
predictions from CV runs whose fold assignments are not aligned across label
axes.

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

Export a compact prediction review table joined to fly-level metadata:

```bash
python -m flygen_ml.cli.inspect_predictions \
  --run-dir runs/logreg_v1_movement_only_antennae_condition_cv \
  --output runs/logreg_v1_movement_only_antennae_condition_cv/valid_error_review.csv \
  --errors-only
```

Add `--include-features` to include all feature columns from the feature table.

Export all between-reward segment rows for selected prediction-review flies:

```bash
python -m flygen_ml.cli.export_prediction_segments \
  --prediction-review runs/logreg_v1_movement_only_antennae_condition_cv/valid_error_review.csv \
  --segments artifacts/segments_with_cohort.csv \
  --output runs/logreg_v1_movement_only_antennae_condition_cv/high_confidence_error_segments.csv \
  --errors-only \
  --min-decision-margin 0.30
```

This creates a plotting-ready segment table with prediction metadata prepended to
each segment row.

### Segment Occlusion For Sequence Models

For trained PyTorch sequence holdout runs, segment occlusion estimates how much
each between-reward trajectory contributes to a fly-level prediction. The
workflow removes one segment at a time from each selected fly, reruns the model,
and records probability/logit deltas for the genotype and cohort heads.

First train a holdout sequence model so that `model_artifact.json` is available:

```bash
python -m flygen_ml.cli.train_sequence_model \
  --config configs/model/segment_gru128_conv1d_headpool_fused_wide_long.yaml \
  --sequences artifacts/sequences_v1.npz \
  --output runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion \
  --seed 0
```

Then run leave-one-segment-out occlusion on the validation flies:

```bash
python -m flygen_ml.cli.explain_sequence_occlusion \
  --run-dir runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion \
  --sequence-path artifacts/sequences_v1.npz \
  --output-csv runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/segment_occlusion_valid.csv \
  --split valid
```

The output contains baseline and occluded probabilities/logits, predicted and
actual class deltas, and flags indicating whether removing a segment changed the
genotype, cohort, or joint prediction. Deltas are computed as:

```text
baseline value - occluded value
```

Thus, a positive logit delta means the removed segment supported the tracked
class; a negative delta means the segment opposed the tracked class.

To prepare visual-review subsets, filter rows by fly type, output head, sign of
the logit delta, and optional correctness of the baseline prediction. For
example, to review PFN>Kir antennae-intact flies where the genotype prediction
was correct:

```bash
python -m flygen_ml.cli.filter_occlusion_segments \
  --occlusion-csv runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/segment_occlusion_valid.csv \
  --output-dir runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/structured_review/pfn_intact/genotype \
  --deficiency pfn-intact \
  --head genotype \
  --require-correct genotype \
  --max-segments-per-fly 3
```

Supported fly-type filters are:

- `pfn-intact`: `PFN>Kir` + `antennae-intact`
- `control-removed`: `Control>Kir` + `antennae-removed`
- `control-intact`: `Control>Kir` + `antennae-intact`

Supported correctness filters are `none`, `genotype`, `cohort`, and `joint`.
The filter command writes separate positive and negative logit-delta CSVs.
Positive rows support the tracked class; negative rows oppose it.

A typical structured review matrix is:

| fly type | head | correctness filter | question |
| --- | --- | --- | --- |
| `pfn-intact` | `genotype` | `genotype` | Which segments support or oppose the PFN>Kir genotype call when antennae are intact? |
| `control-removed` | `cohort` | `cohort` | Which segments support or oppose the antennae-removed call when genotype is control? |
| `control-intact` | `genotype` | `genotype` | Which segments support or oppose the Control>Kir genotype call in intact controls? |
| `control-intact` | `cohort` | `cohort` | Which segments support or oppose the antennae-intact call in intact controls? |

For quick dependency-light inspection, plot normalized tensor paths directly:

```bash
python -m flygen_ml.cli.plot_occlusion_segments \
  --occlusion-csv runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/structured_review/pfn_intact/genotype/pfn-intact_genotype_predicted_positive_logit_delta.csv \
  --sequence-path artifacts/sequences_v1.npz \
  --output-dir runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/structured_review/pfn_intact/genotype/quick_plots_positive \
  --change-head genotype \
  --include-unchanged
```

To plot with external trajectory-plotting tools that need original frame
metadata, join the filtered occlusion rows back to the segment table:

```bash
python -m flygen_ml.cli.export_occlusion_segments \
  --occlusion-csv runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/structured_review/pfn_intact/genotype/pfn-intact_genotype_predicted_positive_logit_delta.csv \
  --segments artifacts/segments_with_cohort.csv \
  --output runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_holdout_for_occlusion/structured_review/pfn_intact/genotype/pfn-intact_genotype_predicted_positive_plot_ready.csv
```

This plot-ready CSV preserves occlusion evidence columns and adds source segment
metadata such as `data_path`, `trx_path`, `experimental_fly_idx`,
`anchor_reward_frame`, and `end_reward_frame`.

Current limitation: occlusion currently explains saved holdout sequence runs.
Grouped CV runs do not yet save fold-specific model artifacts, so validation
flies from CV runs cannot yet be explained using the exact fold model that
predicted them. For stronger systematic analysis, repeat holdout runs across
seeds or extend CV training to save one model artifact per fold.

### Prediction Error Comparison

Compare the error sets from two saved prediction runs:

```bash
python -m flygen_ml.cli.compare_prediction_errors \
  --run-a runs/logreg_v1_movement_only_genotype_cv \
  --run-b runs/segment_gru128_conv1d_headpool_fused_wide_long_v1_cv \
  --axis genotype \
  --run-a-name logreg \
  --run-b-name gru128 \
  --join-without-fold \
  --output runs/error_analysis/logreg_vs_gru128_genotype.csv
```

Summarize correctness buckets after an error comparison:

```bash
python -m flygen_ml.cli.inspect_error_buckets \
  --comparison runs/error_analysis/logreg_vs_gru128_genotype.csv \
  --features artifacts/features_antennae_no_training_end.csv \
  --output runs/error_analysis/logreg_vs_gru128_genotype_summary.json \
  --examples-output runs/error_analysis/logreg_vs_gru128_genotype_examples.csv
```

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

Sequence models also protect evaluation at the fly level: per-segment
trajectories are evidence for a fly-level label, not independent labeled
examples. During training, configs such as `train_max_segments_per_fly: 200`
randomly sample segment evidence per fly each epoch when a fly has more than the
cap. During evaluation, `eval_max_segments_per_fly: 0` means use all available
segments for each fly.

Current interpretation notes:

- Training 1-only GRU-128 is strongest for genotype and joint accuracy.
- Training 1 + Training 2 GRU-128 is strongest for cohort accuracy.
- Adding more trajectory evidence does not automatically help every label axis;
  T1 and T2 appear to emphasize different behavioral signals.
- The preferred next inputs remain behavior-derived trajectory context features,
  rather than shortcut metadata used only to maximize classification.

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
python -m flygen_ml.cli.export_sequence_tensors --help
python -m flygen_ml.cli.train_sequence_model --help
python -m flygen_ml.cli.evaluate_sequence_model --help
python -m flygen_ml.cli.summarize_sequence_runs --help
python -m flygen_ml.cli.evaluate_joint_predictions --help
python -m flygen_ml.cli.compare_prediction_errors --help
python -m flygen_ml.cli.inspect_error_buckets --help
python -m flygen_ml.cli.inspect_segments --help
python -m flygen_ml.cli.inspect_misclassifications --help
python -m flygen_ml.cli.inspect_predictions --help
python -m flygen_ml.cli.export_prediction_segments --help
```

## Reference Docs

- [docs/source-data-contract.md](docs/source-data-contract.md)
- [docs/upstream-notes.md](docs/upstream-notes.md)
- [docs/implementation-spec.md](docs/implementation-spec.md)
- [docs/experiment-plan.md](docs/experiment-plan.md)
