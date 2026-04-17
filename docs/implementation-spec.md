# Implementation Specification

## Purpose

This document turns the planning notes into an implementation-ready v1 specification. It fixes module boundaries, internal data contracts, CLI flow, test scope, and fixture-validation requirements before the true loader and segmenter are implemented.

The emphasis is a clean, narrow v1 rather than early generalization.

## Module Responsibilities

### `flygen_ml/__init__.py`

- package version and top-level exports only
- no runtime side effects

### `flygen_ml/schema.py`

- internal dataclasses for normalized repo-level contracts
- shared types used across loaders, segment extraction, features, and modeling
- no file I/O

### `flygen_ml/data_manifest.py`

- manifest row parsing and validation
- manifest CSV read/write helpers
- path-pair validation at the metadata level
- no raw pickle loading

### `flygen_ml/qc.py`

- lightweight QC flag helpers
- reusable frame/segment/fly QC summaries
- no chamber-specific semantics

### `flygen_ml/loaders/pickle_loader.py`

- load `.data` and `.trx` pickle files
- support `latin1` compatibility behavior
- handle legacy pickled class references when needed
- return raw Python objects only
- TODO: fixture-verify upstream-compatible unpickling edge cases

### `flygen_ml/loaders/protocol_parser.py`

- parse `protocol` from `.data`
- validate `protocol["ct"]` against v1 chamber support
- use `protocol["frameNums"]` to derive experimental-fly structure
- extract selected training boundaries
- extract protocol-side reward metadata needed downstream
- TODO: fixture-verify the exact selected-training boundary semantics

### `flygen_ml/loaders/trajectory_builder.py`

- convert raw `.trx` arrays plus parsed protocol metadata into a normalized recording object
- infer FPS from timestamps when needed
- attach selected experimental fly index
- compute light trajectory QC summaries
- leave reward extraction to `segments/reward_events.py`

### `flygen_ml/segments/reward_events.py`

- define how actual reward metadata and calculated reward metadata are represented internally
- produce the canonical v1 calculated reward-entry stream for the experimental fly
- preserve optional actual-reward metadata for comparison and diagnostics
- TODO: fixture-verify actual-vs-calculated alignment expectations

### `flygen_ml/segments/between_reward.py`

- implement the canonical v1 segment rule:
  - anchor on calculated reward entry
  - start at first frame after reward-circle exit
  - stop at next calculated reward entry or training end
- emit normalized segment records and QC flags
- no feature engineering
- TODO: fixture-verify segment boundaries against trusted upstream expectations

### `flygen_ml/segments/normalization.py`

- transform segment trajectories into normalized model inputs
- reward-centered coordinates
- fixed-length resampling hooks for sequence models
- no model-specific code

### `flygen_ml/features/engineered.py`

- compute interpretable per-segment engineered features
- use normalized segment trajectories and metadata only
- no fly-level aggregation logic

### `flygen_ml/features/sequence.py`

- build fixed-length per-segment sequence tensors or array-like samples
- standardize channel ordering for v1
- no segment pooling or classifier logic

### `flygen_ml/features/aggregation.py`

- aggregate per-segment engineered features to fly-level feature rows
- produce fly-level counts and summary statistics
- no model fitting

### `flygen_ml/modeling/splits.py`

- grouped train/validation splitting
- leakage assertions
- optional stratification hooks
- no model fitting or data loading

### `flygen_ml/modeling/metrics.py`

- compute evaluation metrics
- fold-level and pooled summaries
- no plotting requirement in v1

### `flygen_ml/modeling/baselines.py`

- v1 fly-level classical baseline interfaces
- initial target is a simple engineered-feature fly-level classifier
- TODO: wire a concrete estimator only after dependencies and dataset scale are confirmed

### `flygen_ml/modeling/pooling.py`

- mean pooling and attention-pooling interfaces for future fly-level sequence models
- v1 scaffold only, not implemented

### `flygen_ml/modeling/sequence_models.py`

- segment encoder and fly-level pooling model interfaces
- v1 scaffold only, not implemented

### `flygen_ml/modeling/train.py`

- high-level training orchestration
- dispatch by config/model kind
- save run metadata and placeholder artifacts

### `flygen_ml/modeling/predict.py`

- prediction entry points for trained models
- v1 scaffold only

### `flygen_ml/cli/build_manifest.py`

- create or validate manifest CSV for paired raw files
- no raw semantic parsing

### `flygen_ml/cli/extract_segments.py`

- read manifest
- build normalized recording objects
- extract canonical v1 segments
- write segment table
- keep enough source provenance in the segment table to recover the original trajectory slice later without introducing a second storage layer in v1

### `flygen_ml/cli/build_features.py`

- build engineered or sequence-ready features from extracted segments
- for the first engineered baseline path, reduce each segment to interpretable engineered metrics and then aggregate those metrics to one fly-level row per experimental fly

### `flygen_ml/cli/train_model.py`

- train a fly-level baseline from features
- write run directory metadata

### `flygen_ml/cli/evaluate_model.py`

- load saved run outputs
- compute and print evaluation summaries

## Internal Data Contracts

### Manifest Row

Each manifest row describes one recording pair and one modeling label.

Required fields:
- `sample_key: str`
- `data_path: str`
- `trx_path: str`
- `genotype: str`
- `cohort: str | None`
- `date: str | None`
- `chamber: str`
- `training_idx: int`

Notes:
- `fly_idx` is optional at manifest time for v1 and should usually be derived from source data
- labels come from the manifest, not raw pickle internals

### Normalized Recording Object

Represents one paired recording after local loading and protocol parsing.

Required fields:
- `sample_key: str`
- `manifest: ManifestRow`
- `chamber_type: str`
- `experimental_fly_idx: int`
- `training_idx: int`
- `training_start_frame: int`
- `training_end_frame: int`
- `fps: float`
- `timestamps: sequence[float]`
- `x_by_fly`
- `y_by_fly`
- `protocol: dict`
- `raw_data: dict`
- `raw_trx: dict`
- `reward_context: dict`
- `qc_flags: tuple[str, ...]`

Notes:
- this object is the boundary between raw file handling and semantic extraction
- reward context may include both protocol-side reward metadata and local geometric metadata

### Segment Record

Represents one canonical between-reward sample.

Required fields:
- `segment_id: str`
- `sample_key: str`
- `fly_id: str`
- `genotype: str`
- `chamber_type: str`
- `experimental_fly_idx: int`
- `data_path: Path`
- `trx_path: Path`
- `training_idx: int`
- `training_start_frame: int`
- `training_end_frame: int`
- `anchor_reward_frame: int`
- `start_frame: int`
- `stop_frame: int`
- `end_reward_frame: int | None`
- `duration_frames: int`
- `n_finite_frames: int`
- `finite_frame_fraction: float`
- `qc_flags: tuple[str, ...]`

Recommended optional fields:
- `reward_center_x: float | None`
- `reward_center_y: float | None`
- `reward_radius: float | None`
- `terminated_by_training_end: bool`
- `anchor_reward_kind: str`

Notes:
- the v1 segment table is intentionally self-sufficient for a CSV-only workflow, so it carries source provenance and frame bounds in addition to semantic segment metadata
- the first fly-level baseline does not embed segment trajectories directly; it recomputes per-segment engineered features from these recoverable slices and aggregates them to fly-level summaries

### Fly-Level Engineered Row

Represents one experimental fly after aggregating engineered per-segment features.

Required fields:
- `fly_id: str`
- `sample_key: str`
- `genotype: str`
- `chamber_type: str`
- `training_idx: int`
- `n_segments: int`
- `n_segments_with_qc_flags: int`

Required aggregated engineered fields for the first pass:
- `duration_frames_mean`
- `finite_frame_fraction_mean`
- `path_length_px_mean`
- `net_displacement_px_mean`
- `straightness_mean`
- `mean_step_distance_px_mean`
- `mean_radius_px_mean`
- `radius_std_px_mean`
- `start_radius_px_mean`
- `end_radius_px_mean`
- `radius_delta_px_mean`

Notes:
- this row is the direct input to the first fly-level baseline
- it is intentionally summary-based and does not contain nested path data
- if sequence or pooling models are added later, they should consume segment-level recoverable slices or sequence tables rather than overloading this fly-level table

### Sequence Sample

Represents one model-ready segment sequence.

Required fields:
- `segment_id: str`
- `sample_key: str`
- `fly_id: str`
- `genotype: str`
- `group_id: str`
- `channels: tuple[str, ...]`
- `length: int`
- `mask`
- `x`

Notes:
- `x` is expected to be shaped like `[T, C]`
- `mask` is expected to be shaped like `[T]`
- `group_id` is the grouping unit used in splits, likely fly-level in v1

## Concrete CLI Workflow

The v1 CLI path from raw files to a trained fly-level baseline is:

1. Build or validate the manifest

```bash
python -m flygen_ml.cli.build_manifest \
  --input manifest_seed.csv \
  --output manifest.csv
```

Expected behavior:
- validate required columns
- validate that each row points to one `.data` and one `.trx`
- fail early on duplicate `sample_key`

2. Extract canonical between-reward segments

```bash
python -m flygen_ml.cli.extract_segments \
  --config configs/dataset/v1_binary.yaml \
  --manifest manifest.csv \
  --output artifacts/segments.csv
```

Expected behavior:
- load raw pickle pairs
- parse protocol
- identify experimental fly
- recover selected training boundaries
- extract calculated reward events for that fly
- emit canonical v1 segment records with QC flags

3. Build engineered features

```bash
python -m flygen_ml.cli.build_features \
  --feature-set engineered_v1 \
  --segments artifacts/segments.csv \
  --output artifacts/features.csv
```

Expected behavior:
- compute per-segment engineered features
- aggregate to fly-level rows
- retain grouping keys for leakage-safe evaluation

4. Train fly-level baseline

```bash
python -m flygen_ml.cli.train_model \
  --config configs/model/logreg.yaml \
  --features artifacts/features.csv \
  --output runs/logreg_v1
```

Expected behavior:
- load fly-level feature rows
- split by group without leakage
- fit a v1 baseline classifier
- write run metadata, metrics summary, and placeholder model artifact

5. Evaluate or summarize run outputs

```bash
python -m flygen_ml.cli.evaluate_model \
  --run-dir runs/logreg_v1
```

## Testing Plan

### Pickle loading

Test goals:
- paired `.data` and `.trx` files load without runtime dependency on upstream code
- `latin1`-compatible loading works
- expected top-level raw structures are returned

Fixture needs:
- at least one trusted `.data`/`.trx` pair from the selected chamber
- ideally one pair exercising legacy object-reference handling

### Protocol parsing

Test goals:
- `protocol["ct"]` is read and validated
- selected training boundaries are recovered
- required protocol fields are present or fail clearly

### Experimental-fly identification

Test goals:
- `protocol["frameNums"]` identifies the experimental fly as expected on trusted fixtures
- list and non-list forms are handled as intended for v1

### Reward-event extraction

Test goals:
- canonical calculated reward-entry stream is produced for the experimental fly
- actual reward metadata and calculated rewards can both be represented for comparison
- mismatches are surfaced clearly

### Between-reward segment extraction

Test goals:
- segment start frame is the first frame after reward-circle exit
- segment stop frame is the next calculated reward entry or training end
- QC flags appear for edge cases rather than silently dropping ambiguous cases

### Split leakage prevention

Test goals:
- segments from one fly never appear in both train and validation
- grouping assertions fail loudly when leakage would occur
- grouped split output is stable for a fixed seed

## V1 Exclusions

The following are explicitly out of scope for the first implementation:

- multiple chamber types
- control-fly modeling
- multi-training modeling
- joint multi-genotype tasks
- broad support for all upstream reward-analysis modes
- attention-pooling and segment-encoder implementation
- production-grade artifact serialization
- plotting-heavy experiment reporting

## Fixture-Validation Plan

Before modeling starts, the following properties must match trusted upstream expectations on a small fixture set:

- `protocol["ct"]` resolves to the intended chamber type
- `protocol["frameNums"]` yields the correct experimental fly
- selected training start and end frames match trusted expectations
- reward-circle metadata for the selected training matches trusted expectations
- calculated reward-entry count for the experimental fly matches trusted expectations
- canonical segment count matches trusted expectations
- a small set of manually inspected segment `start_frame` and `stop_frame` values match trusted expectations
- QC flags behave as expected on at least one edge-case fixture

If these checks do not pass, modeling should not proceed.
