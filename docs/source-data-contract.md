# Source Data Contract

## Purpose

This document defines the minimum source-data assumptions for `flygen_ml` v1.

The goal is semantic compatibility with the upstream paired `.data` and `.trx` pickle workflow while keeping this repository self-sufficient. `flygen_ml` should re-implement the required loading, protocol parsing, reward extraction, and segment construction logic locally rather than depending on the upstream repo at runtime.

## V1 Scope

This contract is intentionally narrow.

Included in v1:
- one chamber type
- one selected training
- experimental fly only
- one genotype comparison at a time
- grouped fly-level evaluation
- one canonical reward stream for ML
- one canonical between-reward segment definition

Excluded from v1:
- multiple chamber types
- control-fly modeling
- multi-training pooling
- support for all upstream analysis modes
- runtime imports from the upstream analysis repo

## Required Raw Files

Each recording sample is represented by one paired set of files:

- `*.data`
- `*.trx`

These files must refer to the same recording and must be loadable together.

## Loader Requirements

### Pickle decoding

The loader must support:
- Python pickle loading
- `latin1` decoding compatibility
- legacy object references that may appear in stored structures

The loader should return raw Python objects first. Normalization into `flygen_ml` schemas should happen in a separate parsing layer.

### File pairing

For every sample:
- exactly one `.data` file must pair with exactly one `.trx` file
- pairing is based on shared stem or manifest-provided paths
- missing pairs are a hard error

## Required Fields From `.data`

The `.data` pickle must contain a top-level `protocol` mapping with enough information to recover:

- chamber type
- experimental-fly structure
- training boundaries
- reward-related metadata
- geometry/transform metadata if needed for the supported chamber

Expected protocol components for v1:
- `protocol["ct"]`
- `protocol["frameNums"]`
- `protocol["info"]`
- `protocol["tm"]` when needed for chamber geometry

### Chamber type

`protocol["ct"]` is the chamber-type encoding and remains part of the source-data contract.

For v1:
- the parser should read `protocol["ct"]`
- the selected dataset should use exactly one supported chamber type
- files with any other chamber type should fail clearly unless and until support is added

This is not incidental metadata. It is part of the semantic contract because chamber type can affect geometry interpretation, reward-circle interpretation, and segment normalization.

### Experimental-fly structure

`protocol["frameNums"]` is central to determining experimental-fly structure and should remain part of the contract.

The attached example loader suggests the current upstream-style rule:
- if `protocol["frameNums"]` is a list, truthy entries identify experimental-fly indices
- otherwise default to fly index `0`

For v1, `flygen_ml` should preserve this assumption unless fixture validation shows a mismatch for the chosen dataset slice.

## Required Fields From `.trx`

The `.trx` pickle must provide enough trajectory information to recover:

- x position by fly and frame
- y position by fly and frame
- timestamps or equivalent timing base

Expected fields for v1:
- `trx["x"]`
- `trx["y"]`
- `trx["ts"]`

`trx["ts"]` may be used to infer FPS when FPS is not available elsewhere in a more explicit form.

## Experimental Fly Selection

V1 models the experimental fly only.

The parser must identify the experimental fly index from `protocol["frameNums"]` using the upstream-aligned rule for the selected chamber and dataset slice. This rule should be verified with fixtures and documented in tests.

Minimum normalized fields:
- `sample_key`
- `fly_idx`
- `is_experimental_fly`

## Training Window Contract

The parser must recover, for the selected experimental fly:
- training index
- training start frame
- training end frame
- reward metadata associated with that training

V1 uses one selected training only. The chosen training should be explicit in configuration and preserved in all downstream records.

Minimum normalized representation:
- `sample_key`
- `fly_idx`
- `training_idx`
- `training_start_frame`
- `training_end_frame`

## Reward Semantics

The upstream codebase may support more than one reward-aligned analysis pattern and may distinguish between different reward notions. For `flygen_ml`, this needs to be made explicit rather than implicit.

### Actual rewards vs calculated rewards

For planning and implementation, distinguish two concepts:

- actual rewards:
  events or reward-state markers tied directly to protocol delivery or stored reward bookkeeping

- calculated rewards:
  reward-entry events reconstructed from trajectory and reward-circle semantics for a specific fly

These are related but not interchangeable. Depending on the upstream analysis path, they may differ in timing, count, or intended use.

### Canonical ML reward stream for v1

For v1, the canonical ML reward stream is:
- calculated reward entries for the experimental fly

This choice is intentional. It provides a trajectory-grounded event stream that matches the proposed between-reward segment logic for this repo.

`flygen_ml` should still preserve enough metadata to compare calculated rewards against actual reward information when available, but model training and segment extraction should use the calculated reward stream as the canonical v1 source of anchors.

## Trajectory Preprocessing Contract

Before segmentation, trajectory handling should support:
- conversion of raw x/y arrays to float arrays
- preservation of missing tracking as NaNs
- optional interpolation for geometry-dependent calculations where needed
- basic QC metrics such as:
  - lost-frame fraction
  - lost-sequence count
  - max lost span
  - suspicious jump indicators

V1 should prefer explicit QC flags over silent filtering.

## Canonical V1 Segment Definition

`flygen_ml` may later support additional reward-aligned segment definitions, but v1 should define one canonical ML segment rule and use it consistently across extraction, features, and modeling.

The canonical v1 segment definition is:

1. Anchor on a calculated reward entry for the experimental fly.
2. Find the first frame after reward-circle exit following that reward.
3. Start the segment at that first post-exit frame.
4. Stop the segment at the next calculated reward entry for the same fly.
5. If there is no subsequent calculated reward entry before training end, stop at training end.

Interpretation:
- the reward-circle dwell immediately after the anchor reward is excluded
- the segment captures the excursion between one calculated reward visit and the next calculated reward entry, or training termination

This is the canonical ML segment definition for this repo in v1 even if the upstream codebase contains multiple reward-aligned analysis patterns.

## Segment Inclusion And QC Rules

A candidate segment may be retained with flags or excluded depending on config.

Recommended v1 QC fields:
- `too_short`
- `no_exit_after_reward`
- `no_next_reward_before_training_end`
- `insufficient_finite_frames`
- `bad_tracking_overlap`
- `training_boundary_truncated`

V1 preference:
- keep borderline segments with flags where possible
- exclude only clearly invalid segments
- keep exclusion logic simple and explicit

## Coordinate Normalization

V1 normalized coordinates should be reward-centered.

Preferred normalized representations:
- x relative to reward center
- y relative to reward center
- radial distance from reward center
- optional derived kinematics such as speed or heading change

If chamber transforms are required to interpret reward geometry, that logic should live inside `flygen_ml`.

## Manifest Contract

The manifest is the boundary between raw data discovery and modeling.

Minimum manifest columns for v1:
- `sample_key`
- `data_path`
- `trx_path`
- `genotype`
- `cohort`
- `date`
- `chamber`
- `training_idx`

If experimental fly index is not stored directly, the loader must derive it from the source contract, especially `protocol["frameNums"]`.

Labels must come from the manifest, not from raw pickle internals.

## Downstream Segment Table Contract

Each extracted segment should produce one row with at least:

- `segment_id`
- `sample_key`
- `fly_id`
- `genotype`
- `training_idx`
- `anchor_reward_frame`
- `start_frame`
- `stop_frame`
- `end_reward_frame` if present
- `duration_frames`
- `n_finite_frames`
- `qc_flags`

## Compatibility Philosophy

`flygen_ml` aims for semantic compatibility, not source-code dependency.

Compatibility means:
- the same paired raw files can be loaded
- `protocol["ct"]` and `protocol["frameNums"]` retain their current importance
- the supported training and reward metadata can be recovered
- the canonical v1 calculated-reward segment definition can be reproduced locally

It does not mean:
- matching every upstream internal type
- supporting every chamber or protocol variant
- importing upstream modules at runtime
