# Upstream Notes

## Purpose

This document records which upstream semantics inform `flygen_ml` v1.

It is a traceability document, not a dependency plan. The ML repo should stay self-sufficient at runtime and use the upstream codebase only as a reference during validation and initial implementation.

## Independence Rule

Allowed uses of the upstream repo:
- semantic reference during implementation
- comparison against trusted files
- citation of semantic origin in tests and notes

Disallowed uses:
- runtime imports
- path hacks to reach upstream internals
- treating the upstream repo as a stable external API

## Core Upstream-Aligned Assumptions For V1

The following assumptions remain intentionally in place:

1. `protocol["ct"]` is the chamber-type encoding and should remain part of the source-data contract.
2. `protocol["frameNums"]` is central to determining experimental-fly structure, consistent with upstream loader logic.
3. The canonical ML reward stream for v1 is calculated reward entries for the experimental fly.
4. The canonical ML segment definition for v1 is:
   - anchor on calculated reward entry
   - start at first frame after reward-circle exit
   - stop at next calculated reward entry or training end

These are repo-level decisions for `flygen_ml` v1, not accidental implementation details.

## Semantic Areas To Mirror

### 1. Paired pickle loading

The upstream workflow establishes the paired `.data` and `.trx` convention, including practical compatibility needs such as:
- `latin1` pickle decoding
- tolerance for legacy serialized object references
- reading raw structures before normalization

`flygen_ml` should mirror this behavior locally.

### 2. Chamber typing

The example loader treats `protocol["ct"]` as the authoritative chamber-type signal. `flygen_ml` should preserve that role.

For v1 this means:
- read `protocol["ct"]` explicitly
- constrain the first dataset slice to one chamber type
- fail clearly when encountering unsupported chamber types

### 3. Experimental-fly structure

The example loader also indicates that `protocol["frameNums"]` is central to experimental-fly structure.

Current upstream-aligned expectation:
- if `protocol["frameNums"]` is a list, truthy entries identify experimental-fly indices
- otherwise default to fly index `0`

This should be validated on fixture files for the chosen v1 dataset slice rather than assumed globally.

### 4. Training-window extraction

The upstream codebase contains the training-boundary semantics that `flygen_ml` needs to mirror for one selected training.

The implementation should recover:
- training index
- training start frame
- training end frame
- reward-related metadata scoped to that training

V1 should keep this narrow rather than abstracting over every possible protocol layout.

### 5. Reward semantics

The upstream codebase likely contains multiple reward-related notions and multiple reward-aligned analysis patterns. `flygen_ml` should preserve that nuance in documentation even while choosing one canonical ML path.

#### Actual rewards

Actual rewards are the reward events or bookkeeping markers tied directly to protocol delivery or stored reward state.

#### Calculated rewards

Calculated rewards are reconstructed from trajectory and reward-circle semantics for a particular fly. In practice, these are fly-specific reward-entry events inferred from the behavioral trace and reward geometry.

#### Why calculated rewards are canonical in v1

For `flygen_ml` v1, calculated reward entries for the experimental fly are the canonical ML reward stream because:
- they are directly tied to the fly trajectory being modeled
- they provide consistent anchors for between-reward segment extraction
- they align naturally with excursion-style definitions based on reward-circle exit and re-entry

This does not imply that actual rewards are unimportant. It means that for the v1 ML task, calculated rewards are the canonical event stream used for segment anchoring and training examples.

Where possible, implementation and tests should still compare actual reward information and calculated reward information to detect semantic mismatches.

### 6. Between-reward segmentation

The upstream codebase may include several reward-aligned analysis patterns. `flygen_ml` should acknowledge that plurality while still defining one canonical ML rule for v1.

The canonical `flygen_ml` v1 segment definition is:
- anchor on a calculated reward entry for the experimental fly
- start at the first frame after reward-circle exit
- stop at the next calculated reward entry or training end

This is the canonical ML definition for this repo in v1 even if upstream also contains other excursion-like or reward-window analyses.

## What To Re-Implement Locally

`flygen_ml` should implement locally:
- paired pickle loading
- protocol parsing
- chamber-type validation
- experimental-fly selection from `protocol["frameNums"]`
- training-window extraction
- calculated reward extraction for the experimental fly
- canonical between-reward segment extraction
- grouped fly-level evaluation

## What To Validate Against Upstream

Before trusting the v1 pipeline, compare `flygen_ml` outputs against a small set of trusted recordings for:
- chamber type from `protocol["ct"]`
- experimental-fly index derived from `protocol["frameNums"]`
- selected training boundaries
- reward-circle metadata for the selected training
- calculated reward counts for the experimental fly
- segment counts under the canonical v1 rule
- several manually inspected segment start/stop frames

## Notes From The Example Loader

The attached example loader supports several planning assumptions:

- `protocol["ct"]` is actively used as chamber-type gating
- `protocol["frameNums"]` is used to identify experimental-fly structure
- `protocol["tm"]` carries transform metadata relevant to chamber geometry
- `protocol["info"]` is part of the protocol metadata surface
- `trx["x"]`, `trx["y"]`, and `trx["ts"]` are sufficient to recover core trajectory timing and geometry inputs
- basic trajectory QC should be computed before trusting derived segments or features

## Implementation Posture

The right relationship to upstream is:
- mirror semantics where the ML task depends on them
- keep the v1 scope narrow
- encode decisions in local code and tests
- avoid overgeneralizing before the first dataset slice is working

Once implementation begins, this file should be extended with exact upstream function names, commit references, and any intentional divergences.
