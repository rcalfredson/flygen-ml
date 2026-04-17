# flygen-ml

Standalone machine-learning tooling for classifying fly genotype from between-reward trajectories.

This repository is intentionally separate from the upstream analysis codebase. It aims to stay semantically compatible with the upstream paired `.data` and `.trx` workflow while keeping loading, parsing, segmentation, feature building, and modeling logic self-contained.

## Status

This repo is currently scaffolded for a narrow v1:
- one chamber type
- one selected training
- experimental fly only
- one genotype comparison at a time
- grouped fly-level evaluation

The true loader, reward extraction, and between-reward segment semantics still require fixture-backed verification before implementation is considered complete.

## V1 Outputs

The current v1 path produces two CSV artifacts:

- `segments.csv`: one row per canonical between-reward segment, including source provenance, frame boundaries, minimal QC, and reward geometry. This table stays path-aware because it retains the information needed to recover the original trajectory slice from the paired raw files.
- `fly_features.csv`: one row per experimental fly after aggregating engineered per-segment summaries across that fly's segments. This table is intended for the first fly-level baseline and does not embed raw paths directly.

In other words, v1 keeps raw trajectory recoverability in the segment table, then trains the first baseline from aggregated engineered summaries at the fly level.

## Key Docs

- [docs/source-data-contract.md](docs/source-data-contract.md)
- [docs/upstream-notes.md](docs/upstream-notes.md)
- [docs/implementation-spec.md](docs/implementation-spec.md)
- [docs/experiment-plan.md](docs/experiment-plan.md)

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## CLI Skeletons

```bash
python -m flygen_ml.cli.build_manifest --help
python -m flygen_ml.cli.extract_segments --help
python -m flygen_ml.cli.build_features --help
python -m flygen_ml.cli.train_model --help
python -m flygen_ml.cli.evaluate_model --help
```
