from __future__ import annotations

import argparse

from flygen_ml.modeling.sequence_training import (
    train_and_save_sequence_cross_validation_run,
    train_and_save_sequence_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fly-level mean-pooled trajectory sequence model.")
    parser.add_argument("--config", required=True, help="Model config path.")
    parser.add_argument("--sequences", required=True, help="Input .npz sequence tensor artifact.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Run grouped K-fold cross-validation instead of a single holdout split.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cv_folds is not None:
        metadata = train_and_save_sequence_cross_validation_run(
            config_path=args.config,
            sequence_path=args.sequences,
            output_dir=args.output,
            n_splits=args.cv_folds,
        )
        print(
            f"trained {metadata['model_kind']} with {metadata['n_folds']} "
            f"grouped CV folds at {args.output}"
        )
        return 0

    metadata = train_and_save_sequence_run(
        config_path=args.config,
        sequence_path=args.sequences,
        output_dir=args.output,
    )
    print(
        f"trained {metadata['model_kind']} with {metadata['train_flies']} train flies "
        f"and {metadata['valid_flies']} validation flies at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
