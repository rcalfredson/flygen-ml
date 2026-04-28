from __future__ import annotations

import argparse

from flygen_ml.modeling.train import train_and_save_cross_validation_run, train_and_save_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fly-level baseline model.")
    parser.add_argument("--config", required=True, help="Model config path.")
    parser.add_argument("--features", required=True, help="Input feature table path.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Run grouped, label-aware K-fold cross-validation instead of a single holdout split.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cv_folds is not None:
        run_metadata = train_and_save_cross_validation_run(
            config_path=args.config,
            features_path=args.features,
            output_dir=args.output,
            n_splits=args.cv_folds,
        )
        print(
            "trained "
            f"{run_metadata['model_kind']} with {run_metadata['n_folds']} grouped CV folds at {args.output}"
        )
        return 0

    run_metadata = train_and_save_run(
        config_path=args.config,
        features_path=args.features,
        output_dir=args.output,
    )
    print(
        "trained "
        f"{run_metadata['model_kind']} with {run_metadata['train_rows']} train rows and "
        f"{run_metadata['valid_rows']} validation rows at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
