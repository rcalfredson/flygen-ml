from __future__ import annotations

import argparse
from pathlib import Path

from flygen_ml.modeling.train import train_and_save_cross_validation_run, train_and_save_run


def _append_config_override(config_path: str, output_dir: str, *, label_key: str | None) -> str:
    if label_key is None:
        return config_path
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    override_path = out_dir / "resolved_train_config.yaml"
    config_text = Path(config_path).read_text()
    suffix = "" if config_text.endswith("\n") else "\n"
    override_path.write_text(f"{config_text}{suffix}label_key: {label_key}\n")
    return str(override_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fly-level baseline model.")
    parser.add_argument("--config", required=True, help="Model config path.")
    parser.add_argument("--features", required=True, help="Input feature table path.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    parser.add_argument(
        "--label-key",
        "--target-key",
        dest="label_key",
        default=None,
        help="Column to predict. Defaults to config label_key/target_key, then genotype.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Run grouped, label-aware K-fold cross-validation instead of a single holdout split.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = _append_config_override(args.config, args.output, label_key=args.label_key)
    if args.cv_folds is not None:
        run_metadata = train_and_save_cross_validation_run(
            config_path=config_path,
            features_path=args.features,
            output_dir=args.output,
            n_splits=args.cv_folds,
        )
        print(
            "trained "
            f"{run_metadata['model_kind']} to predict {run_metadata['label_key']} "
            f"with {run_metadata['n_folds']} grouped CV folds at {args.output}"
        )
        return 0

    run_metadata = train_and_save_run(
        config_path=config_path,
        features_path=args.features,
        output_dir=args.output,
    )
    print(
        "trained "
        f"{run_metadata['model_kind']} to predict {run_metadata['label_key']} "
        f"with {run_metadata['train_rows']} train rows and "
        f"{run_metadata['valid_rows']} validation rows at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
