from __future__ import annotations

import argparse

from flygen_ml.modeling.train import train_and_save_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fly-level baseline model.")
    parser.add_argument("--config", required=True, help="Model config path.")
    parser.add_argument("--features", required=True, help="Input feature table path.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
