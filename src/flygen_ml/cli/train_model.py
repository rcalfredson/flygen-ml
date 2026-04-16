from __future__ import annotations

import argparse

from flygen_ml.modeling.train import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fly-level baseline model.")
    parser.add_argument("--config", required=True, help="Model config path.")
    parser.add_argument("--features", required=True, help="Input feature table path.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    write_run_metadata(
        args.output,
        {
            "status": "scaffold_only",
            "config": args.config,
            "features": args.features,
        },
    )
    print(f"initialized scaffold run directory at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
