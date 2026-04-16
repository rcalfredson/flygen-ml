from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved model run.")
    parser.add_argument("--run-dir", required=True, help="Run directory path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"TODO: implement evaluation for run directory {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
