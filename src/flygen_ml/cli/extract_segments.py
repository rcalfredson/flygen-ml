from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract canonical v1 between-reward segments.")
    parser.add_argument("--config", required=True, help="Dataset config path.")
    parser.add_argument("--manifest", required=True, help="Validated manifest CSV.")
    parser.add_argument("--output", required=True, help="Output segment table path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        "TODO: implement segment extraction from "
        f"{args.manifest} using {args.config}; planned output {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
