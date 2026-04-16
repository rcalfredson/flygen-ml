from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build feature tables from extracted segments.")
    parser.add_argument("--feature-set", required=True, help="Feature set name.")
    parser.add_argument("--segments", required=True, help="Input segment table path.")
    parser.add_argument("--output", required=True, help="Output feature table path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        "TODO: implement feature building for "
        f"{args.feature_set} from {args.segments}; planned output {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
