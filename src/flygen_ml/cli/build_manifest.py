from __future__ import annotations

import argparse

from flygen_ml.data_manifest import load_manifest, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and rewrite a manifest CSV.")
    parser.add_argument("--input", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", required=True, help="Output manifest CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_manifest(args.input)
    write_manifest(args.output, rows)
    print(f"wrote validated manifest to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
