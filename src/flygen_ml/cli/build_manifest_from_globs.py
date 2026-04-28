from __future__ import annotations

import argparse

from flygen_ml.data_manifest import write_manifest
from flygen_ml.manifest_globs import build_manifest_rows_from_glob_specs, load_manifest_glob_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a manifest CSV by expanding glob-based recording specs.")
    parser.add_argument("--spec", required=True, help="CSV spec path with grouped glob patterns and labels.")
    parser.add_argument("--output", required=True, help="Output manifest CSV path.")
    parser.add_argument(
        "--repeat-fly-indices",
        help="Comma-separated fly_idx values to emit for every matched recording pair, e.g. 0,1.",
    )
    return parser


def _parse_repeat_fly_indices(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError("--repeat-fly-indices must contain at least one integer")
    return values


def main() -> int:
    args = build_parser().parse_args()
    specs = load_manifest_glob_specs(args.spec)
    rows = build_manifest_rows_from_glob_specs(
        specs,
        repeated_fly_indices=_parse_repeat_fly_indices(args.repeat_fly_indices),
    )
    write_manifest(args.output, rows)
    print(f"wrote {len(rows)} manifest rows from glob spec {args.spec} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
