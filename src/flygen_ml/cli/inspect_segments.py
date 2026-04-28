from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flygen_ml.features.segment_inspection import (
    build_segment_metric_rows_from_table,
    write_segment_metric_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank extracted segments by an engineered feature metric.")
    parser.add_argument("--segments", required=True, help="Input segment table path.")
    parser.add_argument("--metric", required=True, help="Engineered segment metric to rank by.")
    parser.add_argument("--sample-key", help="Restrict output to a single sample_key.")
    parser.add_argument("--fly-id", help="Restrict output to a single fly_id.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of rows to emit.")
    parser.add_argument("--ascending", action="store_true", help="Rank lowest values first.")
    parser.add_argument("--output", help="Output CSV path. Defaults to stdout.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_segment_metric_rows_from_table(
        segments_path=args.segments,
        metric=args.metric,
        sample_key=args.sample_key,
        fly_id=args.fly_id,
        descending=not args.ascending,
        limit=args.limit,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            write_segment_metric_rows(rows, handle)
        print(f"wrote {len(rows)} segment inspection rows to {output_path}")
    else:
        write_segment_metric_rows(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
