from __future__ import annotations

import argparse

from flygen_ml.features.sequence import DEFAULT_SEQUENCE_CHANNELS, write_sequence_npz
from flygen_ml.segment_table import load_segment_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export fixed-length trajectory tensors from segment tables.")
    parser.add_argument("--segments", required=True, help="Input segment table path.")
    parser.add_argument("--output", required=True, help="Output .npz tensor artifact path.")
    parser.add_argument("--target-length", type=int, default=128, help="Resampled frames per segment.")
    parser.add_argument(
        "--channels",
        default=",".join(DEFAULT_SEQUENCE_CHANNELS),
        help="Comma-separated sequence channels.",
    )
    parser.add_argument(
        "--include-training-end-segments",
        action="store_true",
        help="Include segments that ended only because the selected training ended.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    channels = tuple(channel.strip() for channel in args.channels.split(",") if channel.strip())
    segments = load_segment_table(args.segments)
    if not args.include_training_end_segments:
        segments = [segment for segment in segments if not segment.terminated_by_training_end]
    write_sequence_npz(
        args.output,
        segments,
        target_length=args.target_length,
        channels=channels,
    )
    print(
        f"wrote {len(segments)} segment sequence tensors "
        f"with length {args.target_length} and channels {','.join(channels)} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
