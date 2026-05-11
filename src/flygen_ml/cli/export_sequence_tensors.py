from __future__ import annotations

import argparse

from flygen_ml.features.sequence import SEQUENCE_CHANNEL_SETS, write_sequence_npz
from flygen_ml.segment_table import load_segment_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export fixed-length trajectory tensors from segment tables.")
    parser.add_argument("--segments", required=True, help="Input segment table path.")
    parser.add_argument("--output", required=True, help="Output .npz tensor artifact path.")
    parser.add_argument("--target-length", type=int, default=128, help="Resampled frames per segment.")
    parser.add_argument(
        "--channel-set",
        choices=sorted(SEQUENCE_CHANNEL_SETS),
        default="default",
        help="Named sequence channel set to export.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated sequence channels. Overrides --channel-set when provided.",
    )
    parser.add_argument(
        "--include-training-end-segments",
        action="store_true",
        help="Include segments that ended only because the selected training ended.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.channels:
        channels = tuple(channel.strip() for channel in args.channels.split(",") if channel.strip())
    else:
        channels = SEQUENCE_CHANNEL_SETS[args.channel_set]
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
        f"with length {args.target_length}, channel_set={args.channel_set}, "
        f"and channels {','.join(channels)} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
