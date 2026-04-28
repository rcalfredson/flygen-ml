from __future__ import annotations

import argparse
import sys

from flygen_ml.data_manifest import load_manifest
from flygen_ml.errors import MalformedRecordingError
from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.trajectory_builder import build_normalized_recording
from flygen_ml.segment_table import write_segment_table
from flygen_ml.segments.between_reward import extract_between_reward_segments
from flygen_ml.segments.reward_events import extract_reward_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract canonical v1 between-reward segments.")
    parser.add_argument("--config", required=True, help="Dataset config path.")
    parser.add_argument("--manifest", required=True, help="Validated manifest CSV.")
    parser.add_argument("--output", required=True, help="Output segment table path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_rows = load_manifest(args.manifest)
    segments = []
    skipped_sample_keys: list[str] = []
    for manifest_row in manifest_rows:
        try:
            raw_data, raw_trx = load_recording_pair(manifest_row.data_path, manifest_row.trx_path)
            recording = build_normalized_recording(manifest_row, raw_data, raw_trx)
            reward_events = extract_reward_events(recording)
            segments.extend(extract_between_reward_segments(recording, reward_events))
        except MalformedRecordingError as exc:
            skipped_sample_keys.append(manifest_row.sample_key)
            print(f"skipping malformed recording {manifest_row.sample_key}: {exc}", file=sys.stderr)
    write_segment_table(args.output, segments)
    print(
        "wrote "
        f"{len(segments)} canonical segments from {len(manifest_rows) - len(skipped_sample_keys)} "
        f"recordings to {args.output} "
        f"(dataset config path retained for v1 compatibility: {args.config})"
    )
    if skipped_sample_keys:
        print(
            f"skipped {len(skipped_sample_keys)} malformed recordings during extraction",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
