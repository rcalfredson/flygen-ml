from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved model run.")
    parser.add_argument("--run-dir", required=True, help="Run directory path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics summary: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
