from __future__ import annotations

import json
from pathlib import Path


def write_run_metadata(output_dir: str | Path, payload: dict[str, object]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
