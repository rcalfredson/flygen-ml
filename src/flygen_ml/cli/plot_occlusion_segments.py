from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _load_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"CSV table is empty: {path}")
    return rows, fieldnames


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float_value(value: object) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _score_row(row: dict[str, str], *, change_head: str) -> float:
    if change_head == "genotype":
        return abs(_float_value(row.get("predicted_genotype_logit_delta", "")))
    if change_head == "cohort":
        return abs(_float_value(row.get("predicted_cohort_logit_delta", "")))
    return max(
        abs(_float_value(row.get("predicted_genotype_logit_delta", ""))),
        abs(_float_value(row.get("predicted_cohort_logit_delta", ""))),
    )


def _row_changed(row: dict[str, str], *, change_head: str) -> bool:
    if change_head == "genotype":
        return _is_true(row.get("genotype_prediction_changed", ""))
    if change_head == "cohort":
        return _is_true(row.get("cohort_prediction_changed", ""))
    return _is_true(row.get("joint_prediction_changed", ""))


def _selected_rows(
    rows: list[dict[str, str]],
    *,
    change_head: str,
    changed_only: bool,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("occlusion_status", "ok") == "ok"
        and (not changed_only or _row_changed(row, change_head=change_head))
    ]
    selected.sort(key=lambda row: _score_row(row, change_head=change_head), reverse=True)
    if limit is not None:
        selected = selected[:limit]
    return selected


def _channel_index(channels: list[str], name: str) -> int:
    try:
        return channels.index(name)
    except ValueError as exc:
        raise ValueError(f"sequence artifact is missing required channel {name!r}") from exc


def _slug(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug[:max_length] or "segment"


def _scale_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    width: int,
    height: int,
    margin: int,
) -> list[tuple[float, float]]:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[finite]
    y = y_values[finite]
    if x.size == 0:
        return []
    extent = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.1)
    plot_size = min(width, height) - 2 * margin
    cx = width / 2
    cy = height / 2
    return [
        (cx + float(x_value) / extent * plot_size / 2, cy - float(y_value) / extent * plot_size / 2)
        for x_value, y_value in zip(x, y, strict=True)
    ]


def _write_path_svg(
    path: str | Path,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    subtitle: str,
    width: int = 420,
    height: int = 420,
) -> None:
    margin = 42
    points = _scale_points(x_values, y_values, width=width, height=height, margin=margin)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if points:
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    else:
        polyline = ""
    plot_size = min(width, height) - 2 * margin
    radius = plot_size / 2 / max(
        float(np.nanmax(np.abs(x_values))) if np.isfinite(x_values).any() else 1.1,
        float(np.nanmax(np.abs(y_values))) if np.isfinite(y_values).any() else 1.1,
        1.1,
    )
    start_marker = (
        f'<circle cx="{points[0][0]:.2f}" cy="{points[0][1]:.2f}" r="4" fill="#1f77b4" />'
        if points
        else ""
    )
    stop_marker = (
        f'<circle cx="{points[-1][0]:.2f}" cy="{points[-1][1]:.2f}" r="4" fill="#d62728" />'
        if points
        else ""
    )
    out_path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white" />',
                f'<text x="16" y="24" font-family="sans-serif" font-size="14" fill="#111">{escape(title)}</text>',
                f'<text x="16" y="44" font-family="sans-serif" font-size="11" fill="#555">{escape(subtitle)}</text>',
                f'<line x1="{margin}" y1="{height / 2:.2f}" x2="{width - margin}" y2="{height / 2:.2f}" stroke="#ddd" />',
                f'<line x1="{width / 2:.2f}" y1="{margin}" x2="{width / 2:.2f}" y2="{height - margin}" stroke="#ddd" />',
                f'<circle cx="{width / 2:.2f}" cy="{height / 2:.2f}" r="{radius:.2f}" fill="none" stroke="#999" stroke-dasharray="4 4" />',
                f'<polyline points="{polyline}" fill="none" stroke="#111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />',
                start_marker,
                stop_marker,
                "</svg>",
            ]
        )
    )


def _plot_title(row: dict[str, str]) -> str:
    return (
        f"{row.get('actual_genotype', '')}|{row.get('actual_cohort', '')} "
        f"base={row.get('predicted_genotype', '')}|{row.get('predicted_cohort', '')}"
    )


def _plot_subtitle(row: dict[str, str]) -> str:
    return (
        f"occluded={row.get('occluded_predicted_genotype', '')}|{row.get('occluded_predicted_cohort', '')} "
        f"genotype_dlogit={row.get('predicted_genotype_logit_delta', '')} "
        f"cohort_dlogit={row.get('predicted_cohort_logit_delta', '')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot sequence tensor paths for high-impact segment occlusion rows.",
    )
    parser.add_argument("--occlusion-csv", required=True, help="Segment occlusion CSV.")
    parser.add_argument("--sequence-path", required=True, help="Input .npz sequence tensor artifact.")
    parser.add_argument("--output-dir", required=True, help="Directory for SVG path plots and review CSV.")
    parser.add_argument(
        "--change-head",
        choices=("joint", "genotype", "cohort"),
        default="joint",
        help="Which prediction-change flag and logit delta to use for filtering/ranking.",
    )
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="Plot high-scoring rows even when the predicted class did not change.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to plot.")
    parser.add_argument(
        "--review-csv-name",
        default="occlusion_segment_plot_review.csv",
        help="Filename for the selected-row review table within --output-dir.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    occlusion_rows, occlusion_fieldnames = _load_csv_rows(args.occlusion_csv)
    selected = _selected_rows(
        occlusion_rows,
        change_head=args.change_head,
        changed_only=not args.include_unchanged,
        limit=args.limit,
    )
    payload = np.load(args.sequence_path)
    x = np.asarray(payload["x"], dtype=np.float32)
    channels = [str(channel) for channel in payload["channels"]]
    x_idx = _channel_index(channels, "x_rel")
    y_idx = _channel_index(channels, "y_rel")
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    review_rows: list[dict[str, str]] = []
    for rank, row in enumerate(selected, start=1):
        segment_index = int(row["segment_index"])
        segment_id = row.get("segment_id", f"segment_{segment_index}")
        filename = f"{rank:03d}_{_slug(segment_id)}.svg"
        plot_path = plot_dir / filename
        _write_path_svg(
            plot_path,
            x_values=x[segment_index, :, x_idx],
            y_values=x[segment_index, :, y_idx],
            title=_plot_title(row),
            subtitle=_plot_subtitle(row),
        )
        review_rows.append(
            {
                "plot_rank": str(rank),
                "plot_path": str(plot_path),
                "occlusion_score": f"{_score_row(row, change_head=args.change_head):.8g}",
                **row,
            }
        )
    review_path = output_dir / args.review_csv_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["plot_rank", "plot_path", "occlusion_score"] + occlusion_fieldnames
    with review_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)
    print(
        f"wrote {len(review_rows)} occlusion segment plots to {plot_dir} "
        f"and review rows to {review_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
