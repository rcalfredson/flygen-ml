from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

from flygen_ml.modeling.train import load_feature_rows


METADATA_COLUMNS = {
    "fly_id",
    "sample_key",
    "genotype",
    "cohort",
    "chamber",
    "chamber_type",
    "training_idx",
    "date",
    "fly_idx",
}


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _feature_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["fly_id"]), str(row["sample_key"])


def _feature_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        lookup[_feature_key(row)] = row
    return lookup


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def _numeric_columns(rows: list[dict[str, object]]) -> list[str]:
    first = rows[0]
    names: list[str] = []
    for key in first:
        if key in METADATA_COLUMNS:
            continue
        if any(_numeric_value(row.get(key)) is not None for row in rows):
            names.append(key)
    return names


def _categorical_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if value == "":
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _numeric_summary(rows: list[dict[str, object]], key: str) -> dict[str, float | int] | None:
    values = [
        value
        for value in (_numeric_value(row.get(key)) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _enrich_rows(
    comparison_rows: list[dict[str, str]],
    feature_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    features_by_key = _feature_lookup(feature_rows)
    enriched: list[dict[str, object]] = []
    for comparison in comparison_rows:
        key = str(comparison["fly_id"]), str(comparison.get("sample_key", ""))
        feature_row = features_by_key.get(key)
        if feature_row is None:
            raise ValueError(f"missing feature row for comparison row: fly_id={key[0]!r}, sample_key={key[1]!r}")
        enriched.append({**feature_row, **comparison})
    return enriched


def _rows_by_case(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["correctness_case"]), []).append(row)
    return grouped


def _case_summary(
    rows: list[dict[str, object]],
    *,
    numeric_columns: list[str],
) -> dict[str, object]:
    numeric_summaries = {
        key: summary
        for key in numeric_columns
        if (summary := _numeric_summary(rows, key)) is not None
    }
    return {
        "n": len(rows),
        "actual_label": _categorical_counts(rows, "actual_label"),
        "genotype": _categorical_counts(rows, "genotype"),
        "cohort": _categorical_counts(rows, "cohort"),
        "chamber_type": _categorical_counts(rows, "chamber_type"),
        "training_idx": _categorical_counts(rows, "training_idx"),
        "evidence_bin": _categorical_counts(rows, "evidence_bin"),
        "numeric": numeric_summaries,
    }


def _global_numeric_stats(
    rows: list[dict[str, object]],
    numeric_columns: list[str],
) -> dict[str, dict[str, float | int]]:
    return {
        key: summary
        for key in numeric_columns
        if (summary := _numeric_summary(rows, key)) is not None
    }


def _feature_shifts(
    *,
    case_summary: dict[str, object],
    global_summary: dict[str, dict[str, float | int]],
    top_n: int,
) -> list[dict[str, object]]:
    shifts: list[dict[str, object]] = []
    case_numeric = dict(case_summary["numeric"])
    for key, raw_case_stats in case_numeric.items():
        case_stats = dict(raw_case_stats)
        global_stats = global_summary.get(key)
        if global_stats is None:
            continue
        global_std = float(global_stats["std"])
        if global_std <= 0:
            continue
        case_mean = float(case_stats["mean"])
        global_mean = float(global_stats["mean"])
        z_delta = (case_mean - global_mean) / global_std
        shifts.append(
            {
                "feature": key,
                "case_mean": case_mean,
                "global_mean": global_mean,
                "global_std": global_std,
                "z_delta": z_delta,
                "abs_z_delta": abs(z_delta),
            }
        )
    shifts.sort(key=lambda row: float(row["abs_z_delta"]), reverse=True)
    return shifts[:top_n]


def build_error_bucket_report(
    *,
    comparison_rows: list[dict[str, str]],
    feature_rows: list[dict[str, object]],
    top_n_features: int = 8,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    enriched_rows = _enrich_rows(comparison_rows, feature_rows)
    numeric_columns = _numeric_columns(feature_rows)
    global_summary = _global_numeric_stats(enriched_rows, numeric_columns)
    grouped = _rows_by_case(enriched_rows)
    cases: dict[str, object] = {}
    for case_name, case_rows in sorted(grouped.items()):
        summary = _case_summary(case_rows, numeric_columns=numeric_columns)
        summary["top_feature_shifts"] = _feature_shifts(
            case_summary=summary,
            global_summary=global_summary,
            top_n=top_n_features,
        )
        cases[case_name] = summary
    report = {
        "n_examples": len(enriched_rows),
        "cases": cases,
        "global_numeric": global_summary,
    }
    return report, enriched_rows


def _write_enriched_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(report: dict[str, object]) -> None:
    print(f"n_examples: {int(report['n_examples'])}")
    cases = dict(report["cases"])
    for case_name, raw_case in sorted(cases.items()):
        case = dict(raw_case)
        print()
        print(f"{case_name}: n={int(case['n'])}")
        for key in ("actual_label", "genotype", "cohort", "evidence_bin"):
            counts = dict(case.get(key, {}))
            if counts:
                formatted = ", ".join(f"{value}={count}" for value, count in counts.items())
                print(f"  {key}: {formatted}")
        shifts = list(case.get("top_feature_shifts", []))
        if shifts:
            formatted_shifts = ", ".join(
                f"{dict(shift)['feature']}={float(dict(shift)['z_delta']):+.2f}sd"
                for shift in shifts[:5]
            )
            print(f"  top_feature_shifts: {formatted_shifts}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize prediction error buckets against fly metadata/features.")
    parser.add_argument("--comparison", required=True, help="Joined comparison CSV from compare_prediction_errors.")
    parser.add_argument("--features", required=True, help="Fly-level feature table.")
    parser.add_argument("--output", required=True, help="Output JSON summary path.")
    parser.add_argument("--examples-output", help="Optional enriched per-fly CSV output path.")
    parser.add_argument("--top-n-features", type=int, default=8, help="Number of shifted numeric features per bucket.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report, enriched_rows = build_error_bucket_report(
        comparison_rows=_load_csv_rows(args.comparison),
        feature_rows=load_feature_rows(args.features),
        top_n_features=args.top_n_features,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.examples_output:
        _write_enriched_rows(args.examples_output, enriched_rows)
    _print_summary(report)
    print()
    print(f"wrote_bucket_summary: {output_path}")
    if args.examples_output:
        print(f"wrote_enriched_examples: {args.examples_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
