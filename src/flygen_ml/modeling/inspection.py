from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TextIO

from flygen_ml.modeling.metrics import evidence_bin_for_n_segments
from flygen_ml.modeling.train import load_feature_rows


def load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def load_prediction_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "predicted_probability": float(row["predicted_probability"]),
                }
            )
    if not rows:
        raise ValueError(f"prediction table is empty: {path}")
    return rows


def _feature_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["fly_id"]), str(row["sample_key"])


def _feature_lookup(feature_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in feature_rows:
        lookup[_feature_key(row)] = row
    return lookup


def _numeric_feature_value(row: dict[str, object], feature_name: str) -> float:
    value = row.get(feature_name)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _format_contributors(items: list[tuple[str, float]], *, top_n: int) -> str:
    top_items = items[:top_n]
    return "|".join(f"{name}:{contribution:.6g}" for name, contribution in top_items)


def _feature_contributions(
    feature_row: dict[str, object],
    *,
    model: dict[str, object],
) -> list[tuple[str, float]]:
    feature_names = [str(name) for name in model["feature_names"]]
    means = [float(value) for value in model["feature_means"]]
    stds = [float(value) for value in model["feature_stds"]]
    weights = [float(value) for value in model["weights"]]

    contributions: list[tuple[str, float]] = []
    for name, mean, std, weight in zip(feature_names, means, stds, weights):
        raw_value = _numeric_feature_value(feature_row, name)
        standardized = 0.0 if math.isnan(raw_value) else (raw_value - mean) / std
        contributions.append((name, standardized * weight))
    return contributions


def build_prediction_inspection_rows(
    *,
    predictions: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    model: dict[str, object],
    split: str = "valid",
    include_correct: bool = False,
    top_n: int = 5,
) -> list[dict[str, object]]:
    feature_rows_by_key = _feature_lookup(feature_rows)
    labels = [str(label) for label in model["labels"]]
    if len(labels) != 2:
        raise ValueError(f"inspection expects exactly 2 labels, got {labels}")

    report_rows: list[dict[str, object]] = []
    for prediction in predictions:
        if str(prediction["split"]) != split:
            continue
        correct = prediction["actual_genotype"] == prediction["predicted_genotype"]
        if correct and not include_correct:
            continue

        key = _feature_key(prediction)
        feature_row = feature_rows_by_key.get(key)
        if feature_row is None:
            raise ValueError(f"missing feature row for prediction: fly_id={key[0]!r}, sample_key={key[1]!r}")

        contributions = _feature_contributions(feature_row, model=model)
        predicted_label = str(prediction["predicted_genotype"])
        if predicted_label == labels[1]:
            toward_predicted = sorted(
                [(name, value) for name, value in contributions if value > 0.0],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            against_predicted = sorted(
                [(name, value) for name, value in contributions if value < 0.0],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        elif predicted_label == labels[0]:
            toward_predicted = sorted(
                [(name, value) for name, value in contributions if value < 0.0],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            against_predicted = sorted(
                [(name, value) for name, value in contributions if value > 0.0],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        else:
            raise ValueError(f"prediction label {predicted_label!r} not found in model labels {labels}")

        probability = float(prediction["predicted_probability"])
        report_row: dict[str, object] = {
            "split": prediction["split"],
            "fly_id": prediction["fly_id"],
            "sample_key": prediction["sample_key"],
            "actual_genotype": prediction["actual_genotype"],
            "predicted_genotype": prediction["predicted_genotype"],
            "predicted_probability": probability,
            "decision_margin": abs(probability - 0.5),
            "correct": correct,
            "n_segments": feature_row.get("n_segments", ""),
            "n_segments_with_qc_flags": feature_row.get("n_segments_with_qc_flags", ""),
            "evidence_bin": evidence_bin_for_n_segments(feature_row.get("n_segments")),
            "top_toward_predicted": _format_contributors(toward_predicted, top_n=top_n),
            "top_against_predicted": _format_contributors(against_predicted, top_n=top_n),
        }
        for feature_name in model["feature_names"]:
            report_row[str(feature_name)] = feature_row.get(str(feature_name), "")
        report_rows.append(report_row)

    return sorted(report_rows, key=lambda row: float(row["decision_margin"]))


def write_prediction_inspection_rows(rows: list[dict[str, object]], handle: TextIO) -> None:
    base_fieldnames = [
        "split",
        "fly_id",
        "sample_key",
        "actual_genotype",
        "predicted_genotype",
        "predicted_probability",
        "decision_margin",
        "correct",
        "n_segments",
        "n_segments_with_qc_flags",
        "evidence_bin",
        "top_toward_predicted",
        "top_against_predicted",
    ]
    feature_fieldnames = [name for name in rows[0].keys() if name not in base_fieldnames] if rows else []
    writer = csv.DictWriter(handle, fieldnames=base_fieldnames + feature_fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
