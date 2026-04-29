from __future__ import annotations


def evidence_bin_for_n_segments(n_segments: object) -> str:
    try:
        value = int(float(n_segments))
    except (TypeError, ValueError):
        return "unknown"
    if value < 20:
        return "low_n_segments_lt20"
    if value < 50:
        return "moderate_n_segments_20_to_49"
    return "high_n_segments_ge50"


def summarize_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: list[str] | None = None,
) -> dict[str, float | int | dict[str, float | int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot summarize empty predictions")

    resolved_labels = labels or sorted(set(y_true) | set(y_pred))
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    per_label_recalls: dict[str, float] = {}
    per_label_support: dict[str, int] = {}
    for label in resolved_labels:
        support = sum(int(value == label) for value in y_true)
        per_label_support[label] = support
        if support == 0:
            continue
        true_positive = sum(int(actual == label and predicted == label) for actual, predicted in zip(y_true, y_pred))
        per_label_recalls[label] = true_positive / support
    balanced_accuracy = (
        sum(per_label_recalls.values()) / len(per_label_recalls) if per_label_recalls else 0.0
    )
    return {
        "n_examples": len(y_true),
        "n_correct": correct,
        "accuracy": correct / len(y_true),
        "balanced_accuracy": balanced_accuracy,
        "label_support": per_label_support,
        "label_recall": per_label_recalls,
    }


def _actual_label(row: dict[str, object]) -> str:
    if "actual_label" in row:
        return str(row["actual_label"])
    return str(row["actual_genotype"])


def _predicted_label(row: dict[str, object]) -> str:
    if "predicted_label" in row:
        return str(row["predicted_label"])
    return str(row["predicted_genotype"])


def summarize_metrics_by_evidence_bin(
    prediction_rows: list[dict[str, object]],
    *,
    labels: list[str],
) -> dict[str, dict[str, float | int | dict[str, float | int]]]:
    rows_by_bin: dict[str, list[dict[str, object]]] = {}
    for row in prediction_rows:
        evidence_bin = str(row.get("evidence_bin") or evidence_bin_for_n_segments(row.get("n_segments")))
        rows_by_bin.setdefault(evidence_bin, []).append(row)

    summaries: dict[str, dict[str, float | int | dict[str, float | int]]] = {}
    for evidence_bin, rows in sorted(rows_by_bin.items()):
        summaries[evidence_bin] = summarize_metrics(
            [_actual_label(row) for row in rows],
            [_predicted_label(row) for row in rows],
            labels=labels,
        )
    return summaries
