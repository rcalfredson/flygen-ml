from __future__ import annotations


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
