from __future__ import annotations

import pytest

from flygen_ml.modeling.metrics import summarize_metrics


def test_summarize_metrics_reports_accuracy_and_balanced_accuracy():
    metrics = summarize_metrics(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "B"],
        labels=["A", "B"],
    )

    assert metrics["n_examples"] == 4
    assert metrics["n_correct"] == 3
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["label_support"] == {"A": 2, "B": 2}
    assert metrics["label_recall"] == {"A": 0.5, "B": 1.0}
