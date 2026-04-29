from __future__ import annotations

import pytest

from flygen_ml.modeling.metrics import evidence_bin_for_n_segments, summarize_metrics, summarize_metrics_by_evidence_bin


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


def test_evidence_bin_for_n_segments():
    assert evidence_bin_for_n_segments(2) == "low_n_segments_lt20"
    assert evidence_bin_for_n_segments(20) == "moderate_n_segments_20_to_49"
    assert evidence_bin_for_n_segments(50) == "high_n_segments_ge50"
    assert evidence_bin_for_n_segments("") == "unknown"


def test_summarize_metrics_by_evidence_bin():
    rows = [
        {"actual_label": "A", "predicted_label": "A", "n_segments": 2},
        {"actual_label": "B", "predicted_label": "A", "n_segments": 30},
        {"actual_label": "B", "predicted_label": "B", "n_segments": 60},
    ]

    summary = summarize_metrics_by_evidence_bin(rows, labels=["A", "B"])

    assert summary["low_n_segments_lt20"]["accuracy"] == pytest.approx(1.0)
    assert summary["moderate_n_segments_20_to_49"]["accuracy"] == pytest.approx(0.0)
    assert summary["high_n_segments_ge50"]["accuracy"] == pytest.approx(1.0)
