from __future__ import annotations

import csv

from flygen_ml.cli import export_prediction_segments


def test_build_prediction_segment_rows_repeats_prediction_metadata_for_segments():
    prediction_review_rows = [
        {
            "fold": "0",
            "split": "valid",
            "fly_id": "fly0",
            "sample_key": "sample0",
            "label_key": "cohort",
            "actual_label": "intact",
            "predicted_label": "removed",
            "correct": "False",
            "predicted_probability": "0.8",
            "decision_margin": "0.3",
            "evidence_bin": "moderate_n_segments_20_to_49",
            "n_segments": "2",
            "n_segments_with_qc_flags": "1",
        }
    ]
    segment_rows = [
        {
            "segment_id": "seg0",
            "sample_key": "sample0",
            "fly_id": "fly0",
            "genotype": "G0",
            "cohort": "intact",
            "start_frame": "10",
            "stop_frame": "20",
        },
        {
            "segment_id": "seg1",
            "sample_key": "sample0",
            "fly_id": "fly0",
            "genotype": "G0",
            "cohort": "intact",
            "start_frame": "30",
            "stop_frame": "40",
        },
    ]

    rows = export_prediction_segments.build_prediction_segment_rows(
        prediction_review_rows=prediction_review_rows,
        segment_rows=segment_rows,
    )

    assert len(rows) == 2
    assert rows[0]["prediction_actual_label"] == "intact"
    assert rows[0]["prediction_predicted_label"] == "removed"
    assert rows[0]["segment_id"] == "seg0"
    assert rows[1]["prediction_decision_margin"] == "0.3"
    assert rows[1]["segment_id"] == "seg1"


def test_export_prediction_segments_cli_filters_high_confidence_errors(tmp_path, monkeypatch, capsys):
    review_path = tmp_path / "review.csv"
    segments_path = tmp_path / "segments.csv"
    output_path = tmp_path / "selected_segments.csv"

    review_path.write_text(
        "\n".join(
            [
                "fold,split,fly_id,sample_key,label_key,actual_label,predicted_label,correct,predicted_probability,decision_margin,evidence_bin,n_segments,n_segments_with_qc_flags",
                "0,valid,fly0,sample0,cohort,intact,removed,False,0.80,0.30,moderate_n_segments_20_to_49,2,1",
                "0,valid,fly1,sample1,cohort,removed,removed,True,0.90,0.40,high_n_segments_ge50,1,0",
                "0,valid,fly2,sample2,cohort,removed,intact,False,0.49,0.01,high_n_segments_ge50,1,0",
            ]
        )
    )
    segments_path.write_text(
        "\n".join(
            [
                "segment_id,sample_key,fly_id,genotype,cohort,start_frame,stop_frame",
                "seg0,sample0,fly0,G0,intact,10,20",
                "seg1,sample0,fly0,G0,intact,30,40",
                "seg2,sample1,fly1,G1,removed,50,60",
                "seg3,sample2,fly2,G1,removed,70,80",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_prediction_segments",
            "--prediction-review",
            str(review_path),
            "--segments",
            str(segments_path),
            "--output",
            str(output_path),
            "--errors-only",
            "--min-decision-margin",
            "0.30",
        ],
    )

    assert export_prediction_segments.main() == 0

    assert "wrote 2 segment rows from 1 prediction rows" in capsys.readouterr().out
    rows = list(csv.DictReader(output_path.open()))
    assert len(rows) == 2
    assert rows[0]["prediction_fold"] == "0"
    assert rows[0]["prediction_actual_label"] == "intact"
    assert rows[0]["segment_id"] == "seg0"
    assert rows[1]["segment_id"] == "seg1"
