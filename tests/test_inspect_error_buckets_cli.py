from __future__ import annotations

import csv
import json

from flygen_ml.cli import inspect_error_buckets


def test_inspect_error_buckets_summarizes_cases_and_feature_shifts(monkeypatch, tmp_path, capsys):
    comparison_path = tmp_path / "comparison.csv"
    features_path = tmp_path / "features.csv"
    output_path = tmp_path / "summary.json"
    examples_path = tmp_path / "examples.csv"
    comparison_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,split,fold,actual_label,run_a_predicted_label,run_a_predicted_probability,run_a_correct,run_b_predicted_label,run_b_predicted_probability,run_b_correct,correctness_case,n_segments,evidence_bin",
                "fly0,s0,valid,0,A,A,0.9,True,A,0.8,True,both_correct,10,low",
                "fly1,s1,valid,0,A,A,0.8,True,B,0.7,False,run_a_only_correct,20,moderate",
                "fly2,s2,valid,0,B,A,0.6,False,B,0.9,True,run_b_only_correct,30,high",
                "fly3,s3,valid,0,B,A,0.6,False,A,0.7,False,both_wrong,40,high",
            ]
        )
    )
    features_path.write_text(
        "\n".join(
            [
                "fly_id,sample_key,genotype,cohort,chamber_type,training_idx,n_segments,n_segments_with_qc_flags,path_length_px_mean,mean_radius_px_mean",
                "fly0,s0,A,intact,large,1,10,0,1.0,2.0",
                "fly1,s1,A,removed,large,1,20,1,10.0,2.5",
                "fly2,s2,B,intact,large,1,30,0,2.0,9.0",
                "fly3,s3,B,removed,large,2,40,2,3.0,10.0",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect_error_buckets",
            "--comparison",
            str(comparison_path),
            "--features",
            str(features_path),
            "--output",
            str(output_path),
            "--examples-output",
            str(examples_path),
            "--top-n-features",
            "2",
        ],
    )

    assert inspect_error_buckets.main() == 0

    out = capsys.readouterr().out
    assert "n_examples: 4" in out
    assert "run_a_only_correct: n=1" in out
    assert "top_feature_shifts:" in out
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["n_examples"] == 4
    assert payload["cases"]["run_a_only_correct"]["genotype"] == {"A": 1}
    assert payload["cases"]["run_b_only_correct"]["cohort"] == {"intact": 1}
    assert len(payload["cases"]["both_wrong"]["top_feature_shifts"]) == 2

    with examples_path.open("r", newline="") as handle:
        examples = list(csv.DictReader(handle))
    assert len(examples) == 4
    assert examples[0]["genotype"] == "A"
    assert examples[0]["correctness_case"] == "both_correct"


def test_build_error_bucket_report_rejects_missing_feature_rows():
    comparison_rows = [
        {
            "fly_id": "fly0",
            "sample_key": "s0",
            "correctness_case": "both_wrong",
            "actual_label": "A",
        }
    ]
    feature_rows = [{"fly_id": "fly1", "sample_key": "s1", "genotype": "A"}]

    try:
        inspect_error_buckets.build_error_bucket_report(
            comparison_rows=comparison_rows,
            feature_rows=feature_rows,
        )
    except ValueError as error:
        assert "missing feature row" in str(error)
    else:
        raise AssertionError("expected missing feature rows to fail")
