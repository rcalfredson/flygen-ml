from __future__ import annotations

import csv

import pytest

from flygen_ml.cli import export_occlusion_segments


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_occlusion_segments_joins_evidence_to_segment_metadata(tmp_path, monkeypatch):
    occlusion_path = tmp_path / "occlusion.csv"
    segments_path = tmp_path / "segments.csv"
    output_path = tmp_path / "plot_ready.csv"
    occlusion_path.write_text(
        "\n".join(
            [
                "filter_rank,filter_abs_logit_delta,segment_id,actual_genotype,predicted_genotype,actual_cohort,predicted_cohort,baseline_predicted_genotype_probability,occlusion_status,n_segments",
                "1,0.42,seg1,PFN>Kir,PFN>Kir,antennae-intact,antennae-intact,0.91,ok,12",
            ]
        )
    )
    segments_path.write_text(
        "\n".join(
            [
                "segment_id,sample_key,fly_id,genotype,cohort,experimental_fly_idx,data_path,trx_path,training_idx,anchor_reward_frame,end_reward_frame,duration_frames,terminated_by_training_end",
                "seg1,sample0,fly0,PFN>Kir,antennae-intact,0,/tmp/a.data,/tmp/a.trx,1,100,200,100,false",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_occlusion_segments",
            "--occlusion-csv",
            str(occlusion_path),
            "--segments",
            str(segments_path),
            "--output",
            str(output_path),
        ],
    )

    assert export_occlusion_segments.main() == 0

    rows = _read_csv(output_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["segment_id"] == "seg1"
    assert row["data_path"] == "/tmp/a.data"
    assert row["trx_path"] == "/tmp/a.trx"
    assert row["anchor_reward_frame"] == "100"
    assert row["end_reward_frame"] == "200"
    assert row["prediction_actual_label"] == "PFN>Kir|antennae-intact"
    assert row["prediction_predicted_label"] == "PFN>Kir|antennae-intact"
    assert row["prediction_correct"] == "True"
    assert row["prediction_probability"] == "0.91"
    assert row["prediction_decision_margin"] == "0.42"
    assert row["prediction_evidence_bin"] == "ok"


def test_export_occlusion_segments_reports_missing_segment_metadata():
    with pytest.raises(ValueError, match="missing segment metadata"):
        export_occlusion_segments.build_occlusion_segment_rows(
            occlusion_rows=[{"segment_id": "missing"}],
            segment_rows=[{"segment_id": "seg0"}],
        )
