from __future__ import annotations

import csv

from flygen_ml.cli import filter_occlusion_segments


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_filter_occlusion_segments_splits_single_deficiency_by_delta_sign_and_caps_fly(
    tmp_path,
    monkeypatch,
):
    occlusion_path = tmp_path / "segment_occlusion.csv"
    output_dir = tmp_path / "filtered"
    occlusion_path.write_text(
        "\n".join(
            [
                "segment_id,fly_id,actual_genotype,actual_cohort,occlusion_status,predicted_genotype_logit_delta,actual_genotype_logit_delta,predicted_cohort_logit_delta,actual_cohort_logit_delta",
                "seg0,fly0,PFN>Kir,antennae-intact,ok,0.5,0.5,0.1,0.1",
                "seg1,fly0,PFN>Kir,antennae-intact,ok,0.4,0.4,0.1,0.1",
                "seg2,fly0,PFN>Kir,antennae-intact,ok,0.3,0.3,0.1,0.1",
                "seg3,fly1,PFN>Kir,antennae-intact,ok,-0.7,-0.7,0.1,0.1",
                "seg4,fly1,PFN>Kir,antennae-intact,ok,-0.2,-0.2,0.1,0.1",
                "seg5,fly2,Control>Kir,antennae-removed,ok,0.9,0.9,0.1,0.1",
                "seg6,fly3,PFN>Kir,antennae-intact,skipped_single_segment,1.0,1.0,0.1,0.1",
                "seg7,fly4,PFN>Kir,antennae-removed,ok,1.0,1.0,0.1,0.1",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_occlusion_segments",
            "--occlusion-csv",
            str(occlusion_path),
            "--output-dir",
            str(output_dir),
            "--deficiency",
            "pfn-intact",
            "--head",
            "genotype",
            "--max-segments-per-fly",
            "2",
        ],
    )

    assert filter_occlusion_segments.main() == 0

    positive = _read_csv(output_dir / "pfn-intact_genotype_predicted_positive_logit_delta.csv")
    negative = _read_csv(output_dir / "pfn-intact_genotype_predicted_negative_logit_delta.csv")
    assert [row["segment_id"] for row in positive] == ["seg0", "seg1"]
    assert [row["filter_rank"] for row in positive] == ["1", "2"]
    assert [row["filter_abs_logit_delta"] for row in positive] == ["0.5", "0.4"]
    assert [row["segment_id"] for row in negative] == ["seg3", "seg4"]
    assert {row["actual_cohort"] for row in positive + negative} == {"antennae-intact"}


def test_filter_occlusion_segments_can_require_correct_genotype_predictions(
    tmp_path,
    monkeypatch,
):
    occlusion_path = tmp_path / "segment_occlusion.csv"
    output_dir = tmp_path / "filtered"
    occlusion_path.write_text(
        "\n".join(
            [
                "segment_id,fly_id,actual_genotype,predicted_genotype,actual_cohort,predicted_cohort,occlusion_status,predicted_genotype_logit_delta,actual_genotype_logit_delta,predicted_cohort_logit_delta,actual_cohort_logit_delta",
                "seg0,fly0,PFN>Kir,PFN>Kir,antennae-intact,antennae-removed,ok,0.5,0.5,0.1,0.1",
                "seg1,fly1,PFN>Kir,Control>Kir,antennae-intact,antennae-intact,ok,0.6,0.6,0.1,0.1",
                "seg2,fly2,PFN>Kir,PFN>Kir,antennae-intact,antennae-intact,ok,-0.4,-0.4,0.1,0.1",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_occlusion_segments",
            "--occlusion-csv",
            str(occlusion_path),
            "--output-dir",
            str(output_dir),
            "--deficiency",
            "pfn-intact",
            "--head",
            "genotype",
            "--require-correct",
            "genotype",
        ],
    )

    assert filter_occlusion_segments.main() == 0

    positive = _read_csv(output_dir / "pfn-intact_genotype_predicted_positive_logit_delta.csv")
    negative = _read_csv(output_dir / "pfn-intact_genotype_predicted_negative_logit_delta.csv")
    assert [row["segment_id"] for row in positive] == ["seg0"]
    assert [row["segment_id"] for row in negative] == ["seg2"]


def test_filter_occlusion_segments_can_require_correct_cohort_predictions(
    tmp_path,
    monkeypatch,
):
    occlusion_path = tmp_path / "segment_occlusion.csv"
    output_dir = tmp_path / "filtered"
    occlusion_path.write_text(
        "\n".join(
            [
                "segment_id,fly_id,actual_genotype,predicted_genotype,actual_cohort,predicted_cohort,occlusion_status,predicted_genotype_logit_delta,actual_genotype_logit_delta,predicted_cohort_logit_delta,actual_cohort_logit_delta",
                "seg0,fly0,Control>Kir,PFN>Kir,antennae-removed,antennae-removed,ok,0.1,0.1,0.5,0.5",
                "seg1,fly1,Control>Kir,Control>Kir,antennae-removed,antennae-intact,ok,0.1,0.1,0.6,0.6",
                "seg2,fly2,Control>Kir,Control>Kir,antennae-removed,antennae-removed,ok,0.1,0.1,-0.4,-0.4",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_occlusion_segments",
            "--occlusion-csv",
            str(occlusion_path),
            "--output-dir",
            str(output_dir),
            "--deficiency",
            "control-removed",
            "--head",
            "cohort",
            "--require-correct",
            "cohort",
        ],
    )

    assert filter_occlusion_segments.main() == 0

    positive = _read_csv(output_dir / "control-removed_cohort_predicted_positive_logit_delta.csv")
    negative = _read_csv(output_dir / "control-removed_cohort_predicted_negative_logit_delta.csv")
    assert [row["segment_id"] for row in positive] == ["seg0"]
    assert [row["segment_id"] for row in negative] == ["seg2"]
