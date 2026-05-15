from __future__ import annotations

import csv

import numpy as np

from flygen_ml.cli import plot_occlusion_segments


def test_plot_occlusion_segments_cli_writes_review_csv_and_svgs(tmp_path, monkeypatch):
    sequence_path = tmp_path / "sequences.npz"
    occlusion_path = tmp_path / "segment_occlusion.csv"
    output_dir = tmp_path / "plots"
    x = np.zeros((3, 8, 2), dtype=np.float32)
    x[0, :, 0] = np.linspace(1.0, 2.0, num=8)
    x[0, :, 1] = np.linspace(0.0, 1.0, num=8)
    x[1, :, 0] = np.linspace(-1.0, 0.0, num=8)
    x[1, :, 1] = np.linspace(0.0, -1.0, num=8)
    x[2, :, 0] = np.linspace(0.5, 0.5, num=8)
    x[2, :, 1] = np.linspace(-0.5, 0.5, num=8)
    np.savez_compressed(
        sequence_path,
        x=x,
        mask=np.ones((3, 8), dtype=bool),
        segment_id=np.asarray(["seg0", "seg1", "seg2"]),
        sample_key=np.asarray(["s0", "s0", "s1"]),
        fly_id=np.asarray(["fly0", "fly0", "fly1"]),
        genotype=np.asarray(["A", "A", "B"]),
        cohort=np.asarray(["intact", "intact", "removed"]),
        qc_flags=np.asarray(["", "", ""]),
        channels=np.asarray(["x_rel", "y_rel"]),
        target_length=np.asarray(8),
    )
    occlusion_path.write_text(
        "\n".join(
            [
                "split,segment_id,fly_id,sample_key,segment_index,actual_genotype,predicted_genotype,occluded_predicted_genotype,genotype_prediction_changed,actual_cohort,predicted_cohort,occluded_predicted_cohort,cohort_prediction_changed,joint_prediction_changed,occlusion_status,predicted_genotype_logit_delta,predicted_cohort_logit_delta",
                "valid,seg0,fly0,s0,0,A,A,B,True,intact,intact,intact,False,True,ok,0.8,0.1",
                "valid,seg1,fly0,s0,1,A,A,A,False,intact,intact,removed,True,True,ok,0.2,1.5",
                "valid,seg2,fly1,s1,2,B,B,B,False,removed,removed,removed,False,False,ok,0.1,0.1",
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_occlusion_segments",
            "--occlusion-csv",
            str(occlusion_path),
            "--sequence-path",
            str(sequence_path),
            "--output-dir",
            str(output_dir),
            "--change-head",
            "joint",
        ],
    )

    assert plot_occlusion_segments.main() == 0

    review_path = output_dir / "occlusion_segment_plot_review.csv"
    rows = list(csv.DictReader(review_path.open()))
    assert [row["segment_id"] for row in rows] == ["seg1", "seg0"]
    assert [row["plot_rank"] for row in rows] == ["1", "2"]
    assert (output_dir / "plots" / "001_seg1.svg").exists()
    assert (output_dir / "plots" / "002_seg0.svg").exists()
    assert "<polyline" in (output_dir / "plots" / "001_seg1.svg").read_text()
