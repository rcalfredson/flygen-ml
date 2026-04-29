from __future__ import annotations

from pathlib import Path

from flygen_ml.cli import extract_segments
from flygen_ml.errors import MalformedRecordingError
from flygen_ml.schema import ManifestRow, SegmentRecord


def test_extract_segments_skips_malformed_recordings(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "segments.csv"
    manifest_rows = [
        ManifestRow(
            sample_key="good__fly0",
            data_path=Path("/tmp/good.data"),
            trx_path=Path("/tmp/good.trx"),
            genotype="A",
            chamber="large",
            training_idx=1,
            fly_idx=0,
        ),
        ManifestRow(
            sample_key="bad__fly0",
            data_path=Path("/tmp/bad.data"),
            trx_path=Path("/tmp/bad.trx"),
            genotype="B",
            chamber="large",
            training_idx=1,
            fly_idx=0,
        ),
    ]

    def fake_load_manifest(path: str):
        assert path == "manifest.csv"
        return manifest_rows

    def fake_load_recording_pair(data_path: Path, trx_path: Path):
        return {"protocol": {"ct": "large"}}, {"ts": [0.0, 1.0]}

    def fake_build_normalized_recording(manifest_row: ManifestRow, raw_data, raw_trx):
        if manifest_row.sample_key == "bad__fly0":
            raise MalformedRecordingError('selected frameNums entry is missing "startTrain" sequence')
        return object()

    segment = SegmentRecord(
        segment_id="good__fly0__tr1__seg0",
        sample_key="good__fly0",
        fly_id="good__fly0__fly0",
        genotype="A",
        cohort="intact",
        chamber_type="large",
        experimental_fly_idx=0,
        data_path=Path("/tmp/good.data"),
        trx_path=Path("/tmp/good.trx"),
        training_idx=1,
        training_start_frame=10,
        training_end_frame=20,
        anchor_reward_frame=12,
        start_frame=13,
        stop_frame=18,
        end_reward_frame=None,
        duration_frames=5,
        n_finite_frames=5,
        finite_frame_fraction=1.0,
    )

    monkeypatch.setattr(extract_segments, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(extract_segments, "load_recording_pair", fake_load_recording_pair)
    monkeypatch.setattr(extract_segments, "build_normalized_recording", fake_build_normalized_recording)
    monkeypatch.setattr(extract_segments, "extract_reward_events", lambda recording: object())
    monkeypatch.setattr(extract_segments, "extract_between_reward_segments", lambda recording, reward_events: [segment])

    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_segments",
            "--config",
            "config.yaml",
            "--manifest",
            "manifest.csv",
            "--output",
            str(output_path),
        ],
    )

    exit_code = extract_segments.main()

    assert exit_code == 0
    assert output_path.exists()

    captured = capsys.readouterr()
    assert "wrote 1 canonical segments from 1 recordings" in captured.out
    assert "skipping malformed recording bad__fly0" in captured.err
    assert "skipped 1 malformed recordings during extraction" in captured.err
