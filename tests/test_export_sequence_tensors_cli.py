from __future__ import annotations

from flygen_ml.cli import export_sequence_tensors
from flygen_ml.features.sequence import RICH_SEQUENCE_CHANNELS


def test_export_sequence_tensors_cli_supports_rich_channel_set(monkeypatch, tmp_path):
    calls = []

    def fake_load_segment_table(path):
        calls.append(("load", path))
        return []

    def fake_write_sequence_npz(path, segments, *, target_length, channels):
        calls.append(
            (
                "write",
                path,
                segments,
                target_length,
                channels,
            )
        )

    monkeypatch.setattr(export_sequence_tensors, "load_segment_table", fake_load_segment_table)
    monkeypatch.setattr(export_sequence_tensors, "write_sequence_npz", fake_write_sequence_npz)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_sequence_tensors",
            "--segments",
            "segments.csv",
            "--output",
            str(tmp_path / "sequences_rich.npz"),
            "--target-length",
            "64",
            "--channel-set",
            "rich",
        ],
    )

    assert export_sequence_tensors.main() == 0

    assert calls == [
        ("load", "segments.csv"),
        ("write", str(tmp_path / "sequences_rich.npz"), [], 64, RICH_SEQUENCE_CHANNELS),
    ]


def test_export_sequence_tensors_cli_channels_override_channel_set(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(export_sequence_tensors, "load_segment_table", lambda path: [])
    monkeypatch.setattr(
        export_sequence_tensors,
        "write_sequence_npz",
        lambda path, segments, *, target_length, channels: calls.append((path, target_length, channels)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_sequence_tensors",
            "--segments",
            "segments.csv",
            "--output",
            str(tmp_path / "sequences_custom.npz"),
            "--channel-set",
            "rich",
            "--channels",
            "x_rel,y_rel",
        ],
    )

    assert export_sequence_tensors.main() == 0

    assert calls == [(str(tmp_path / "sequences_custom.npz"), 128, ("x_rel", "y_rel"))]
