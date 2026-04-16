from __future__ import annotations

from flygen_ml.loaders.pickle_loader import load_recording_pair

from tests.fixture_registry import require_available_fixture


def test_load_recording_pair_fixture():
    fixture = require_available_fixture()
    raw_data, raw_trx = load_recording_pair(fixture.data_path, fixture.trx_path)

    assert isinstance(raw_data, dict)
    assert isinstance(raw_trx, dict)
    assert "protocol" in raw_data
    assert "x" in raw_trx
    assert "y" in raw_trx
    assert "ts" in raw_trx
