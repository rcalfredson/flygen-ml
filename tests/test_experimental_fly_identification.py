from __future__ import annotations

from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.protocol_parser import get_experimental_fly_indices, get_protocol

from tests.fixture_registry import optional_expected_value, require_available_fixture


def test_experimental_fly_identification_fixture():
    fixture = require_available_fixture()
    raw_data, _ = load_recording_pair(fixture.data_path, fixture.trx_path)
    protocol = get_protocol(raw_data)
    experimental_fly_indices = get_experimental_fly_indices(protocol)

    assert isinstance(experimental_fly_indices, list)
    assert experimental_fly_indices
    assert all(isinstance(idx, int) for idx in experimental_fly_indices)
    assert all(idx >= 0 for idx in experimental_fly_indices)

    expected_indices = optional_expected_value(fixture, "experimental_fly_indices")
    if expected_indices is not None:
        assert experimental_fly_indices == expected_indices
