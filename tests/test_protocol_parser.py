from __future__ import annotations

from flygen_ml.loaders.pickle_loader import load_recording_pair
from flygen_ml.loaders.protocol_parser import get_chamber_type, get_protocol

from tests.fixture_registry import optional_expected_value, require_available_fixture


def test_protocol_parser_reads_protocol_and_chamber_type():
    fixture = require_available_fixture()
    raw_data, _ = load_recording_pair(fixture.data_path, fixture.trx_path)
    protocol = get_protocol(raw_data)
    chamber_type = get_chamber_type(protocol)

    assert isinstance(protocol, dict)
    assert isinstance(chamber_type, str)
    assert chamber_type

    expected_ct = optional_expected_value(fixture, "protocol_ct")
    if expected_ct is not None:
        assert chamber_type == expected_ct
