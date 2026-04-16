from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


REGISTRY_PATH = Path(__file__).parent / "fixtures" / "local_fixture_registry.json"


@dataclass(frozen=True)
class LocalFixture:
    fixture_id: str
    sample_key: str
    data_path: Path
    trx_path: Path
    notes: str
    expected: dict[str, Any]

    @property
    def is_available(self) -> bool:
        return self.data_path.exists() and self.trx_path.exists()


def load_local_fixtures() -> list[LocalFixture]:
    payload = json.loads(REGISTRY_PATH.read_text())
    fixtures: list[LocalFixture] = []
    for item in payload.get("fixtures", []):
        fixtures.append(
            LocalFixture(
                fixture_id=item["fixture_id"],
                sample_key=item["sample_key"],
                data_path=Path(item["data_path"]),
                trx_path=Path(item["trx_path"]),
                notes=item.get("notes", ""),
                expected=item.get("expected", {}),
            )
        )
    return fixtures


def require_available_fixture() -> LocalFixture:
    fixtures = load_local_fixtures()
    if not fixtures:
        pytest.skip("no local fixtures configured")
    for fixture in fixtures:
        if fixture.is_available:
            return fixture
    pytest.skip(f"configured local fixtures are unavailable; checked registry at {REGISTRY_PATH}")


def optional_expected_value(fixture: LocalFixture, key: str) -> Any | None:
    return fixture.expected.get(key)
