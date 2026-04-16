from __future__ import annotations

import pytest

from flygen_ml.modeling.splits import assert_no_group_leakage


def test_assert_no_group_leakage_allows_disjoint_groups():
    assert_no_group_leakage(["a", "b"], ["c"])


def test_assert_no_group_leakage_raises_on_overlap():
    with pytest.raises(ValueError):
        assert_no_group_leakage(["a", "b"], ["b", "c"])
