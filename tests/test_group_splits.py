from __future__ import annotations

import pytest

from flygen_ml.modeling.splits import assert_no_group_leakage, grouped_k_fold_splits, grouped_split


def test_assert_no_group_leakage_allows_disjoint_groups():
    assert_no_group_leakage(["a", "b"], ["c"])


def test_assert_no_group_leakage_raises_on_overlap():
    with pytest.raises(ValueError):
        assert_no_group_leakage(["a", "b"], ["b", "c"])


def test_grouped_split_is_deterministic_and_label_aware():
    rows = [
        {"fly_id": "a0", "genotype": "A"},
        {"fly_id": "a1", "genotype": "A"},
        {"fly_id": "a2", "genotype": "A"},
        {"fly_id": "b0", "genotype": "B"},
        {"fly_id": "b1", "genotype": "B"},
        {"fly_id": "b2", "genotype": "B"},
    ]

    train_rows, valid_rows = grouped_split(rows, group_key="fly_id", random_seed=7, valid_fraction=0.34)

    assert {row["fly_id"] for row in train_rows} == {"a0", "a1", "b0", "b1"}
    assert {row["fly_id"] for row in valid_rows} == {"a2", "b2"}


def test_grouped_k_fold_splits_are_label_aware_and_leak_free():
    rows = [
        {"fly_id": "a0", "genotype": "A"},
        {"fly_id": "a1", "genotype": "A"},
        {"fly_id": "a2", "genotype": "A"},
        {"fly_id": "b0", "genotype": "B"},
        {"fly_id": "b1", "genotype": "B"},
        {"fly_id": "b2", "genotype": "B"},
    ]

    folds = grouped_k_fold_splits(rows, group_key="fly_id", random_seed=7, n_splits=3)

    assert len(folds) == 3
    valid_groups_seen: set[str] = set()
    for train_rows, valid_rows in folds:
        train_groups = {str(row["fly_id"]) for row in train_rows}
        valid_groups = {str(row["fly_id"]) for row in valid_rows}
        assert train_groups.isdisjoint(valid_groups)
        assert {row["genotype"] for row in valid_rows} == {"A", "B"}
        valid_groups_seen.update(valid_groups)
    assert valid_groups_seen == {"a0", "a1", "a2", "b0", "b1", "b2"}


def test_grouped_k_fold_splits_requires_each_label_in_each_fold():
    rows = [
        {"fly_id": "a0", "genotype": "A"},
        {"fly_id": "a1", "genotype": "A"},
        {"fly_id": "b0", "genotype": "B"},
        {"fly_id": "b1", "genotype": "B"},
    ]

    with pytest.raises(ValueError, match="needs at least 3 groups"):
        grouped_k_fold_splits(rows, group_key="fly_id", n_splits=3)


def test_grouped_split_uses_non_genotype_label_key():
    rows = [
        {"fly_id": "a0", "genotype": "G0", "cohort": "intact"},
        {"fly_id": "a1", "genotype": "G1", "cohort": "intact"},
        {"fly_id": "a2", "genotype": "G0", "cohort": "intact"},
        {"fly_id": "b0", "genotype": "G0", "cohort": "removed"},
        {"fly_id": "b1", "genotype": "G1", "cohort": "removed"},
        {"fly_id": "b2", "genotype": "G0", "cohort": "removed"},
    ]

    _, valid_rows = grouped_split(
        rows,
        group_key="fly_id",
        label_key="cohort",
        random_seed=7,
        valid_fraction=0.34,
    )

    assert {row["cohort"] for row in valid_rows} == {"intact", "removed"}
