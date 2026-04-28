from __future__ import annotations

import random


def assert_no_group_leakage(train_groups: list[str], valid_groups: list[str]) -> None:
    overlap = set(train_groups) & set(valid_groups)
    if overlap:
        raise ValueError(f"group leakage detected: {sorted(overlap)}")


def grouped_split(
    rows: list[dict[str, object]],
    *,
    group_key: str,
    label_key: str = "genotype",
    random_seed: int = 0,
    valid_fraction: float = 0.25,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        raise ValueError("cannot split empty rows")
    if not (0.0 < valid_fraction < 1.0):
        raise ValueError(f"valid_fraction must be between 0 and 1, got {valid_fraction}")

    grouped_rows: dict[str, list[dict[str, object]]] = {}
    label_by_group: dict[str, str] = {}
    for row in rows:
        group_value = str(row[group_key])
        label_value = str(row[label_key])
        grouped_rows.setdefault(group_value, []).append(row)
        existing = label_by_group.get(group_value)
        if existing is None:
            label_by_group[group_value] = label_value
        elif existing != label_value:
            raise ValueError(f"group {group_value!r} has inconsistent labels: {existing!r} vs {label_value!r}")

    groups_by_label: dict[str, list[str]] = {}
    for group_value, label_value in label_by_group.items():
        groups_by_label.setdefault(label_value, []).append(group_value)

    valid_groups: set[str] = set()
    rng = random.Random(random_seed)
    for label_value, group_values in sorted(groups_by_label.items()):
        if len(group_values) < 2:
            raise ValueError(
                f"label {label_value!r} needs at least 2 groups for holdout evaluation, got {len(group_values)}"
            )
        shuffled = sorted(group_values)
        rng.shuffle(shuffled)
        n_valid = max(1, int(round(len(shuffled) * valid_fraction)))
        n_valid = min(n_valid, len(shuffled) - 1)
        valid_groups.update(shuffled[:n_valid])

    train_groups = sorted(set(grouped_rows) - valid_groups)
    valid_groups_sorted = sorted(valid_groups)
    assert_no_group_leakage(train_groups, valid_groups_sorted)

    train_rows = [row for row in rows if str(row[group_key]) in train_groups]
    valid_rows = [row for row in rows if str(row[group_key]) in valid_groups]
    if not train_rows or not valid_rows:
        raise ValueError("split produced an empty train or validation partition")
    return train_rows, valid_rows
