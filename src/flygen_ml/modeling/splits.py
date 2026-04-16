from __future__ import annotations


def assert_no_group_leakage(train_groups: list[str], valid_groups: list[str]) -> None:
    overlap = set(train_groups) & set(valid_groups)
    if overlap:
        raise ValueError(f"group leakage detected: {sorted(overlap)}")


def grouped_split(rows: list[dict[str, object]], *, group_key: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    del rows
    del group_key
    # TODO: implement deterministic grouped splitting after deciding the exact v1 CV policy.
    raise NotImplementedError("Grouped splitting is not implemented yet.")
