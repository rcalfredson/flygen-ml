from __future__ import annotations


def merge_qc_flags(*flag_groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in flag_groups:
        for flag in group:
            if flag not in merged:
                merged.append(flag)
    return tuple(merged)
