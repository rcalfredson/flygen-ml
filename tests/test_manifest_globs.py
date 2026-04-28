from __future__ import annotations

from flygen_ml.cli.build_manifest_from_globs import _parse_repeat_fly_indices
from flygen_ml.manifest_globs import build_manifest_rows_from_glob_specs, load_manifest_glob_specs


def test_load_manifest_glob_specs_splits_pattern_lists(tmp_path):
    spec_path = tmp_path / "spec.csv"
    spec_path.write_text(
        "\n".join(
            [
                "genotype,cohort,chamber,training_idx,fly_idx,patterns",
                'Control>Kir,antennae-intact,large,1,,"/tmp/a_*,/tmp/b_*"',
            ]
        )
    )

    specs = load_manifest_glob_specs(spec_path)

    assert len(specs) == 1
    assert specs[0].genotype == "Control>Kir"
    assert specs[0].cohort == "antennae-intact"
    assert specs[0].patterns == ("/tmp/a_*", "/tmp/b_*")


def test_build_manifest_rows_from_glob_specs_discovers_paired_data_and_trx(tmp_path):
    day_dir = tmp_path / "2025-07-15" / "night"
    day_dir.mkdir(parents=True)
    stem_a = day_dir / "c31_test"
    stem_b = day_dir / "c32_test"
    for stem in (stem_a, stem_b):
        stem.with_suffix(".avi").write_text("")
        stem.with_suffix(".data").write_text("")
        stem.with_suffix(".trx").write_text("")

    spec_path = tmp_path / "spec.csv"
    spec_path.write_text(
        "\n".join(
            [
                "genotype,cohort,chamber,training_idx,fly_idx,patterns",
                f'PFN>Kir,antennae-removed,large,1,,"{day_dir}/c3[12]_*"',
            ]
        )
    )

    specs = load_manifest_glob_specs(spec_path)
    rows = build_manifest_rows_from_glob_specs(specs)

    assert len(rows) == 2
    assert {row.genotype for row in rows} == {"PFN>Kir"}
    assert {row.cohort for row in rows} == {"antennae-removed"}
    assert {row.chamber for row in rows} == {"large"}
    assert {row.training_idx for row in rows} == {1}
    assert {row.date for row in rows} == {"2025-07-15"}
    assert all(row.data_path.suffix == ".data" for row in rows)
    assert all(row.trx_path.suffix == ".trx" for row in rows)


def test_build_manifest_rows_from_glob_specs_repeats_fly_indices(tmp_path):
    day_dir = tmp_path / "2025-07-15" / "night"
    day_dir.mkdir(parents=True)
    stem = day_dir / "c31_test"
    stem.with_suffix(".data").write_text("")
    stem.with_suffix(".trx").write_text("")

    spec_path = tmp_path / "spec.csv"
    spec_path.write_text(
        "\n".join(
            [
                "genotype,cohort,chamber,training_idx,fly_idx,patterns",
                f'PFN>Kir,antennae-removed,large,1,,"{day_dir}/c31_*"',
            ]
        )
    )

    specs = load_manifest_glob_specs(spec_path)
    rows = build_manifest_rows_from_glob_specs(specs, repeated_fly_indices=(0, 1))

    assert len(rows) == 2
    assert {row.fly_idx for row in rows} == {0, 1}
    assert {row.sample_key.endswith("__fly0") for row in rows if row.fly_idx == 0} == {True}
    assert {row.sample_key.endswith("__fly1") for row in rows if row.fly_idx == 1} == {True}


def test_parse_repeat_fly_indices_parses_csv_argument():
    assert _parse_repeat_fly_indices(None) is None
    assert _parse_repeat_fly_indices("0,1") == (0, 1)
