"""Rendering behaviour of the results table."""

from __future__ import annotations

import pytest

from picker.run import _VariantStats, _print_table


def _row_for(capsys: pytest.CaptureFixture[str], name: str) -> str:
    """Return the printed table row that begins with ``name``."""
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith(name)]
    assert rows, f"no table row for {name}"
    return rows[0]


def test_empty_subsample_renders_na(capsys: pytest.CaptureFixture[str]) -> None:
    """With no covered instances, the comparative columns show n/a, not 0.0."""
    stats = _VariantStats("V_TEST")
    stats.instances = 5
    stats.successes = 5
    stats.baseline_moves_all = [10, 12, 14, 16, 18]
    # stats.covered intentionally left empty.

    _print_table([stats])
    row = _row_for(capsys, "V_TEST")

    assert row.count("n/a") == 3  # base_mv, opt_mv and gap_%
    assert " 0.0" not in row  # the empty subsample must not read as optimal


def test_covered_subsample_renders_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    """With covered instances, the comparative columns are numeric, not n/a."""
    stats = _VariantStats("V_TEST")
    stats.instances = 2
    stats.successes = 2
    stats.baseline_moves_all = [10, 20]
    stats.covered = [(10, 8), (20, 16)]

    _print_table([stats])
    row = _row_for(capsys, "V_TEST")

    assert "n/a" not in row
