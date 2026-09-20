"""Coverage guard: the exact solver must apply to every generated instance."""

from __future__ import annotations

import pytest

from picker import solver
from picker.generator import VARIANTS, generate
from picker.layout import Layout

_CASES = [(variant, seed) for variant in VARIANTS for seed in range(50)]


@pytest.mark.parametrize("variant,seed", _CASES)
def test_solver_covers_every_instance(variant: str, seed: int) -> None:
    """optimal_cost must return a concrete value for all variants, seeds 0-49."""
    instance = generate(variant, seed)
    layout = Layout(instance.rows, instance.cols)
    assert solver.optimal_cost(layout, instance) is not None
