"""The single entry point: ``python -m picker.run``.

Generates instances for each variant and seed, runs the nearest-neighbour
baseline, verifies each resulting final state, and reports success rate and the
optimality gap against the exact solver.
"""

from __future__ import annotations

import argparse
import sys

from . import baseline, solver
from .env import PickerEnv
from .generator import VARIANTS, generate
from .verifier import verify


class _VariantStats:
    """Accumulates results for one variant across seeds."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.instances = 0
        self.successes = 0
        self.baseline_moves: list[int] = []
        self.optimal_moves: list[int] = []
        self.gaps: list[float] = []


def _run_instance(variant: str, seed: int, stats: _VariantStats) -> None:
    """Generate, solve, verify one instance and fold results into ``stats``."""
    instance = generate(variant, seed)
    env = PickerEnv(instance)
    env.reset()
    actions = baseline.solve(env.layout, instance)
    for action in actions:
        env.step(action)

    ok = verify(env.state().to_json())
    moves = sum(1 for a in actions if a.kind == "MOVE")

    stats.instances += 1
    stats.successes += ok
    stats.baseline_moves.append(moves)

    optimal = solver.optimal_cost(env.layout, instance)
    if optimal is not None:
        stats.optimal_moves.append(optimal)
        if optimal > 0:
            stats.gaps.append((moves - optimal) / optimal * 100.0)


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _print_table(all_stats: list[_VariantStats]) -> None:
    """Print the per-variant results table."""
    header = (
        f"{'variant':<24}{'inst':>6}{'success':>9}"
        f"{'base_mv':>9}{'opt_mv':>9}{'gap_%':>8}"
    )
    print(header)
    print("-" * len(header))
    for st in all_stats:
        rate = st.successes / st.instances if st.instances else 0.0
        print(
            f"{st.name:<24}{st.instances:>6}{rate:>9.2%}"
            f"{_mean(st.baseline_moves):>9.1f}"
            f"{_mean(st.optimal_moves):>9.1f}"
            f"{_mean(st.gaps):>8.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation sweep and return the process exit code."""
    parser = argparse.ArgumentParser(prog="picker.run")
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    parser.add_argument("--variant", type=str, default=None, help="restrict to one variant")
    args = parser.parse_args(argv)

    variants = [args.variant] if args.variant else VARIANTS
    for variant in variants:
        if variant not in VARIANTS:
            parser.error(f"unknown variant: {variant}")

    all_stats: list[_VariantStats] = []
    total_instances = 0
    total_successes = 0
    for variant in variants:
        stats = _VariantStats(variant)
        for seed in range(args.seeds):
            _run_instance(variant, seed, stats)
        all_stats.append(stats)
        total_instances += stats.instances
        total_successes += stats.successes

    _print_table(all_stats)
    overall = total_successes / total_instances if total_instances else 0.0
    print(f"\noverall success rate: {overall:.2%} ({total_successes}/{total_instances})")

    return 0 if total_successes == total_instances else 1


if __name__ == "__main__":
    sys.exit(main())
