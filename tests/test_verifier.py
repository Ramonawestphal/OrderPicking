"""The four required verifier cases, run across several variants and seeds."""

from __future__ import annotations

import pytest

from picker import baseline
from picker.env import PickerEnv
from picker.generator import generate
from picker.layout import DIRECTION_DELTAS
from picker.types import Action
from picker.verifier import verify

_DELTA_TO_DIR = {delta: name for name, delta in DIRECTION_DELTAS.items()}

# A representative matrix of variants and seeds.
_CASES = [
    (variant, seed)
    for variant in ("V1_SINGLE_AISLE", "V2_MULTI_AISLE", "V3_DUPLICATE_LOCATIONS")
    for seed in (0, 1, 2, 3, 4)
]


def _baseline_final_json(variant: str, seed: int) -> dict:
    """Run the baseline to termination and return the serialised final state."""
    instance = generate(variant, seed)
    env = PickerEnv(instance)
    env.reset()
    for action in baseline.solve(env.layout, instance):
        env.step(action)
    return env.state().to_json()


@pytest.mark.parametrize("variant,seed", _CASES)
def test_initial_state_scores_zero(variant: str, seed: int) -> None:
    """A freshly reset state with an empty action log must score 0."""
    env = PickerEnv(generate(variant, seed))
    env.reset()
    assert verify(env.state().to_json()) == 0


@pytest.mark.parametrize("variant,seed", _CASES)
def test_correct_solution_scores_one(variant: str, seed: int) -> None:
    """The baseline's final state must score 1."""
    assert verify(_baseline_final_json(variant, seed)) == 1


@pytest.mark.parametrize("variant,seed", _CASES)
def test_invalid_action_scores_zero(variant: str, seed: int) -> None:
    """Splicing a MOVE into a shelf cell into a valid log must score 0."""
    instance = generate(variant, seed)
    env = PickerEnv(instance)
    env.reset()
    log = baseline.solve(env.layout, instance)

    spliced: list[Action] | None = None
    for i, action in enumerate(log):
        pos = env.state().pos
        shelves_adj = env.layout.adjacent_shelves(pos)
        if shelves_adj:
            shelf = shelves_adj[0]
            delta = (shelf[0] - pos[0], shelf[1] - pos[1])
            bad_move = Action("MOVE", _DELTA_TO_DIR[delta])
            spliced = log[:i] + [bad_move] + log[i:]
            break
        env.step(action)

    assert spliced is not None, "expected the baseline to stand beside a shelf"

    replay = PickerEnv(instance)
    replay.reset()
    for action in spliced:
        replay.step(action)
    assert verify(replay.state().to_json()) == 0


@pytest.mark.parametrize("variant,seed", _CASES)
def test_written_goal_state_scores_zero(variant: str, seed: int) -> None:
    """A hand-written goal state with no genuine action log must score 0."""
    instance = generate(variant, seed)
    goal = {
        "variant": variant,
        "seed": seed,
        "pos": [0, 0],
        "tote": {sku: qty for sku, qty in sorted(instance.order.items())},
        "shelves": {},
        "steps_taken": 0,
        "invalid_count": 0,
        "done": True,
        "action_log": [],
    }
    assert verify(goal) == 0
