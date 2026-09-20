"""The environment must honour a pluggable reward function."""

from __future__ import annotations

from collections.abc import Callable

from picker import baseline
from picker.env import PickerEnv
from picker.generator import generate
from picker.layout import Layout
from picker.rewards import naive_reward, shaped_reward
from picker.types import Action, Instance, State

_RewardFn = Callable[[Layout, State, Action, State], float]


def _cumulative(instance: Instance, reward_fn: _RewardFn | None) -> float:
    """Return the total reward the baseline earns under ``reward_fn``.

    ``None`` builds the environment with no explicit reward function, exercising
    the default.
    """
    env = PickerEnv(instance) if reward_fn is None else PickerEnv(instance, reward_fn=reward_fn)
    env.reset()
    total = 0.0
    for action in baseline.solve(env.layout, instance):
        _obs, reward, _done, _info = env.step(action)
        total += reward
    return total


def test_reward_fn_changes_cumulative_reward() -> None:
    """A naive-reward env must return a different episode total than the default."""
    instance = generate("V2_MULTI_AISLE", 0)
    naive_total = _cumulative(instance, naive_reward)
    shaped_total = _cumulative(instance, shaped_reward)
    default_total = _cumulative(instance, None)
    assert default_total == shaped_total  # the default is shaped_reward
    assert naive_total != shaped_total  # the broken reward is reachable via the API
