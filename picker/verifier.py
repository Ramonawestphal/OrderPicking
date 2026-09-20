"""Final-state-only scoring by deterministic log replay.

The verifier receives *only* the serialised final state, never the live
environment. It regenerates the instance from the submitted ``variant`` and
``seed`` and replays the ``action_log`` through a fresh environment, comparing
the replayed final state to the submission.

Two deliberate design notes:

- Exact multiset equality of tote and order is required. ``tote`` being a
  superset of the order would let a policy that picks everything score 1, so a
  superset must score 0.
- Replay-based verification shares the transition implementation with the
  environment, so a bug present in both would not be caught. This is a known
  limitation and is recorded in ``DECISIONS.md``.
"""

from __future__ import annotations

from typing import Any

from .env import PickerEnv
from .generator import generate
from .types import Action

_DEPOT = (0, 0)


def verify(final_state_json: dict) -> int:
    """Return 1 if the final state is a valid, correct solution, else 0. Never raises."""
    try:
        return _verify(final_state_json)
    except Exception:
        return 0


def _verify(final_state_json: dict) -> int:
    """Core verification logic; may raise, callers wrap it to return 0."""
    variant = final_state_json["variant"]
    seed = final_state_json["seed"]

    raw_log = final_state_json.get("action_log")
    if not isinstance(raw_log, list) or len(raw_log) == 0:
        return 0
    actions = [Action.from_json(entry) for entry in raw_log]

    # Regenerate the instance from scratch; never trust a submitted instance.
    instance = generate(variant, seed)
    env = PickerEnv(instance)
    env.reset()

    for action in actions:
        _obs, _reward, _done, info = env.step(action)
        if info.get("invalid"):
            return 0

    if actions[-1].kind != "DONE":
        return 0

    replay = env.state()
    if not replay.done:
        return 0
    if replay.pos != _DEPOT:
        return 0

    replay_json = replay.to_json()

    # The replayed final state must match the submission exactly.
    if list(replay.pos) != _as_pair(final_state_json.get("pos")):
        return 0
    if replay_json["tote"] != _normalise_counts(final_state_json.get("tote")):
        return 0
    if replay_json["shelves"] != _normalise_shelves(final_state_json.get("shelves")):
        return 0
    if replay_json["steps_taken"] != final_state_json.get("steps_taken"):
        return 0

    # The tote must equal the order as a multiset, exactly (no superset).
    order = {sku: qty for sku, qty in instance.order.items() if qty > 0}
    if replay_json["tote"] != order:
        return 0

    return 1


def _as_pair(value: Any) -> list[int]:
    """Coerce a submitted position into a two-element list, else raise."""
    return [int(value[0]), int(value[1])]


def _normalise_counts(counts: Any) -> dict[str, int]:
    """Return a positive-count-only dict, sorted, from a submitted count map."""
    if not isinstance(counts, dict):
        return {}
    return {sku: int(counts[sku]) for sku in sorted(counts) if int(counts[sku]) > 0}


def _normalise_shelves(shelves: Any) -> dict[str, dict[str, int]]:
    """Return a normalised shelf map dropping empty cells and zero counts."""
    if not isinstance(shelves, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for key in sorted(shelves):
        stock = _normalise_counts(shelves[key])
        if stock:
            out[key] = stock
    return out
