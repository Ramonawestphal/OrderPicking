"""Reward functions for the order-picking task.

Two functions live side by side and share one signature. ``naive_reward`` is a
required exhibit and is *deliberately broken*: its shaping term is rectified, so
approaching a target is rewarded but moving away is not penalised, letting an
agent farm reward around a closed loop without completing the order.

``shaped_reward`` is the fix. It replaces the rectified term with a
potential-based shaping term (Ng, Harada & Russell, 1999): with
``Phi(s) = -0.1 * goal_distance(s)`` and shaping ``GAMMA * Phi(after) -
Phi(before)``, the shaping telescopes to zero around any closed loop, so the
oscillation exploit earns nothing while the optimal policy is unchanged.
"""

from __future__ import annotations

from .layout import Layout
from .types import Action, State

GAMMA: float = 1.0

_DEPOT = (0, 0)


def goal_distance(layout: Layout, state: State) -> int:
    """Shortest walking distance from the agent to its next useful target.

    If items are still required, this is the minimum over all aisle cells
    adjacent to a shelf holding a still-required SKU. If the order is complete,
    it is the distance to the depot.
    """
    order = state.instance.order
    remaining = {sku for sku in order if order[sku] - state.tote.get(sku, 0) > 0}
    if not remaining:
        return layout.distance(state.pos, _DEPOT)
    best: int | None = None
    for shelf in sorted(state.shelves):
        stock = state.shelves[shelf]
        if any(stock.get(sku, 0) > 0 for sku in remaining):
            for aisle in layout.adjacent_aisles(shelf):
                d = layout.distance(state.pos, aisle)
                if best is None or d < best:
                    best = d
    # Conservation guarantees a still-required SKU is always present on a shelf.
    assert best is not None
    return best


def _units_still_required_added(before: State, after: State) -> float:
    """Count units of still-required SKUs added to the tote by this transition."""
    order = before.instance.order
    total = 0
    for sku in sorted(order):
        delta = after.tote.get(sku, 0) - before.tote.get(sku, 0)
        if delta > 0:
            required_before = max(0, order[sku] - before.tote.get(sku, 0))
            total += min(delta, required_before)
    return float(total)


def _order_satisfied_at_depot(action: Action, after: State) -> bool:
    """Return True if ``action`` is DONE with the tote exactly the order at depot."""
    if action.kind != "DONE" or after.pos != _DEPOT:
        return False
    order = after.instance.order
    tote = {sku: n for sku, n in after.tote.items() if n > 0}
    return tote == {sku: n for sku, n in order.items() if n > 0}


def naive_reward(layout: Layout, before: State, action: Action, after: State) -> float:
    """Deliberately broken reward with a rectified (one-sided) shaping term."""
    base = _units_still_required_added(before, after)
    step = -0.01
    shape = 0.1 * max(0, goal_distance(layout, before) - goal_distance(layout, after))
    finish = 10.0 if _order_satisfied_at_depot(action, after) else 0.0
    return base + step + shape + finish


def shaped_reward(layout: Layout, before: State, action: Action, after: State) -> float:
    """Correct reward using potential-based shaping (Ng, Harada & Russell, 1999)."""
    base = _units_still_required_added(before, after)
    step = -0.01
    phi_before = -0.1 * goal_distance(layout, before)
    phi_after = -0.1 * goal_distance(layout, after)
    shape = GAMMA * phi_after - phi_before
    finish = 10.0 if _order_satisfied_at_depot(action, after) else 0.0
    return base + step + shape + finish
