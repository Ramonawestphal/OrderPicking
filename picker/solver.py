"""Exact brute-force optimum for small instances.

Computes the minimum number of MOVE actions needed to collect the order and
return to the depot. It searches over which shelves to visit (the choice of
source location per unit) and over the order in which they are visited, using
the layout's cached BFS distances. Returns ``None`` when the instance is too
large.
"""

from __future__ import annotations

from itertools import permutations

from .layout import Cell, Layout
from .types import Instance

MAX_ITEMS_FOR_EXACT = 7

_DEPOT: Cell = (0, 0)


def optimal_cost(layout: Layout, instance: Instance) -> int | None:
    """Minimum number of MOVE actions over all pick sequences, or None if large."""
    order = {sku: qty for sku, qty in instance.order.items() if qty > 0}
    total_units = sum(order.values())
    if total_units > MAX_ITEMS_FOR_EXACT:
        return None

    # Shelves that stock at least one ordered SKU are the only relevant stops.
    relevant: list[Cell] = [
        cell
        for cell in sorted(instance.shelves)
        if any(instance.shelves[cell].get(sku, 0) > 0 for sku in order)
    ]

    best: int | None = None
    # Enumerate every subset of relevant shelves; keep those that can cover the
    # order, and take the cheapest closed tour visiting the subset's aisles.
    for mask in range(1, 1 << len(relevant)):
        subset = [relevant[i] for i in range(len(relevant)) if mask & (1 << i)]
        if not _covers(instance, subset, order):
            continue
        cost = _tour_cost(layout, subset)
        if best is None or cost < best:
            best = cost
    return best


def _covers(instance: Instance, subset: list[Cell], order: dict[str, int]) -> bool:
    """Return True if the shelves in ``subset`` jointly stock the whole order."""
    for sku, qty in order.items():
        available = sum(instance.shelves[cell].get(sku, 0) for cell in subset)
        if available < qty:
            return False
    return True


def _tour_cost(layout: Layout, subset: list[Cell]) -> int:
    """Minimum closed-walk length from the depot visiting one aisle per shelf."""
    ports: list[list[Cell]] = [layout.adjacent_aisles(shelf) for shelf in subset]
    best: int | None = None
    for perm in permutations(range(len(subset))):
        # DP over which aisle port is used at each stop in this ordering.
        prev: dict[Cell, int] = {
            aisle: layout.distance(_DEPOT, aisle) for aisle in ports[perm[0]]
        }
        for idx in perm[1:]:
            cur: dict[Cell, int] = {}
            for aisle in ports[idx]:
                cur[aisle] = min(
                    cost + layout.distance(prev_aisle, aisle)
                    for prev_aisle, cost in prev.items()
                )
            prev = cur
        total = min(cost + layout.distance(aisle, _DEPOT) for aisle, cost in prev.items())
        if best is None or total < best:
            best = total
    assert best is not None
    return best
