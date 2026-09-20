"""Exact brute-force optimum for small instances.

Computes the minimum number of MOVE actions needed to collect the order and
return to the depot, searching over which shelves to visit and the order in
which to visit them, using the layout's cached BFS distances.

The search enumerates subsets of the shelves that stock ordered SKUs and
permutations of each covering subset, so its cost is governed by the number of
distinct *stops*, not the number of ordered units: three units taken from one
shelf is a single stop, no harder than one unit. The guard is therefore on the
stop count (``MAX_STOPS_FOR_EXACT``), applied before enumeration begins; the
search is super-exponential in stops and would blow up on a larger warehouse.
Returns ``None`` only when an instance has more relevant shelves than that bound.
"""

from __future__ import annotations

from itertools import permutations

from .layout import Cell, Layout
from .types import Instance

MAX_STOPS_FOR_EXACT = 9

_DEPOT: Cell = (0, 0)


def optimal_cost(layout: Layout, instance: Instance) -> int | None:
    """Minimum number of MOVE actions over all pick sequences, or None if large."""
    order = {sku: qty for sku, qty in instance.order.items() if qty > 0}

    # Shelves that stock at least one ordered SKU are the only relevant stops;
    # the search cost is driven by the number of stops, not by ordered units.
    relevant: list[Cell] = [
        cell
        for cell in sorted(instance.shelves)
        if any(instance.shelves[cell].get(sku, 0) > 0 for sku in order)
    ]
    if len(relevant) > MAX_STOPS_FOR_EXACT:
        return None

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
