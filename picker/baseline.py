"""A nearest-neighbour heuristic picking agent.

No training and no search beyond greedy nearest-neighbour: repeatedly walk to
the closest pickable location for a still-required SKU, pick one unit, and when
the order is complete return to the depot and finish.
"""

from __future__ import annotations

from .layout import DIRECTION_DELTAS, Cell, Layout
from .types import Action, Instance

_DEPOT: Cell = (0, 0)
_DELTA_TO_DIR: dict[tuple[int, int], str] = {
    delta: name for name, delta in DIRECTION_DELTAS.items()
}


def _moves_along(layout: Layout, a: Cell, b: Cell) -> list[Action]:
    """Return the MOVE actions that walk the shortest path from ``a`` to ``b``."""
    cells = layout.path(a, b)
    actions: list[Action] = []
    for prev, cur in zip(cells, cells[1:]):
        delta = (cur[0] - prev[0], cur[1] - prev[1])
        actions.append(Action("MOVE", _DELTA_TO_DIR[delta]))
    return actions


def solve(layout: Layout, instance: Instance) -> list[Action]:
    """Nearest-neighbour pick ordering with shortest-path routing.

    Returns an action log. Ties are broken by sorted SKU name, then by
    lexicographic ``(row, col)`` of the target aisle cell.
    """
    shelves: dict[Cell, dict[str, int]] = {
        cell: dict(stock) for cell, stock in instance.shelves.items()
    }
    remaining: dict[str, int] = {
        sku: qty for sku, qty in instance.order.items() if qty > 0
    }
    pos: Cell = _DEPOT
    actions: list[Action] = []

    while any(n > 0 for n in remaining.values()):
        best: tuple[int, str, Cell] | None = None  # (distance, sku, aisle)
        for sku in sorted(remaining):
            if remaining[sku] <= 0:
                continue
            for shelf in sorted(shelves):
                if shelves[shelf].get(sku, 0) <= 0:
                    continue
                for aisle in layout.adjacent_aisles(shelf):
                    cand = (layout.distance(pos, aisle), sku, aisle)
                    if best is None or cand < best:
                        best = cand
        assert best is not None
        _dist, sku, aisle = best
        actions.extend(_moves_along(layout, pos, aisle))
        pos = aisle
        # Mirror the environment's PICK rule: take from the lexicographically
        # smallest adjacent shelf that holds the SKU.
        for shelf in layout.adjacent_shelves(pos):
            if shelves.get(shelf, {}).get(sku, 0) > 0:
                shelves[shelf][sku] -= 1
                break
        actions.append(Action("PICK", sku))
        remaining[sku] -= 1

    actions.extend(_moves_along(layout, pos, _DEPOT))
    actions.append(Action("DONE", None))
    return actions
