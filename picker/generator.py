"""Seeded, reproducible generation of order-picking instances.

``generate(variant, seed)`` is a pure function of its arguments: it uses only a
local ``random.Random(seed)`` and never touches the global RNG or any external
state. Three variants of increasing difficulty are supported.
"""

from __future__ import annotations

import random

from .layout import Cell, get_layout
from .types import Instance

VARIANTS: list[str] = [
    "V1_SINGLE_AISLE",
    "V2_MULTI_AISLE",
    "V3_DUPLICATE_LOCATIONS",
]

_DEFAULT_ROWS = 7
_DEFAULT_COLS = 9
_DECOY_COUNT = 3


def _sku(i: int) -> str:
    """Return the canonical SKU name for index ``i`` (0 -> ``SKU-A``)."""
    return f"SKU-{chr(ord('A') + i)}"


def _add(shelves: dict[Cell, dict[str, int]], cell: Cell, sku: str, n: int) -> None:
    """Add ``n`` units of ``sku`` to the shelf at ``cell``."""
    shelves.setdefault(cell, {})
    shelves[cell][sku] = shelves[cell].get(sku, 0) + n


def generate(variant: str, seed: int) -> Instance:
    """Generate a deterministic :class:`Instance` for ``(variant, seed)``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    rng = random.Random(seed)
    layout = get_layout(_DEFAULT_ROWS, _DEFAULT_COLS)
    rows, cols = layout.rows, layout.cols
    columns = sorted({c for _r, c in layout.storage_locations()})
    cells_by_col: dict[int, list[Cell]] = {
        c: [(r, c) for r in range(1, rows - 1)] for c in columns
    }
    all_cells: list[Cell] = [cell for c in columns for cell in cells_by_col[c]]
    step_limit = 6 * (rows * cols)

    shelves: dict[Cell, dict[str, int]] = {}
    order: dict[str, int] = {}

    if variant == "V1_SINGLE_AISLE":
        n_ord = rng.randint(2, 3)
        ordered = [_sku(i) for i in range(n_ord)]
        order = {s: 1 for s in ordered}
        col = rng.choice(columns)
        cells = cells_by_col[col]
        for s in ordered:
            _add(shelves, rng.choice(cells), s, 1)
        for s in (_sku(n_ord + i) for i in range(_DECOY_COUNT)):
            _add(shelves, rng.choice(cells), s, rng.randint(1, 2))

    elif variant == "V2_MULTI_AISLE":
        n_ord = rng.randint(4, 6)
        ordered = [_sku(i) for i in range(n_ord)]
        order = {s: rng.randint(1, 3) for s in ordered}
        cols_shuf = columns[:]
        rng.shuffle(cols_shuf)
        for idx, s in enumerate(ordered):
            col = cols_shuf[idx % len(cols_shuf)]
            _add(shelves, rng.choice(cells_by_col[col]), s, order[s])
        for s in (_sku(n_ord + i) for i in range(_DECOY_COUNT)):
            _add(shelves, rng.choice(all_cells), s, rng.randint(1, 2))

    else:  # V3_DUPLICATE_LOCATIONS
        n_ord = rng.randint(4, 6)
        ordered = [_sku(i) for i in range(n_ord)]
        order = {s: rng.randint(1, 3) for s in ordered}
        cols_shuf = columns[:]
        rng.shuffle(cols_shuf)
        placement: dict[str, Cell] = {}
        for idx, s in enumerate(ordered):
            col = cols_shuf[idx % len(cols_shuf)]
            cell = rng.choice(cells_by_col[col])
            _add(shelves, cell, s, order[s])
            placement[s] = cell
        dup = ordered[rng.randrange(n_ord)]
        alt = rng.choice([c for c in all_cells if c != placement[dup]])
        _add(shelves, alt, dup, 1)
        for s in (_sku(n_ord + i) for i in range(_DECOY_COUNT)):
            _add(shelves, rng.choice(all_cells), s, rng.randint(1, 2))

    instance = Instance(
        variant=variant,
        seed=seed,
        rows=rows,
        cols=cols,
        shelves=shelves,
        order=order,
        step_limit=step_limit,
    )
    _assert_solvable(instance)
    return instance


def _assert_solvable(instance: Instance) -> None:
    """Assert every ordered SKU has enough total shelf stock to satisfy the order."""
    assert instance.order, "order must be non-empty"
    for sku, qty in instance.order.items():
        total = sum(stock.get(sku, 0) for stock in instance.shelves.values())
        assert total >= qty, f"insufficient stock for {sku}: {total} < {qty}"
