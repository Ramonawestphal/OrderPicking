"""Warehouse grid geometry, walkability, and cached BFS distances.

A single-block warehouse laid out on an integer grid. Rows 0 and ``rows - 1``
are horizontal cross-aisles; even columns are vertical picking aisles; odd
interior columns hold shelves. All-pairs shortest-path distances over walkable
cells are precomputed once per :class:`Layout` and cached, because the grid is
tiny and this keeps every downstream module simple.
"""

from __future__ import annotations

from collections import deque

Cell = tuple[int, int]

# Neighbour offsets in the fixed tie-break order N, S, E, W.
# N = row + 1, S = row - 1, E = col + 1, W = col - 1.
_DIRECTIONS: list[tuple[str, int, int]] = [
    ("N", 1, 0),
    ("S", -1, 0),
    ("E", 0, 1),
    ("W", 0, -1),
]

# Mapping from direction name to (drow, dcol), used by the environment.
DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    name: (dr, dc) for name, dr, dc in _DIRECTIONS
}


class Layout:
    """Warehouse grid with walkability queries and cached path metrics."""

    def __init__(self, rows: int = 7, cols: int = 9) -> None:
        """Create a ``rows`` x ``cols`` warehouse and precompute BFS distances."""
        self.rows = rows
        self.cols = cols
        self._walkable: list[Cell] = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if self.is_walkable((r, c))
        ]
        self._dist: dict[Cell, dict[Cell, int]] = {}
        self._parent: dict[Cell, dict[Cell, Cell]] = {}
        self._path_cache: dict[tuple[Cell, Cell], list[Cell]] = {}
        self._precompute()

    def is_walkable(self, cell: Cell) -> bool:
        """Return True if ``cell`` is in bounds and is an aisle cell."""
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return r == 0 or r == self.rows - 1 or c % 2 == 0

    def is_shelf(self, cell: Cell) -> bool:
        """Return True if ``cell`` is an in-bounds shelf (blocked) cell."""
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return c % 2 == 1 and 0 < r < self.rows - 1

    def storage_locations(self) -> list[Cell]:
        """Return all shelf cells, sorted by ``(row, col)``."""
        return sorted(
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.is_shelf((r, c))
        )

    def adjacent_shelves(self, cell: Cell) -> list[Cell]:
        """Return shelf cells horizontally adjacent to an aisle ``cell``, sorted."""
        r, c = cell
        return sorted(
            s for s in ((r, c - 1), (r, c + 1)) if self.is_shelf(s)
        )

    def adjacent_aisles(self, shelf: Cell) -> list[Cell]:
        """Return walkable aisle cells horizontally adjacent to ``shelf``, sorted."""
        r, c = shelf
        return sorted(
            a for a in ((r, c - 1), (r, c + 1)) if self.is_walkable(a)
        )

    def distance(self, a: Cell, b: Cell) -> int:
        """Return the BFS shortest-path length between walkable cells ``a`` and ``b``."""
        return self._dist[a][b]

    def path(self, a: Cell, b: Cell) -> list[Cell]:
        """Return a deterministic shortest path of cells from ``a`` to ``b`` inclusive.

        Ties are broken by expanding neighbours in the fixed order N, S, E, W,
        so the path is a pure function of ``(a, b)``.
        """
        if (a, b) in self._path_cache:
            return list(self._path_cache[(a, b)])
        parents = self._parent[a]
        cells: list[Cell] = [b]
        cur = b
        while cur != a:
            cur = parents[cur]
            cells.append(cur)
        cells.reverse()
        self._path_cache[(a, b)] = cells
        return list(cells)

    def _neighbours(self, cell: Cell) -> list[Cell]:
        """Return walkable neighbours of ``cell`` in the fixed N, S, E, W order."""
        r, c = cell
        result: list[Cell] = []
        for _name, dr, dc in _DIRECTIONS:
            nxt = (r + dr, c + dc)
            if self.is_walkable(nxt):
                result.append(nxt)
        return result

    def _precompute(self) -> None:
        """Run BFS from every walkable cell, caching distances and parents."""
        for source in self._walkable:
            dist: dict[Cell, int] = {source: 0}
            parent: dict[Cell, Cell] = {}
            queue: deque[Cell] = deque([source])
            while queue:
                cur = queue.popleft()
                for nxt in self._neighbours(cur):
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        parent[nxt] = cur
                        queue.append(nxt)
            self._dist[source] = dist
            self._parent[source] = parent


_LAYOUT_CACHE: dict[tuple[int, int], Layout] = {}


def get_layout(rows: int = 7, cols: int = 9) -> Layout:
    """Return a shared :class:`Layout` for ``(rows, cols)``, building it once.

    A :class:`Layout` is a pure function of its dimensions and is never mutated
    through its public API (``path`` only fills an internal, result-preserving
    memo), so a single instance is safely shared across every environment with
    the same grid size instead of rebuilding the all-pairs BFS each time.
    """
    key = (rows, cols)
    layout = _LAYOUT_CACHE.get(key)
    if layout is None:
        layout = Layout(rows, cols)
        _LAYOUT_CACHE[key] = layout
    return layout
