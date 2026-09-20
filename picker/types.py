"""Core dataclasses: :class:`Action`, :class:`Instance`, and :class:`State`.

All types are JSON-serialisable. SKU names have the form ``SKU-A``, ``SKU-B``,
and are always iterated in sorted order wherever output depends on ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Cell = tuple[int, int]

ActionKind = Literal["MOVE", "PICK", "DROP", "DONE"]


@dataclass(frozen=True)
class Action:
    """A single agent action.

    ``arg`` is a direction ("N"/"S"/"E"/"W") for MOVE, a SKU for PICK/DROP, and
    None for DONE.
    """

    kind: ActionKind
    arg: str | None = None

    def to_json(self) -> list:
        """Serialise to a ``[kind, arg]`` list, e.g. ``["MOVE", "N"]``."""
        return [self.kind, self.arg]

    @staticmethod
    def from_json(obj: list) -> "Action":
        """Reconstruct an :class:`Action` from a ``[kind, arg]`` list."""
        kind, arg = obj[0], obj[1]
        return Action(kind=kind, arg=arg)


def _shelves_to_json(shelves: dict[Cell, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Serialise a shelf-stock map to JSON, dropping empty entries, sorted."""
    out: dict[str, dict[str, int]] = {}
    for cell in sorted(shelves):
        stock = {sku: shelves[cell][sku] for sku in sorted(shelves[cell]) if shelves[cell][sku] > 0}
        if stock:
            out[f"{cell[0]},{cell[1]}"] = stock
    return out


@dataclass(frozen=True)
class Instance:
    """An immutable, fully specified order-picking problem."""

    variant: str
    seed: int
    rows: int
    cols: int
    shelves: dict[Cell, dict[str, int]]
    order: dict[str, int]
    step_limit: int

    def to_json(self) -> dict:
        """Serialise the instance to a JSON-compatible dict."""
        return {
            "variant": self.variant,
            "seed": self.seed,
            "rows": self.rows,
            "cols": self.cols,
            "shelves": _shelves_to_json(self.shelves),
            "order": {sku: self.order[sku] for sku in sorted(self.order)},
            "step_limit": self.step_limit,
        }


@dataclass
class State:
    """Mutable episode state. ``to_json`` produces the verifier's input."""

    instance: Instance
    pos: Cell
    shelves: dict[Cell, dict[str, int]]
    tote: dict[str, int]
    steps_taken: int
    action_log: list[Action] = field(default_factory=list)
    invalid_count: int = 0
    done: bool = False

    def to_json(self) -> dict:
        """Serialise the full final state, including provenance and action log."""
        return {
            "variant": self.instance.variant,
            "seed": self.instance.seed,
            "pos": [self.pos[0], self.pos[1]],
            "tote": {sku: self.tote[sku] for sku in sorted(self.tote) if self.tote[sku] > 0},
            "shelves": _shelves_to_json(self.shelves),
            "steps_taken": self.steps_taken,
            "invalid_count": self.invalid_count,
            "done": self.done,
            "action_log": [a.to_json() for a in self.action_log],
        }
