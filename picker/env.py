"""The order-picking environment: ``reset``, ``step``, observation, invariants.

The environment is fully observable and deterministic. Item conservation is
asserted after every transition; it is the single most important invariant in
the codebase. Invalid actions never raise and never terminate the episode.
"""

from __future__ import annotations

import copy

from .layout import DIRECTION_DELTAS, Cell, Layout
from .rewards import shaped_reward
from .types import Action, Instance, State


class PickerEnv:
    """A single-agent order-picking episode over one :class:`Instance`."""

    def __init__(self, instance: Instance) -> None:
        """Build the environment and its layout, then reset it."""
        self.instance = instance
        self.layout = Layout(instance.rows, instance.cols)
        self._initial_counts: dict[str, int] = {}
        self._state: State
        self.reset()

    def reset(self) -> dict:
        """Reset to the initial state at the depot and return the observation."""
        shelves = {cell: dict(stock) for cell, stock in self.instance.shelves.items()}
        self._initial_counts = self._count_all(shelves)
        self._state = State(
            instance=self.instance,
            pos=(0, 0),
            shelves=shelves,
            tote={},
            steps_taken=0,
            action_log=[],
            invalid_count=0,
            done=False,
        )
        self._check_invariants()
        return self.observation()

    def state(self) -> State:
        """Return the live mutable state object."""
        return self._state

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        """Apply ``action`` and return ``(observation, reward, done, info)``."""
        info: dict = {}
        st = self._state

        # Any action after termination is invalid and does not change the state.
        if st.done:
            info["invalid"] = True
            return self.observation(), 0.0, True, info

        before = self._snapshot()
        valid = self._apply(action)
        st.action_log.append(action)
        st.steps_taken += 1
        if not valid:
            st.invalid_count += 1
            info["invalid"] = True
        if valid and action.kind == "DONE":
            st.done = True
        if st.steps_taken >= self.instance.step_limit:
            st.done = True

        reward = shaped_reward(self.layout, before, action, st)
        self._check_invariants()
        return self.observation(), reward, st.done, info

    def observation(self) -> dict:
        """Return a JSON-serialisable observation of the current state."""
        st = self._state
        order = self.instance.order
        tote = st.tote
        remaining = {
            sku: max(0, order[sku] - tote.get(sku, 0)) for sku in sorted(order)
        }
        shelves_out: dict[str, dict[str, int]] = {}
        for cell in sorted(st.shelves):
            stock = {
                sku: st.shelves[cell][sku]
                for sku in sorted(st.shelves[cell])
                if st.shelves[cell][sku] > 0
            }
            if stock:
                shelves_out[f"{cell[0]},{cell[1]}"] = stock
        return {
            "pos": [st.pos[0], st.pos[1]],
            "tote": {sku: tote[sku] for sku in sorted(tote) if tote[sku] > 0},
            "order": {sku: order[sku] for sku in sorted(order)},
            "remaining": remaining,
            "shelves": shelves_out,
            "steps_taken": st.steps_taken,
            "steps_remaining": self.instance.step_limit - st.steps_taken,
            "done": st.done,
        }

    # -- internal helpers -------------------------------------------------

    def _apply(self, action: Action) -> bool:
        """Mutate the state for ``action``; return True if the action was valid."""
        st = self._state
        kind = action.kind
        if kind == "DONE":
            return True
        if kind == "MOVE":
            delta = DIRECTION_DELTAS.get(action.arg or "")
            if delta is None:
                return False
            target = (st.pos[0] + delta[0], st.pos[1] + delta[1])
            if not self.layout.is_walkable(target):
                return False
            st.pos = target
            return True
        if kind == "PICK":
            sku = action.arg
            for shelf in self.layout.adjacent_shelves(st.pos):
                if st.shelves.get(shelf, {}).get(sku, 0) > 0:
                    st.shelves[shelf][sku] -= 1
                    st.tote[sku] = st.tote.get(sku, 0) + 1
                    return True
            return False
        if kind == "DROP":
            sku = action.arg
            if st.tote.get(sku, 0) <= 0:
                return False
            shelves = self.layout.adjacent_shelves(st.pos)
            if not shelves:
                return False
            shelf = shelves[0]
            st.tote[sku] -= 1
            st.shelves.setdefault(shelf, {})
            st.shelves[shelf][sku] = st.shelves[shelf].get(sku, 0) + 1
            return True
        return False

    def _snapshot(self) -> State:
        """Return a deep-enough copy of the current state for reward shaping."""
        st = self._state
        return State(
            instance=self.instance,
            pos=st.pos,
            shelves={cell: dict(stock) for cell, stock in st.shelves.items()},
            tote=dict(st.tote),
            steps_taken=st.steps_taken,
            action_log=list(st.action_log),
            invalid_count=st.invalid_count,
            done=st.done,
        )

    @staticmethod
    def _count_all(shelves: dict[Cell, dict[str, int]]) -> dict[str, int]:
        """Return total stock per SKU across all shelves."""
        counts: dict[str, int] = {}
        for stock in shelves.values():
            for sku, n in stock.items():
                counts[sku] = counts.get(sku, 0) + n
        return counts

    def _check_invariants(self) -> None:
        """Assert conservation, walkability, non-negativity, and step bounds."""
        for sku, n0 in self._initial_counts.items():
            on_shelves = sum(s.get(sku, 0) for s in self._state.shelves.values())
            assert (
                on_shelves + self._state.tote.get(sku, 0) == n0
            ), f"conservation violated for {sku}"
        assert self.layout.is_walkable(self._state.pos)
        assert all(v >= 0 for v in self._state.tote.values())
        assert all(v >= 0 for s in self._state.shelves.values() for v in s.values())
        assert self._state.steps_taken <= self.instance.step_limit
