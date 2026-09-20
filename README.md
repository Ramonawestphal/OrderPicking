# Order-picking RL environment

Run the full evaluation sweep from a fresh clone, no install step required:

    python -m picker.run

## What this models

A single agent walks a small single-block warehouse on a grid, picking the
units named in an order from shelves and returning them to the depot. It is a
locally runnable RL environment (no agent is trained here) plus a replay-based
verifier, a nearest-neighbour baseline, and an exact solver for small instances.

## Action space

Actions are `MOVE d` (`d` in N/S/E/W), `PICK sku`, `DROP sku`, and `DONE`.

- `MOVE` is invalid if the target cell is out of bounds or not walkable.
- `PICK`/`DROP` act on an adjacent shelf; ties break to the lexicographically
  smallest `(row, col)` shelf.
- Invalid actions never raise and never terminate: the state is unchanged,
  `invalid_count` increments, and the action is still logged.
- Item conservation (`shelf_count + tote_count == initial_count`) is asserted
  after every transition.

## Verifier contract

`picker.verifier.verify(final_state_json)` returns `1` for a valid, correct
solution and `0` otherwise, and never raises. It regenerates the instance from
the submitted `(variant, seed)`, replays `action_log` through a fresh
environment, and scores `0` unless the replay is byte-consistent with the
submission, ends with `DONE` at the depot, and the tote equals the order as an
**exact multiset** (a superset scores `0`).

## Rewards: broken vs. fixed

Both live in `picker/rewards.py`:

- `naive_reward` is **deliberately broken**: its distance-shaping term is
  rectified (approaching a target is rewarded, moving away is not penalised),
  so reward can be farmed around a closed loop without completing the order.
- `shaped_reward` is the fix: potential-based shaping (Ng, Harada & Russell,
  1999) that telescopes to zero around any closed loop, leaving the optimal
  policy unchanged.

`PickerEnv(instance, reward_fn=...)` selects which reward `step()` returns; it
defaults to `shaped_reward`, and passing `naive_reward` exercises the broken
exhibit through the normal environment API.

## Repo map

    picker/
      layout.py      grid geometry, walkability, cached BFS distances
      types.py       Action, Instance, State dataclasses (JSON-serialisable)
      env.py         reset(), step(), observation, invariants
      generator.py   seeded instance generation, 3 variants
      rewards.py     naive_reward (broken) and shaped_reward (fixed)
      verifier.py    final-state-only scoring via log replay
      baseline.py    nearest-neighbour heuristic agent
      solver.py      exact brute-force optimum for small instances
      run.py         the one command
    tests/
      test_verifier.py    the four required verifier cases
      test_solver.py      exact solver covers every generated instance
      test_env_reward.py  step() honours a pluggable reward_fn

## Variants

`V1_SINGLE_AISLE` (sanity), `V2_MULTI_AISLE`, and `V3_DUPLICATE_LOCATIONS`
(a SKU stocked at two or more locations). `run.py` accepts `--seeds N` and
`--variant NAME`.

The results table reports `base_all` (mean baseline moves over every instance)
and `opt_n` (how many instances the exact solver covered); the `base_mv`,
`opt_mv` and `gap_%` columns are all computed over that same covered subsample
so they compare like for like. If any variant is only partially covered, a
warning naming those variants is printed after the table.

## Running the tests

    python -m pytest -q

`pytest` is the only permitted third-party dependency and is used only under
`tests/`.
