# Design decisions

Choices made where the build spec was silent, each with a one-line rationale.

## Required by the spec

- **PICK/DROP tie-break.** When several adjacent shelves qualify, act on the
  lexicographically smallest `(row, col)` shelf, so the transition is
  deterministic and easy to reason about.
- **Invalid actions do not terminate.** They leave the state unchanged, bump
  `invalid_count`, are still appended to `action_log`, and still consume a step;
  this keeps episodes robust and makes the log a faithful record for the
  verifier.
- **Replay-based verification limitation.** The verifier replays through the
  same `PickerEnv` transition code, so a bug present in both the environment and
  the replay would not be caught; this shared-implementation risk is accepted
  for this pass.
- **Step limit.** `step_limit = 6 * (rows * cols)` (378 for the default 7x9)
  gives the baseline ample slack, well above its observed move counts.

## Other open choices

- **Post-termination actions.** Once `done` is set, any further action is
  reported as invalid and does not mutate the state or the log; the episode is
  over, so nothing should change.
- **Environment `step` reward.** `PickerEnv.step` returns `shaped_reward` (the
  correct one), since the broken `naive_reward` is only an exhibit and no code
  path depends on the returned value.
- **Ordered-SKU stock levels.** Non-duplicated ordered SKUs are stocked at a
  single location in exactly the ordered quantity; only V3 adds a second
  location, keeping the duplicate-location behaviour isolated to that variant.
- **Decoy stock.** Every instance places exactly three decoy SKUs (SKUs absent
  from the order), the minimum that makes exact-multiset verification meaningful.
- **Solvability check.** `generate` asserts each ordered SKU's total shelf stock
  meets its ordered quantity; combined with the generous step limit and fully
  connected aisles, this guarantees a solution exists.
- **Solver search.** The exact solver enumerates covering subsets of the shelves
  that stock ordered SKUs and takes the cheapest closed tour over their aisle
  cells, which yields the minimum MOVE count while reusing cached BFS distances.
- **Solver guard is on stops, not units.** Because the search enumerates subsets
  and permutations of the shelves stocking ordered SKUs, its cost scales with the
  number of distinct *stops*, not the number of ordered units (three units from
  one shelf is one stop). The guard is therefore `MAX_STOPS_FOR_EXACT` on
  `len(relevant)`, checked before enumeration; the earlier unit-based guard
  discarded large-but-cheap orders for no computational reason.
- **Pluggable reward.** `PickerEnv.__init__` takes a `reward_fn` keyword that
  defaults to `shaped_reward`, so a consumer can swap in `naive_reward` (the
  broken exhibit) through the normal environment API rather than importing it
  directly; the verifier keeps the default and is unaffected.
- **SKU naming.** SKUs are named `SKU-A`, `SKU-B`, ... in order of assignment,
  with ordered SKUs taking the first letters and decoys the next, for
  reproducible, readable identifiers.
- **State JSON contents.** `State.to_json` also includes `invalid_count` and
  `done` beyond the fields the verifier compares, so a submission is a complete,
  self-describing record.
