# Notes

## Task and scope

Order picking on a 7×9 single-block warehouse grid: walk the aisles, collect the units named in an order, return to the depot. Shelves are blocked cells reachable from either side, so the agent cannot move straight between picks. The problem is a Steiner TSP on a sparse graph, which is what makes aisle traversal the dominant cost in real picking.

Success is machine-checkable, the optimum is exactly computable at this size, and the reward admits a failure mode that is easy to write by accident.

Deliberately excluded: tote capacity (it turns one tour into a vehicle-routing problem), travel times, congestion, stochastic picks, partial observability.

## Decisions

- **The verifier reads only the final state, which carries an append-only action log.** Final-state-only scoring and "a written goal state scores 0" conflict unless the state proves how it was reached. The verifier regenerates the instance from `(variant, seed)` and replays the log.
- **Exact multiset equality, not containment.** Every instance carries three decoy SKUs; under `tote ⊇ order` an agent that sweeps the warehouse scores 1 everywhere.
- **Invalid actions never raise and never terminate.** They leave the state unchanged, consume a step, stay in the log. An environment that crashes on malformed actions is unusable for training; the constraint moves to scoring time.
- **Conservation asserted every transition:** `shelf + tote == initial` per SKU.



## Results

Greedy nearest-neighbour baseline, BFS routing, no training. The exact solver covers all 150 instances, so gaps are full-sample.


| Variant                | Verified | Baseline | Optimal | Gap   |
| ---------------------- | -------- | -------- | ------- | ----- |
| V1 single aisle        | 50/50    | 13.5     | 13.5    | 0.0%  |
| V2 multi aisle         | 50/50    | 30.4     | 23.9    | 27.1% |
| V3 duplicate locations | 50/50    | 30.4     | 23.6    | 28.4% |


0% on V1 is expected: nearest-neighbour is provably optimal within one aisle.

## The documented exploit

`naive_reward` shapes with a rectified term, `0.1 · max(0, d_before − d_after)`: approaching pays, moving away costs nothing. A clipped term is not a potential difference and does not telescope around a loop. `shaped_reward` uses potential-based shaping (Ng, Harada & Russell 1999), `Φ(s) = −0.1 · goal_distance(s)`: zero around any cycle, optimal policy provably unchanged.

V1 seed 0, inside the environment's own 378-step limit:


| Reward          | Oscillator (picks nothing) | Correct solution |
| --------------- | -------------------------- | ---------------- |
| `naive_reward`  | **+15.04**                 | +14.94           |
| `shaped_reward` | −3.76                      | **+13.44**       |


Under the broken reward, doing nothing outscores doing the job. Not via an unbounded horizon, but within the budget the environment already enforces. `tests/test_exploit.py` proves the fix: under shaping the oscillator earns *exactly* `steps × −0.01`, and so does a seeded random closed walk, so the property holds for arbitrary loops and not one shape. A hoarding agent satisfies `tote ⊇ order` and is rejected by exact equality.

338 tests. Beyond the four required: an independent 0-1 BFS optimum agrees with the solver on 36 instances; 600 random-action episodes break no invariant; 14 verifier attacks (seed swap, log reversal, post-`DONE` appends, tampered logs) are rejected while the honest baseline still scores 1.

## Limits

- One exploit found and fixed is not evidence the reward is unhackable; a trained policy would likely find others.
- The verifier is independent of the *agent*, not the *environment.* Replay shares transition code with `PickerEnv`, so a bug in both would pass. Separate for a real application.
- The 27% gap bounds headroom against nearest-neighbour, not against a good learned policy.
- Three variants at fifty seeds is not a difficulty distribution and does not establish generalisation.
- Scoring is binary, so there is no signal on near misses.



## What a buyer would need next

A controlled difficulty distribution over hundreds of instances; graded partial credit; an adversarial audit of the verifier by someone other than its author, since a lab training against a gameable verifier gets a model that games it; a held-out split; cached layout construction and parallel-safe seeding for throughput; documented provenance of the real workflow this derives from, and the rights to it.

## Next step

Train a policy against it. Everything here is inference-free by design, and the honest test of an RL environment is whether a learned agent beats the heuristic *without* finding a hole in the verifier, which would tell me whether 27% is real headroom or an artefact of the instance distribution.