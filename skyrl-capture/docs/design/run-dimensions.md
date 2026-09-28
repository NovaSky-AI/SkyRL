# What a run is made of, and why `task × step` is not it

Design note, and a record of what was removed. The grid is **gone**: the
endpoint, the SQL, the record backend's copy, the CLI command and the SDK
method. What replaced it was already there — `GET /v1/trajectories` filters on
`run_id`, `task_id`, `step` and `status`, and a record answers that same route
through the same reader.

## The claim

A run has three primitives:

| | |
| --- | --- |
| **run id** | the job |
| **session id** | one attempt — the trajectory id, which is also the engine's session key |
| **step** | a tag saying when in the job it happened |

`task_id` is not one of them. It is an *opinion* about what groups sessions —
a good one for a fixed benchmark suite, and wrong the moment the harness does
not attempt the same tasks at every step.

**Sessions are not guaranteed to recur across steps.** A shuffled dataloader, an
epoch boundary, curriculum sampling, or rejection sampling all break it.

## What assumed otherwise, and is gone

The storage layer was already fine: `task_id` and `step` are both nullable
columns on `trajectories`, so nothing forced a task-shaped world. The assumption
lived one layer up, and all of it is removed:

| | |
| --- | --- |
| `store/runs.py:grid()` | `GROUP BY task_id, step` — deleted |
| the record backend's `grid()` | the same view over its own index — deleted with it |
| `GET /v1/runs/{id}/grid` | on both the live and record-backed servers — deleted |
| `skyrl-capture grid`, `client.run_grid()` | deleted |
| `Run.task_count`, `Run.steps` | kept as **counts and filter values**, not axes |

`Run.steps` earns its place in the new shape: it is what `?step=` accepts for a
run, so a viewer can offer the filter without a second query.

## Why the grid is the wrong shape

**It is a dense cross-product over sparse data.** It reads well only when the
same `task_id` appears at every step. Otherwise most cells are empty, and the
view degrades quietly rather than failing — the worst way to be wrong.

The sharper problem is what it invites. A grid is **relational**: put cells next
to each other and the reader asks "did *this* one improve over *that* one".
When sessions do not recur, that question has no answer — there is no pair to
compare. The honest cross-step view in that regime is **distributional**: per
step, the spread of reward, of context length, of turn count, compared step to
step as distributions rather than cells.

A second cost, independent of recurrence: `task_id` occupies the only
privileged dimension slot. Anything else worth grouping by — dataset shard,
difficulty, prompt template, seed, model variant — goes in `annotations` and is
second class, not reachable by the same machinery.

## What it is now

Flat, filtered:

```
GET /v1/trajectories?run_id=<id>            every attempt in the run
GET /v1/trajectories?run_id=<id>&step=0     one step
GET /v1/trajectories?run_id=<id>&task_id=t1 one task across the run
```

`task_id` stays a **column** — it is indexed and hot, and demoting it to a tag
would trade an index for a jsonb lookup to buy purity. What changed is that
nothing calls it an axis, and nothing builds a matrix out of it.

Two things the grid had to decide, that a list does not have to:

* **a resample.** Two attempts at one task and step were averaged into a cell.
  They are two rows now, and the reader decides what that means.
* **an absent cell.** A grid had to distinguish "never attempted" from "scored
  zero", and document the distinction. A listing has nothing to distinguish:
  the attempt is simply not in it.

**Still open: aggregation.** Grouped statistics — average context length at
step 3, reward spread per step — have no endpoint. When one is wanted, it should
be `group(run_id, by=[...], metrics=[...])` with `by=` accepting annotation keys
as well as `task_id` and `step`, rather than another fixed view. Nothing needs
it yet, and the note below says what it would cost.

## The one part that is not free

Cross-step statistics need per-trajectory numbers to aggregate. Today that is
`exchange_count`, `node_count`, `accepted_count`, and whatever the harness
writes into `annotations`.

**Context length is not among them.** It lives in the token graph and the
export, so "average context length at step 3" means reading every record for
that step. Derivable, but not cheap.

If that becomes a common question, the hook is a numeric summary written at
finalization — prompt tokens, total tokens, turns — beside the counters that are
already there. It is the only piece with any "capture it now" character, and
even that is pay-later rather than lose-forever: the records hold the tokens.

## What was deferred, and what it costs

Only aggregation, and nothing is lost by waiting:

* the schema permits a run with no tasks — `task_id` is nullable;
* the filter-and-explore path is `GET /v1/trajectories?run_id=&step=`, which
  exists and is tested on both backends;
* the token summary can be backfilled from the records, which hold the tokens.

**Done, for the part that is a removal.** The grid is gone, so nothing is
building on it.

**Not done: aggregation.** `group(run_id, by=..., metrics=...)` is unwritten,
and cross-step statistics are the reason it would exist. Revisit when a question
actually needs it — and note that the expensive half is the metrics, not the
grouping.
