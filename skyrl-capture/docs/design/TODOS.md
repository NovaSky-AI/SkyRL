# Open work

What is decided but unbuilt, each entry pointing at the note that argued it so
this file stays a list and not a second argument. `docs/LOG.md` is where
finished work is accounted for; nothing closed is kept here.

**What is being worked on right now is in
[demo_2026-09-18.md](demo_2026-09-18.md).** This file is what is *not* --
decided, unscheduled, and waiting for a reason to start. When that file is
empty it goes, and anything that survived comes back here.

---

# Next next

Very far into the future. Nothing here is scheduled and nothing here blocks
anything. Kept because the reasoning is worth more than the re-derivation.

## Metrics across steps

**[run-dimensions.md](run-dimensions.md)**

The cheap half is built: a run summary carries `step_counts`, so "how many
trajectories at each step" is answerable without reading anything.

What is left is the expensive half -- `group(run_id, by=[...], metrics=[...])`
with metrics like average context length, which means reading every record
because no token summary is stored per trajectory. Backfillable from the
records, so waiting costs nothing.

## `tree_samples`: one row per trajectory, tree attention mask

**[packed-tree-export.md](packed-tree-export.md)**

Additive; `token_samples` does not change. Needs `node_spans` first, which is
in the current iteration.

`examples/agents/recursive.py` gave this a real number on 2026-09-18:
**2.35x** -- 5,471 tokens across 22 paths against 2,325 distinct node tokens,
on a workload that forks at every level. Still 0% on one that never rewrites
its history, which remains the thing to check before building.

## A return path from deriving to the state

**[single-pod-store.md](single-pod-store.md)**

Deriving an exchange has to happen in the process that owns the state, because
a spawned worker writes into its own copy. That caps a pod at one core's worth
of parsing and graph-building.

Nothing needs more yet -- a 1000-trajectory run did not come near it -- and the
answer when something does is probably another pod rather than this. But if one
pod ever has to derive faster than one core can, the shape is a return path:
workers parse, and hand `ExchangeCommitted` events back to the owner rather
than writing them.

## Snapshots in the record

**[record-format.md](record-format.md)**

Opening a record replays its whole log. That is fast into the hundreds of
thousands of events and will not stay fast for ever. The shape is a snapshot
carrying the highest sequence it includes, plus the tail replayed over it --
disposable, never a second source of truth, and a full-log rebuild always
permitted. Not built, because nothing yet opens a record large enough to want
it.
