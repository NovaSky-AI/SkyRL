# Exports

An export turns captured trajectories into a JSONL file. Pick a format, point
it at a project or a single trajectory, and read the result with `jq`.

```bash
skyrl-capture export --trajectory tr_123 --format graph --output trace.jsonl
skyrl-capture export --project terminal-bench --format text-samples --output dataset.jsonl
```

A project export is a concatenation of its trajectories, so the same commands and the same tools work whether you exported one trajectory or a hundred.

## Choosing a format

| You want to | Use | One line per |
| --- | --- | --- |
| Keep everything, or look at what happened | `graph` | trajectory |
| Re-issue the captured traffic preserving the relative timing information, at a scale factor | `replay` | session |
| Build a fine-tuning or distillation set from the text space | `text-samples` | root-to-leaf path |
| Train on the exact tokens the endpoint produced. This requires token-in/token-out server implementation and is by definition a tokenizer-aware layer | `token-samples` | root-to-leaf path |

`graph` is the complete record; the other three are projections of it. Pick the
one that matches what you are doing rather than the largest.

## `graph`

One line per trajectory. `nodes` is a flat array; `parent` and `children` link
it into a tree.

This trajectory asked one question and sampled the answer twice:

```json
{
  "trajectory_id": "tr_...",
  "nodes": [
    {
      "node_id": "nd_1",
      "parent": null,
      "children": ["nd_2", "nd_3"],
      "author": "client",
      "message": {"role": "user", "content": "Name one cause of the flaky auth test."}
    },
    {
      "node_id": "nd_2",
      "parent": "nd_1",
      "children": [],
      "author": "model",
      "message": {"role": "assistant", "content": "The token expiry check is inverted."}
    },
    {
      "node_id": "nd_3",
      "parent": "nd_1",
      "children": [],
      "author": "model",
      "message": {"role": "assistant", "content": "The test shares a fixture with the session test."}
    }
  ]
}
```

See [graph.md](graph.md) for more details.

## `replay`

Requests only — no responses — grouped into sessions, with enough timing info to re-issue them. A **session** is a straight run of calls without forks, so a linear trajectory is one session and a branching one is several. If you run all of the sessions concurrently respecting their timing info you basically replay the traffic.

```json
{
  "session_id": "tr_...-s0000",
  "parent_session_id": null,
  "arrival_ms": 0,
  "delay_ms": null,
  "forks": [],
  "turns": [
    {
      "delay_ms": null,
      "endpoint": "/v1/chat/completions",
      "payload": {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What does the retry decorator do?"}]
      }
    },
    {
      "delay_ms": 85,
      "endpoint": "/v1/chat/completions",
      "payload": {
        "messages": [{"role": "user", "content": "Is the backoff exponential?"}]
      }
    }
  ]
}
```

The first turn carries the full context it sent; later turns carry only what
they added. `method` is on every turn, and `forks` lists the sessions that
branch off this one.

### When to issue each call

Timing is relative, so a replay can work against a model faster or slower than the
one captured.

| Field | Start it at |
| --- | --- |
| `delay_ms` | the end of the parent session, when `parent_session_id` is set |
| `arrival_ms` | the start of the trace, when it is not |
| turn `delay_ms` | that long after the previous turn in the session |

You can run a captured replay at Nx scale by dividing all of these values by N. `arrival_ms` runs across the whole
export, so a project export puts its trajectories on one shared timeline.

### One thing to watch

If your agent fans out and then folds the results back in — three sub-agents,
then a call that combines their findings — the combining call looks like a
fourth branch of the fork, and its `delay_ms` includes however long the three
siblings took in the original run. Replayed against a much slower model, it can
go out before its inputs would be ready. This is a known source of drift from real-world traffic distribution during replay which is related to the inherent property of the system.

## `text-samples`

One row per root-to-leaf path: a complete conversation, with each message
marked trainable or not. The two-sample trajectory from `graph` above comes out
as two rows; this is the first.

```json
{
  "path_id": "tr_...-p0000",
  "model": "gpt-4o-mini",
  "tools": null,
  "abandoned": true,
  "trainable_count": 1,
  "messages": [
    {
      "author": "client",
      "trainable": false,
      "message": {"role": "user", "content": "Name one cause of the flaky auth test."}
    },
    {
      "author": "model",
      "trainable": true,
      "message": {"role": "assistant", "content": "The token expiry check is inverted."}
    }
  ]
}
```

`message` is the message as captured — role, content, and anything structured
such as `tool_calls`. The fields beside it are what capture added: who wrote
it, and whether you may train on it.

`model` and `tools` sit on the row because they shaped every prompt on the
path, and a path has exactly one of each: a run that switches either starts a
new branch, so its samples arrive as separate rows. That is what lets you
filter a dataset by model. Sampling parameters are not here — they belong to a
replay, and `replay` carries them.

Rows also carry `trajectory_id`, `node_ids`, `labels`, `annotations`,
`schema_version`, a `node_id` on each message, and `masked_reason` when nothing
in the row is trainable. You can use `annotations` for scoring the training samples. 

## `token-samples`

The same envelope with token arrays instead of messages. Requires capture in
`tokens` mode.

```json
{
  "path_id": "tr_...-p0000",
  "model": "...",
  "tools": null,
  "abandoned": false,
  "trainable_count": 12,
  "input_ids": [128000, 9906, 791, 3944, "..."],
  "loss_mask": [0, 0, 1, 1, "..."],
  "annotations": {"reward": 0.9}
}
```

`input_ids` is the whole conversation as one sequence, and `loss_mask` is the
same length: 1 where the model produced the token, 0 everywhere else. There is
no prompt/response split, because a multi-turn path has no single point where
one ends and the other begins — later user messages and tool results sit
between generations. Train on the positions the mask marks.

A message your harness repaired was produced by no model, so its positions are
always 0.

Rewards live in `annotations`, which belongs to the trajectory, so every row of
a branched trace carries the same values — one outcome for the whole tree.
There is no dedicated reward field: whatever key you annotated with is the key
you read.

`rollout_logprobs` and `rollout_expert_indices` align to the same index.
Rows also carry `labels`, `stop_reason` and `tokenizer`.

Asking for `token-samples` from text capture is an error rather than an empty
file:

```
error 400 token_samples needs capture in 'tokens' mode, and none of the 1
selected trajectories is (1 in 'text'). ... use text_samples for text capture.
```

A project export that mixes modes succeeds and exports the `tokens`
trajectories.

## What trains, and what does not

**Rows are never dropped.** Every path is always a row; `trainable` on each
message is what changes. So the row count means the same thing under every
flag, and a row that trains on nothing still shows you the branch your agent
walked away from.

A message is trainable when all of these hold:

- the model produced it (`author: model`);
- no earlier row already claimed it, unless `--allow-repeated-targets`;
- the path is not abandoned, or you did not pass `--mask-abandoned`.

| Flag | Effect | Formats |
| --- | --- | --- |
| `--allow-repeated-targets` | let one generation be a target in several rows | both sample formats |
| `--mask-abandoned` | abandoned rows train on nothing | both sample formats |
| `--overlong-filtering` | rollouts stopped by the context limit train on nothing | `token-samples` |

**Each generation trains once by default.** Branches share their ancestors: in
a four-way fan-out the orchestrator's reply is in all four rows, and counting
it four times weights one generation as if the model had produced it four
times. Pass `--allow-repeated-targets` if you want the shared prefix weighted
by how many outcomes it led to.

### What `abandoned` means

A path is abandoned when **both** of these hold:

1. its last message was produced by the model, and
2. that message has siblings — the conversation forked at its parent.

An ordinary ending satisfies neither, so it is not abandoned.

It marks *one of several endings from the same point*. It does **not** mark a
branch that was discarded: capture sees that two continuations came from one
context and has no evidence of which one your agent went on to use. Two shapes
produce it, and they behave differently:

| Shape | Which paths are abandoned |
| --- | --- |
| A repair — the model's output is corrected, and the corrected version continues | only the model's original |
| Best-of-N — one context sampled several times | all of them, including the one you kept |

That second row is worth knowing before you pass `--mask-abandoned`: on a
best-of-N trajectory it masks every row. Use it when your agent retries and you
want only the branch that carried on; it cannot pick a winner out of parallel
samples, because nothing in the capture says which one won.

**`masked_reason`** appears when `trainable_count` is 0 and says why:
`abandoned`, `repeated_target`, or `context_length`.

`node_id` and `author` are on every message regardless, so you can ignore
`trainable` and apply your own policy.

### Example: a repaired tool call

Your harness samples a malformed tool call, fixes it, and sends the fixed
version back as history. That produces two rows, and the call that actually
worked is trainable nowhere, because no model wrote it:

```
p0000  abandoned=true   trainable=0  reason=abandoned
  client  system     train=False
  client  user       train=False
  model   assistant  train=False   CALL get_weather(city="Berlin"     <- sampled, malformed
p0001  abandoned=false  trainable=1
  client  system     train=False
  client  user       train=False
  client  assistant  train=False   CALL get_weather(city="Berlin")    <- your repair
  client  user       train=False
  model   assistant  train=True    It is 18C and clear in Berlin.
```

If your agent repairs often, expect training data where the fix is invisible
and the broken output is what was sampled. Both stay in the file.

## Output and delivery

`--output` writes the file. `--compression` chooses how: `none` (default),
`gzip`, or `zst`. The default is uncompressed because you usually want to read
the file next; the stored artifact stays compressed either way.

Every row carries the run, task and step that produced it, so two steps' rows
can be compared without a side table:

```json
{"trajectory_id": "tr_...", "project": "terminal-bench",
 "run_id": "run-2026-09-17a", "task_id": "task-17", "step": 7, ...}
```

## Exporting without a service

With `--record` the export runs in your shell, against a
[record directory](record.md) — no database, no server:

```bash
skyrl-capture export --record ./traces --run-id run-2026-09-17a \
  --format token-samples --output rl.jsonl
```

The four exporters are the same code either way. They take a trajectory view
and nothing else, and the record stores exactly that view, so the artifact is
byte-identical to one the service would have produced.

A service-side export keeps its job record under
`<record dir>/exports/jobs/` and its artifact under
`<record dir>/exports/artifacts/`, and serves it from
`GET /v1/exports/{id}/download`. There is no delivery to a second destination
and no signed URL: one copy, one way to reach it, and moving it somewhere else
is `cp`.

Bulk exports belong to the **viewer**. A capture replica started with
`--disable-viewer` has no `/v1/exports`; one process per record directory
serves them, or a standalone `skyrl-capture view --record` does.

Creating one waits for the record directory to have been read through at least
once. A listing may be a scan behind and still be a listing; a dataset may not,
because a training set that quietly omitted whatever the background scan had
not reached looks exactly like a short run.

## One trajectory needs no job

`finish` returns the rows. It compiles the trajectory's record, writes it, and
renders it in whichever of the four formats you ask for — so an RL harness
scoring a rollout gets its training rows in the call it was already making:

```python
result = trajectory.finish(
    annotations={"reward": 1.0},
    format="token-samples",
)
rows = result["records"]
```

The same formatters, the same rows. `POST /v1/exports` and
`skyrl-capture export` are for a run or a project, and for re-exporting later.

