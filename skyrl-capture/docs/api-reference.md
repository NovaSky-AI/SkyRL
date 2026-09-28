# API reference

The control plane is a `/v1` JSON API. The data plane is the
trajectory-specific proxy URL returned by trajectory creation.

Interactive docs are served at `/v1/docs`, and the OpenAPI document at
`/v1/openapi.json`.

## Authentication

**Capture authenticates nothing, on either plane.** It is an ephemeral
in-cluster job, brought up beside the inference server it captures and torn
down with it, reachable only from inside that job's network — exactly like the
inference server it sits in front of. There is nothing to administer over the
control plane either: the upstream is chosen when the process starts.

On the data plane, the trajectory id in the route says which trajectory a
request belongs to. **It is correlation, not authorization.** Capture issues no
credential of its own, and a client that can reach this process and knows an
active trajectory id can write to it. A deployment that needs callers
authenticated puts that in front: an authenticating reverse proxy, service-mesh
identity, a network policy.

Three different things are easy to confuse, so:

| | What it is | Who owns it |
| --- | --- | --- |
| The trajectory id in a route | Correlation | Capture |
| Authenticating the caller | Optional, and outside capture | Your deployment |
| The upstream credential | Applied to every forwarded call | `UPSTREAM_API_KEY`, in memory |

Some provider SDKs refuse to construct a client with an empty key variable. Put
a placeholder there — the SDK's `Trajectory.env()` uses `unused` — or your
deployment's own ingress credential. Capture ignores it, strips it, and never
records it.

## The hierarchy

**project / run / trajectory.** A project groups work over time. A *run* is one
execution inside it — one training run, one evaluation sweep. A *trajectory* is
one attempt inside that run, naming the task it attempted (`task_id`) and the
training step that produced it (`step`).

All three are optional on a trajectory: a wrapped agent command is one trial and
belongs to no run. When a trajectory does name a run, the run is created on
first use, like the project.

`run_id`, `task_id` and `step` are indexed columns rather than annotations,
because every question a run is read with groups by them: one task across
steps, one step across tasks, the reward surface over both.

## Conventions

- `project` is a free string, created implicitly on first use. It is the
  grouping key for listing and bulk export.
- List endpoints use cursor pagination: `{"data": [...], "next_cursor": ..., "has_more": bool}`.
  Pass `cursor` to continue.
- **There is no idempotency key.** Create and finish are idempotent by what
  they are: a create by its trajectory id and a hash of its body, a finish by a
  hash of its outcome, both persisted beside the trajectory. That is stronger
  than a key table, because it survives the process that answered the first
  time. A repeat with a different body is `409`.
- The reads and the bulk exports are served by the **viewer**, which a replica
  may be started without (`--disable-viewer`). On such a replica those routes
  do not exist; create, finish, annotate and health do.
- Timestamps are RFC 3339 UTC. Durations are milliseconds as floating point.

---

## Health

### `GET /healthz`

Liveness plus capture health. Unauthenticated, so a load balancer can use it.

```json
{
  "status": "ok",
  "persistence": "ok",
  "version": "0.1.0",
  "schema_version": 4,
  "mode": "text",
  "record": "/records",
  "requests_served": 1240,
  "clock_epoch": "864df0287fec4708a912fff94872cc25",
  "capture": {"capture_errors": 0, "capture_refused": 0, "capture_enabled": true},
  "registry": {"hot_trajectories": 8, "recovered": 0, "evicted": 132},
  "commits": {
    "pending_commits": 0, "pending_high_water": 6, "capacity": 1024,
    "oldest_pending_age_s": 0.0, "commits": 1240, "refused": 0,
    "failures": 0, "unwritten_gaps": 0, "undelivered_nodes": 0,
    "last_error": null
  },
  "store": {
    "records_written": 1381, "bytes_written": 8912345, "open_journals": 8,
    "append_ms_mean": 0.41, "fsync_ms_mean": 0.33, "fsyncs": 1381,
    "recoveries": 0, "recovery_ms_mean": 0.0, "torn_tails": 0,
    "committed_records_written": 132, "committed_bytes_written": 5512300,
    "compress_ms_mean": 1.9, "commit_write_ms_mean": 2.4
  }
}
```

`mode` is the one this process was started in, and `capture` is whatever the
proxy it built counts. In token capture that is the renderer's phase timings,
the loop lag, the trace cache and the wait a response spends on durability; in
text capture it is the forward path's counters. There is no block for the mode
this process is not running -- it was never constructed.

`status` is `degraded` when an append has failed (`commits.failures`), when
capture work has been refused because the bound was reached
(`commits.refused`), or when a gap is marked in memory and not yet on disk
(`commits.unwritten_gaps`). `undelivered_nodes` beside it counts graph nodes
from lost records that are waiting for a later record to carry them, which is
how the journal stays free of dangling parents. Those are the things that can be wrong. There is no
database to be unreachable and no worker process to go missing: the serving
process is the writer, so if this answered, it is writing. A capture outage
cannot be seen from the provider response — by design — so this is where it is
visible.

`commits.pending_commits` and `commits.oldest_pending_age_s` are the pair to
watch: appends queued behind the response, and how long the oldest has been
waiting. A crash loses exactly those.

A standalone viewer answers this too, with `source: "record"`, `indexing` and
`indexed_trajectories` in place of the write-path blocks.

**It always returns 200.** `status` is for a human and a dashboard; a liveness
probe keyed on the HTTP status is therefore unaffected by a capture problem,
which is the point. A proxy that is serving correctly is not restarted because
the disk is behind.

### `GET /readyz`

Whether this replica should take traffic. Unauthenticated.

```json
{"ready": true, "proxy": "ok"}
```

Deliberately narrow: it answers for the proxy, not for capture, because a
capture failure that pulled a healthy replica out of the load balancer would be
the one thing the design forbids. Reaching this handler is what "the proxy is
up" means, so it does not report `not ready`.

There used to be one escalation — a replica short of ingestion workers asked to
be replaced. There are no worker processes to be short of, so it is gone.

### `GET /metrics`

Prometheus text exposition:

| Metric | Meaning |
| --- | --- |
| `capture_requests_total` | Requests served by the data plane |
| `capture_commits_total` | Journal appends completed |
| `capture_commits_pending` / `capture_commits_pending_high_water` | Appends queued and not yet durable, now and ever |
| `capture_commit_oldest_pending_seconds` | How long the oldest queued append has waited |
| `capture_commits_refused_total` | Capture work refused because the bound was reached. Text data loss; nothing else surfaces it |
| `capture_commit_failures_total` | Failed appends. Any increase is the disk in trouble |
| `capture_gaps_unwritten` | Gaps marked in memory that the disk has not taken yet |
| `capture_undelivered_nodes` | Graph nodes of lost records waiting for a later record to carry them |
| `capture_hot_trajectories` | Trajectories this process holds in memory |
| `capture_lazy_recoveries_total` | Trajectories adopted from a journal another process wrote |
| `capture_journal_append_ms` / `capture_journal_fsync_ms` | Mean append and fsync latency |
| `capture_recovery_ms` | Mean time to recover one trajectory |
| `capture_torn_tails_total` | Journals found ending in a torn record |

In token capture, three more:

| Metric | Meaning |
| --- | --- |
| `capture_tito_close_wait_ms` | Mean time a response waits for its exchange to become durable |
| `capture_tito_poisoned_total` | Trajectories a turn could not be attributed to |
| `capture_tito_delivery_unconfirmed_total` | Responses whose delivery was never recorded |

In text capture, one: `capture_text_errors_total`, capture failures, each of
which is a gap.

There are no worker-process metrics. The serving process is the writer, so a
count of live workers would only restate that the process is up, which `up`
already says.

---

## Upstream

There is no route describing what this process captures, because every
trajectory already carries it. Creating one returns the wire to speak and the
mode it will capture in:

```json
{"protocol": "tokens", "mode": "tokens", "base_url": "..."}
```

and reading it back gives the whole snapshot it ran against:

```json
{
  "upstream": {
    "type": "tokens",
    "url": "http://skyrl-router/generate",
    "model": "glm-5.2",
    "tokenizer": "zai-org/GLM-5.2",
    "config": {"max_model_len": 32768}
  },
  "mode": "tokens",
  "bodies": "full"
}
```

That is better than asking the process: the mode decides whether token exports
are available at all and the tokenizer decides how the tokens were produced, and
the snapshot says what *these* calls ran against rather than how the process is
configured now.

The upstream credential is never here. It is read from `UPSTREAM_API_KEY` at startup and
held in the process; it is not stored, not returned, and not written into any
capture record.

There is no way to *set* this over the API. The upstream is startup
configuration (`skyrl-capture serve --upstream-type ... --upstream-url ...`), so changing
it means relaunching the process. That is the whole reason there is no target
CRUD any more: a definition that cannot change while trajectories run against it needs
no drain and no snapshot reconciliation.

---

## Runs

### `GET /v1/runs?project=&limit=`

Newest first, each with its counts and the values its `step` filter can take:

```json
{"data": [{
  "id": "run-2026-09-17a",
  "project": "terminal-bench",
  "created_at": "2026-09-17T09:00:00Z",
  "trajectory_count": 480,
  "task_count": 60,
  "steps": [0, 1, 2, 3, 4, 5, 6, 7],
  "step_counts": {"0": 60, "1": 60}
}]}
```

**A run is derived, not stored.** Nothing creates one, nothing writes to one,
and it has no metadata of its own: it is the grouping its trajectories imply,
computed when this listing is read. Hyperparameters and a git sha go on the
trajectories, which is also where an export row carries them.

`steps` is what `GET /v1/trajectories?step=` accepts for this run, so a caller
can offer the filter without a second query. It is a set of values, not an
axis: nothing here says a task is attempted at every step.

The counts are derived at read time rather than kept as columns. A counter
maintained from the capture path is a counter that can be wrong after a
restart, and a run holds thousands of trajectories, not millions.

There is no per-run read. An entry in this listing is the whole of a run, so
one run is picked out of it.

Pass `refresh=true` to rescan the record directory before answering. That is
what a manual refresh in the viewer sends; the index also refreshes on its own.

### A run's attempts

There is no grid endpoint. A run is read as a flat listing, filtered:

```
GET /v1/trajectories?run_id=run-2026-09-17a&step=0
GET /v1/trajectories?run_id=run-2026-09-17a&task_id=task-01
```

`task_id` and `step` are filters on the same list, and a `task × step` matrix
is one way of arranging it rather than the shape of the data — nothing
guarantees a task is attempted at every step. Two attempts at the same task and
step are two rows; a task absent from a step's list was not attempted there,
which needs no rule to tell it apart from a zero.

Reward is `annotations.reward`. There is no reward column: a reward is an
annotation like any other, and it may not be a number.

### There is no run write route

`PATCH /v1/runs/{id}/metadata` is gone, with the run record it wrote to. Run
metadata now lives on the trajectories that name the run — where a listing can
filter on it and every export row already carries it, without a join.

---

## Trajectories

### `POST /v1/trajectories` → `201`

One route for exactly one trial.

```json
{
  "project": "terminal-bench",
  "run_id": "run-2026-09-17a",
  "task_id": "task-17",
  "step": 7,
  "labels": [],
  "bodies": "full",
  "trajectory_id": "rollout-step7-042"
}
```

`trajectory_id` is **required**. The SDK generates it before it sends this, so
the id exists before the trajectory does — which is what makes a create whose
response was lost safe to repeat. It becomes the first path segment of the
route and the name of this trajectory's files, so it must be 1–128 characters
of letters, digits, `.`, `_` or `-`; a malformed one is `400`. It is also the
session key sent to the inference engine, so naming it is how a caller aligns
capture's session with its own.

**Creation is idempotent by the id and a hash of this body**, both persisted
before the route is handed back. Repeating it returns the same answer, after a
restart as well as before one. Reusing the id with a different body is `409`,
and so is reusing the id of a trajectory that has already finished.

There is nothing here that changes the call the proxy makes on your behalf. A
trajectory is capture metadata: deployment and authentication come from the
capture process's configuration, and per-inference sampling and behaviour come
from the inference request. A `cache_salt` keyed on a weight version that moves
every training step rides in the request that wants it, alongside every other
sampling field.

```json
{
  "id": "rollout-step7-042",
  "base_url": "http://127.0.0.1:8080/route/rollout-step7-042/v1",
  "mode": "tokens",
  "protocol": "openai",
  "status": "created"
}
```

There is no `api_key` and no `expires_at`: capture issues no credential, and
nothing expires a trajectory on a clock. `mode` and `protocol` describe the
process this route belongs to, and are metadata rather than routing
instructions -- the process captures in one mode whatever a caller does with
them. `protocol` is what a
*client* speaks to this route: a token-capture route is `openai`, because a
caller sends it OpenAI chat completions whatever engine wire capture uses
behind it. The engine's own type is on the upstream snapshot.

`base_url` ends in `/v1` for an OpenAI-shaped route, token capture included.
For an Anthropic upstream it deliberately does not, because the Anthropic SDK
appends `/v1/messages` itself.

`bodies` is `full` (the default, and what a replay-grade trajectory
needs) or `sampled`.

### `GET /v1/trajectories?project=&run_id=&task_id=&step=&status=&limit=&cursor=`

Every level of the hierarchy is a filter. `step=0` means step zero, not "no
step given".

### `GET /v1/trajectories/{id}`

```json
{
  "id": "tr_...",
  "project": "terminal-bench",
  "run_id": "run-2026-09-17a",
  "task_id": "task-17",
  "step": 7,
  "upstream": {"type": "openai", "url": "https://api.openai.com/v1", "model": null},
  "mode": "text",
  "status": "finished",
  "command_result": "success",
  "labels": ["task-17", "success"],
  "annotations": {"rlvr_reward": 1.0, "task": "task-17"},
  "capture": {
    "exchange_count": 3,
    "node_count": 7,
    "calls_after_close": 0,
    "complete": true,
    "calls_missing": 0,
    "recovery_uncertain": false,
    "delivery_uncertain": [],
    "errors": []
  },
  "revision": 1,
  "gap_distribution": {"samples": 2, "p50_ms": 0.45, "p99_ms": 0.5, "overlapping": 0}
}
```

Lifecycle: `created` → `active` → `finalizing` → `finished`, with `poisoned`
as the terminal alternative. Nothing expires.

`capture` is what capture knows it does not know, and the three doubts are kept
apart on purpose:

| Field | Means |
| --- | --- |
| `calls_missing` / `complete` | Calls capture saw and could not write. Text capture only |
| `recovery_uncertain` | A replacement process adopted this text trajectory and cannot say whether the previous one served a turn it never wrote |
| `delivery_uncertain` | Exchange ids that are durable and exact, and whose delivery to the client cannot be vouched for. Token capture only |
| `errors` | Why, in words |

`revision` counts metadata edits. A reward that arrives after finishing
rewrites the committed record and bumps it.

`status` is the trajectory's lifecycle; `command_result` is how the wrapped
command exited (`success`, `failed`, `aborted`, `error`). They are separate
axes — a trajectory is routinely `status: finished` with
`command_result: failed`, meaning capture completed cleanly around an agent
that did not.

### `POST /v1/trajectories/{id}/finish` → `200`

```json
{
  "labels": ["task-17"],
  "annotations": {"grader": "rubric-v2", "rlvr_reward": 1.0},
  "command_result": "success",
  "format": "graph",
  "options": {}
}
```

Closes the route to new traffic, waits for the turns already in flight and
their commits, compiles the canonical record and writes it atomically, and
returns it rendered.

Both waits are bounded by `FINISH_GRACE_SECONDS`, and **reaching the bound
means not finishing**: `503`, retryable, trajectory untouched. Compiling anyway
and declaring the difference missing cannot be made correct -- a turn awaiting
its own commit is both an open turn and a pending commit, so it would count
twice; the append keeps running past the timeout, so it may land before the
journal is read; and the record would then hold the exchange and say it was
lost. A trajectory that cannot finish stays open, which nothing expires, and a
retry finishes it. After a restart the retry is trivial: the hung turn went
with the process that held it, and the journal has everything.

Retrying is safe at every step. A finish already recorded on the trajectory --
because the commit failed, or because a restart replayed it from the journal --
is not recorded again, so the same metadata is not merged twice and `revision`
counts the one correction that was actually made.

Once the record is committed the journal is deleted, and an append to it fails
from then on -- text counts a gap, TITO fails the connection.

Create, finish and metadata take a **per-trajectory lock**. Each has awaits in
the middle, and an annotation that landed after `finish` read its snapshot
would otherwise be acknowledged and then deleted with the journal.

`200`, not `202`: by the time this answers the record is on disk and the reply
was read from it.

```json
{"id": "rollout-step7-042", "status": "finished", "format": "graph", "records": [ ... ]}
```

`format` is `graph` (the default — the whole trajectory), `replay`,
`text-samples` or `token-samples`, and `options` are the same flags
[`POST /v1/exports`](#post-v1exports--202) takes. **A trajectory's own rows
need no export job**: they are rendered here from what was just committed.

- **Idempotent, by a hash of the outcome.** That hash is persisted with the
  record, so a repeat gets the same answer after a restart as well as before
  one. The format is *not* part of it — asking for the same trajectory again in
  a different format is a retry, not a conflict — but a different outcome is
  `409`.
- **Retryable failures are `503`.** The SDK retries those three times with
  backoff; nothing was decided wrongly, the disk did not take it.
- **Metadata is not sealed.** Labels and annotations given here are merged like
  any other write, and stay editable afterwards.
- **There is no reward field.** A reward is an annotation like any other, under
  whatever key you choose. It belongs to the whole trajectory, however many
  branches it has, so a branched run cannot reject it.

### There is no delete route

A record is the result of a run, and removing one is a file operation on the
record directory rather than an API call. `DELETE /v1/trajectories/{id}` is
gone: leaving it would have meant a training run could lose its own data over
HTTP, and it never deleted the file anyway.

---

## Capture data

### `GET /v1/trajectories/{id}/exchanges`

Filters: `provider`, `model`, `status`, `retry_attempt`, plus `limit` and
`cursor`.

Each exchange carries observed facts and derived attribution:

| Field | Meaning |
| --- | --- |
| `request_start_at`, `response_end_at` | Wall clock, comparable across processes |
| `duration_ms`, `ttft_ms` | From the monotonic clock |
| `gap_ms` | Signed wait from the *parent* call's response end, derived at read time. Negative means the child started before its parent finished |
| `overlapping` | This call's interval intersected another's, derived at read time from the timestamps. Independent of graph parentage |
| `chunk_count`, `stream_summary` | Streaming cadence: first chunk, last chunk, inter-chunk mean and max |
| `usage` | Prompt, completion, cached, and reasoning tokens when the provider reports them |
| `input_prefix_node_id` | The matched prefix leaf |
| `input_node_ids` | Nodes this exchange newly committed |
| `input_leaf_node_id` | The node holding its last request message |
| `output_node_id` | Its assistant node |
| `parent_output_node_id` | Set only when the prefix matched exactly through another exchange's model output — the only relationship exportable as an AIPerf fork |
| `is_duplicate_retry` | Its entire request context already existed |
| `late` | Accepted after finish was requested |

Listings also take `refresh=true`, which rescans the record directory first,
and carry `indexing` and `indexed_trajectories` beside `total`. While
`indexing` is true, `total` is `null`: a count that is about to change is
worse than no count, because a pager prints it as fact.

### `GET /v1/trajectories/{id}/graph`

```json
{
  "nodes": [{"node_id": "nd_...", "parent_node_id": null, "role": "system",
             "author": "client", "depth": 0, "message_hash": "...",
             "delta_hash": "...", "token_count": null, "sampled_start": null,
             "derivation": {"matched_prefix_messages": 0}}],
  "leaf_node_ids": ["nd_..."],
  "leaf_assistant_node_ids": ["nd_..."],
  "branch_points": [{"node_id": "nd_...", "child_count": 2, "child_ids": ["nd_...", "nd_..."]}]
}
```

`author` is `client` for a message the caller sent or `model` for a sampled
assistant message. token nodes additionally report `token_count`,
`sampled_start`, `sampled_token_count`, `has_logprobs`, `has_routed_experts`,
and `tokenizer`.

There is no route for one node's payload. The messages come with the
[`graph` export](exports.md#graph), which inlines the whole tree in one
artifact rather than one call per node. `has_payload` on a row says whether
there is a stored body at all — it is false when `--bodies sampled` skipped
it, so it reports capture completeness rather than offering a fetch.

---

## Metadata

Labels and annotations are one mutable surface on the **trajectory**. Labels are
bare string tags; annotations are a free-form key/value document. Neither is
sealed by finishing, neither keeps history, and no field is predeclared —
what a key means is the caller's business.

They live on the trajectory rather than on a node because a branched run —
sub-agents, a re-sample, a repaired turn — yields several training samples that
share one reward. Attributing that to a single node cannot express it.

### `PATCH /v1/trajectories/{id}/metadata`

```json
{
  "annotations": {"rlvr_reward": 0.9},
  "remove_annotations": ["draft_note"],
  "labels": ["success"],
  "remove_labels": ["draft"]
}
```

Returns the resulting `labels` and `annotations`. Any subset of the four fields
may be given.

- Annotations **merge** rather than replace, so two writers touching different
  keys do not clobber each other. Writing the same value again is a no-op.
- Labels are bare tags. A value like `task=task-17` is rejected — that belongs
  in an annotation.
- Nothing here modifies the captured request, response, or node delta.

**It works after finishing too.** An active trajectory takes the edit as one
more journal record; a finished one has its committed record loaded, edited,
given a new `revision` and replaced atomically. Exports already rendered are
historical snapshots and are not rewritten; later reads and exports use what
this wrote.

Nothing is coerced. A reward may be a number, a string, or `true` — it comes
back as it went in, and filtering on it is the caller's to do over the listing:

```bash
skyrl-capture list --run-id run-2026-09-17a --json \
  | jq '[.data[] | select((.annotations.reward // 0) > 0.4) | .id]'
```

---

## Exports

### `POST /v1/exports` → `202`

```json
{"run": "run-2026-09-17a", "format": "text_samples",
 "options": {"mask_abandoned": true}}
```

Bulk exports are served by the **viewer**. A replica started with
`--disable-viewer` has no `/v1/exports` at all.

Exactly one of `project`, `run` or `trajectory` is required. A project or run
export selects every trajectory whose record is **committed** at the moment the
request is accepted, and that snapshot is fixed in the job record — an active
trajectory's rows would change under the job that was reading them. A run is
the scope an RL harness wants: the rows this training run produced.

For a single trajectory there is usually nothing to schedule:
[`finish`](#post-v1trajectoriesidfinish--200) renders the same rows from the
record it just committed, in whichever format you ask for.

The job record is a file under `<record dir>/exports/jobs/`, and the artifact
is written to `<record dir>/exports/artifacts/` and served from there. There is no delivery to a second destination: capture writes
one file to the directory it was told about, and moving it elsewhere is `cp` or
whatever a deployment already uses to move files.

`token_samples` is rejected with `400` when none of the selected trajectories
was captured in `tokens` mode, rather than producing an empty artifact.

Formats: `graph`, `replay`, `text_samples`, `token_samples`. Hyphenated
spellings are accepted.

`options`:

| Option | Format | Default |
| --- | --- | --- |
| `allow_repeated_targets` | samples | `false` |
| `mask_abandoned` | samples | `false` |
| `overlong_filtering` | `token_samples` | `false` |

### `GET /v1/exports/{id}`

```json
{
  "id": "exp_...",
  "status": "ready",
  "format": "text_samples",
  "selected_trajectory_ids": ["tr_..."],
  "output_uri": "file:///records/exports/artifacts/exp_.../run-2026-09-17a-text_samples.jsonl.zst",
  "download_url": "http://127.0.0.1:8080/v1/exports/exp_.../download",
  "byte_count": 4211,
  "record_count": 3,
  "checksum": "sha256:..."
}
```

`output_uri` is where the artifact was written; `download_url` is how to fetch
it over HTTP. One copy, one way to reach it.

### `GET /v1/exports/{id}/download`

Streams the artifact.

---

## Data plane

The trajectory URL accepts the upstream's client-facing protocol unchanged.

```http
POST /route/rollout-step7-042/v1/chat/completions
Content-Type: application/json

{"model": "glm-5.2", "messages": [{"role": "user", "content": "Solve this task"}]}
```

No capture header, no request id, no custom field. A token-capture route that
dies mid-turn fails the connection, and the client's ordinary retry of the
unchanged request is the recovery — which is exactly why there is nothing here
to send.

Supported: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages,
streaming and non-streaming, plus pass-through for other paths on the same
base URL (embeddings, model listings) which are captured but not graphed.

Errors are returned in the provider's own error shape, so an unchanged SDK
parses them normally:

| Status | Meaning |
| --- | --- |
| `404` | No such trajectory, here or in the record |
| `410` | The trajectory has finished, is finishing, or was poisoned — permanent |
| `413` | Request body over `PROXY_MAX_REQUEST_BYTES` |
| `502` | The upstream could not be reached |
