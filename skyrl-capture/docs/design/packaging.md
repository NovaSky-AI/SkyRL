# Where this repo is going

Notes for someone opening this repository for the first time. What it is, why
it is shaped this way, what we intend to ship as open source, and what Anyscale
adds on top. Written 2026-09-17. Direction, not commitment.

The repo is currently named `skyrl-capture` and is an internal prototype.
The open-source name we are settling on is **`skyrl-capture`**.

---

## 1. What this is

An inference proxy that records exactly what your model saw and said.

Point any agent or application at it, unchanged, and every model call is
recorded: the request and response as they went over the wire, the timings, and
— in tokens mode — the exact token IDs the engine received and returned, with
per-token logprobs and the branch structure of the conversation.

The invariant everything here serves:

> For every committed turn, the tokens stored along the path from the root to
> the assistant node equal exactly `prompt_token_ids + completion_ids` — the
> sequence the engine received and returned. Not a reconstruction of it.

If that ever fails, an export's `input_ids` are not what the model saw, and
training on them is training on a reconstruction.

## 2. Why it exists

SkyRL is a full-stack library for continual learning — SFT, RL, Tinker.
Capture is the piece that comes before all of them, because continual learning
has two jobs and both start with a faithful record:

1. **Teach a model something it could not do before.** Frontier capability,
   which needs RL on real trajectories.
2. **Make inference cheaper.** Distil a smaller model with SFT on the traffic it
   will actually serve. RL a model on a niche task so it is better and faster
   there. Collect draft data for speculative decoding.

You cannot distil on traffic you did not record, and you cannot do RL on tokens
you only approximately have. Capture is the prerequisite, not the accessory.

### The one idea

> **Capture belongs in front of your inference, not inside your trainer.**

This is the whole architecture in one line, and it is the difference from the
closest comparable system. Miles (LMSYS) solves token alignment with a session
server whose lifetime is one optimizer step: it hands the trainer its samples,
the session is deleted, the tokens are gone. That is a correctness shim, and
federating it across rollout workers is fine because nothing needs to be queried
afterwards.

We treat persistence and inspection as the product, not an afterthought — which
means the record has to outlive the job, which means capture cannot live inside
the training loop. It also means we can capture *production* traffic, which a
trainer-internal session server structurally cannot.

`docs/design/vs_miles.md` has the full mechanism comparison.

## 3. The shape of the system

Two halves that touch only through storage.

```
WRITE PATH   in the job, dies with it
  agent ──HTTP──▶ capture proxy ──▶ database + blob storage
                        │
                        └──▶ inference engine

READ PATH    separate, whenever, outlives everything
  viewer / exporter ◀── database + blob storage
```

Because the halves are joined only by the stored record, the viewer and the
exporters do not care how the data got there, and the capture service does not
care who reads it later.

### Navigating the code

| Directory | What lives there |
| --- | --- |
| `data_plane/` | The shared inbound ASGI boundary. Raw ASGI, no framework on the request path. |
| `text/`, `tito/` | The mutually exclusive process modes: transparent text forwarding or token-exact inference. |
| `transport/` | Shared HTTP transport, header policy, and ASGI response mechanics. |
| `writer/` | The observed exchange, its event derivation, and the one commit boundary. |
| `domain/`, `persistence/`, `reader/` | The events and the state they project to; the sink that writes them; the readers that answer over them. |
| `export/` | `view.py` materialises a trajectory; `formats.py` turns it into the four export formats. |
| `control_plane/`, `ui/` | The API and the viewer. |
| `upstream/` | Provider wires and the plugin registry. |
| `sdk.py`, `service.py`, `cli/` | How you embed it and how you run it. |

The seam worth knowing about on day one: every exporter takes a
`TrajectoryView`, and `load_view()` is the only function in that path that
touches SQL. That is what lets the same exports and the same viewer work
against a directory of files or against Postgres.

## 4. Two modes

A service speaks one upstream wire, and **the wire determines the mode**. You
name it with `upstream_type`:

| `upstream_type` | Wire | Mode | Needs a tokenizer |
| --- | --- | --- | --- |
| `openai` *(default)* | `/v1/chat/completions` | text | no |
| `anthropic` | `/v1/messages` | text | no |
| `tokens` | token-in/token-out, OpenAI-shaped | tokens | yes |
| `skyrl` | the SkyRL router's `/skyrl/v1/generate` — registered by SkyRL's plugin, not built in | tokens | yes |

Name it explicitly whenever it is not `openai`. Getting it wrong between the two
token wires is quiet rather than loud: a `tokens` upstream pointed at the SkyRL
router gets the path, the singular request shape, the `X-Session-ID` affinity
header, vLLM's sampling-parameter rules, and the `logprobs` count-versus-boolean
difference all wrong. `openai` is a safe default only because *its* mistakes
fail immediately with a 4xx.

Several models can share one service — same URL, same credential, `model` stays
a per-request field as OpenAI already treats it. What they cannot do is mix
modes: a judge or summariser reached through a token-in/token-out router is
rendered and captured in tokens mode like everything else, which costs a
renderer pool per distinct model but is otherwise fine. Genuinely mixed modes
against one upstream would be a per-model mode map — a deliberate feature, not
something the design gives away.

| | **text mode** | **tokens mode** |
| --- | --- | --- |
| What it records | Requests and responses as they went over the wire, with timings | Exact token IDs, per-token logprobs, sampling masks, routed experts |
| Who builds the graph | The ingest worker, asynchronously | The proxy, synchronously, before responding |
| Prefix identity | Message hash | Exact token IDs |
| On capture failure | Fail-open: the response is never held up | Fail-closed: commit before responding, poison and stop on failure |
| Needs a renderer | No | Yes |
| Extra dependencies | None | `renderers`, `transformers` |

Tokens mode is fail-closed on purpose. In text mode the proxy is *observing*
someone else's inference, so a capture failure must never break a working
response. In tokens mode the proxy *produced* the tokens, so a response whose
tokens could not be attributed exactly would be silent training-data corruption.
`docs/design/token-capture-parity.md` argues this at length.

### Which mode serves which job

| Job | Mode | Why |
| --- | --- | --- |
| RL | tokens | Needs the exact sequence the engine saw. Nothing else will do. |
| SFT / distillation | text is enough | You retokenize with the student's tokenizer anyway. |
| Speculative decoding | text, if the provider returns logprobs | They land in the envelope verbatim. |
| Load-testing a candidate | text | That is the `replay` export. |

## 5. How work is organised

Four levels, borrowed from what people already expect of an experiment tracker.

```
project                 a long-lived name          "swe-bench-qwen3"
└── run                 one job, sweep, or window
    └── trajectory      one episode, one attempt
        └── graph       messages, branches, tokens
```

Two fields cut across every trajectory and are first-class columns, not tags,
because they are what you filter and sort on constantly:

- **`task_id`** — the stable identity of the problem, *across* steps.
- **`step`** — which training iteration produced this attempt.

Plus `labels` (bare tags) and `annotations` (free key/value — rewards live
here), both already in the schema and both indexed.

The two queries that matter follow directly:

- `WHERE task_id = ? ORDER BY step` — is the model getting better at this problem?
- `WHERE step = ?` — what did iteration N look like?

Which gives the view worth building first: a **task × step grid**, rows are
tasks, columns are training steps, each cell is reward and status, clicking a
cell opens that attempt's token graph.

> **[SUPERSEDED]** The grid was built and then removed. It is a dense view of
> sparse data: nothing guarantees a task is attempted at every step, so the
> matrix degrades into empty cells, and putting cells side by side invites a
> comparison that has no answer when the attempts do not recur. What replaced
> it is the line above it — `WHERE step = ?` on a flat listing, with `task_id`
> one filter among whatever else a harness annotates.
> See [run-dimensions.md](run-dimensions.md).

The levels read loosely for production capture, and should be designed to:

| | Training | Production |
| --- | --- | --- |
| `project` | the experiment | the application |
| `run` | one job | a deployment version, or a day |
| `task_id` | the problem | your app's session id |
| `step` | training iteration | unset |
| `labels` | whatever | model version, region, customer tier |

The grid becomes task × time and the same viewer works.

---

# What an open-source user chooses

There is exactly one decision: **where the record goes.** Everything else —
the proxy, the graph, both modes, all four export formats, the viewer — is the
same code either way.

```bash
pip install skyrl-capture            # text mode: capture, replay, SFT data
pip install skyrl-capture[tokens]    # adds renderers + transformers for RL
```

The light install is not a crippled version. It is the whole production-capture
product, and it has no opinion about your `transformers` pin.

## Option A — files

Nothing to provision. The record is a directory.

```python
from skyrl_capture import CaptureService

capture = CaptureService(
    upstream="https://api.openai.com/v1",   # or your router
    upstream_type="openai",                 # openai | anthropic | tokens | skyrl
    api_key_env="OPENAI_API_KEY",
    project="my-app",                       # this something like experiment name, etc.
    run="2026-09-17",                       # if not specified it will be auto-generated like wandb
    db={"path": "file://./traces"},    # it can have things like auth etc.
    storage="file://./traces",         # defaults to the db root
)
capture.start()                             # makes SKYRL_CAPTURE_ENDPOINT reachable
```

On disk:

```
traces/
  <project>/
    <run>/
      run.json                      model, upstream, config, started/ended
      index.jsonl                   one append-only line per finished trajectory
      trajectories/<traj_id>.json   graph + step, task_id, labels, annotations
      blobs/<key>
```

I can view a nice hosted UI by doing `skyrl-capture view --record ./traces`. 

`index.jsonl` is what makes the viewer fast without a database: one line
carrying `(trajectory_id, task_id, step, labels, reward, node_count, status)`.
The list view and the grid read only the index; a trajectory file is opened when
you click it. Append-only with a single writer per run means no locking.

Two consequences, both chosen deliberately:

- **Retention is `rm -rf <run>`.** A run is one directory and deleting it cannot
  orphan anything.
- **Blobs are per-run, so there is no cross-run prefix dedup.** You trade some
  storage efficiency for atomic deletion.

Where it runs out: listing is a directory scan and opening a trajectory is one
file read, which is fine into the thousands. What you do not get is
cross-trajectory queries, concurrent writers from several drivers, retention
policies, or anything resembling a join.

## Option B — your own PostgreSQL

Two strings change. Nothing else does.

```python
capture = CaptureService(
    upstream=router_url,
    upstream_type="skyrl",
    project="swe-bench-qwen3",
    run=job_id,
    db={"path": "postgres://user@host/traces"},
    storage="s3://my-bucket/traces",
)
capture.start()
```

Now the record outlives the job, several drivers can write to one database, and
the viewer can answer questions across runs. You own migrations, backups,
retention, and scaling.

## Using it: text mode

The driver starts the service once. Every worker uses the same three lines
regardless of which option you picked.

```python
from skyrl_capture import CaptureClient

client = CaptureClient()                    # CAPTURE_ENDPOINT + token from env
with client.trajectory(task_id="session-9f2", labels=["prod", "us-west"]) as t:
    run_my_app(base_url=t.base_url, api_key=t.api_key)
```

For an application you cannot pass arguments to, the environment is set for you:

```python
with client.trajectory(task_id="session-9f2") as t, t.environment():
    subprocess.run(["python", "my_app.py"])   # OPENAI_BASE_URL / OPENAI_API_KEY
```

Then, later:

```bash
skyrl-capture view --record ./traces                              # or --db postgres://…
skyrl-capture export ./traces --format replay -o replay.jsonl
skyrl-capture export ./traces --format text-samples -o sft.jsonl
```

And the thirty-second version, for a laptop:

```bash
skyrl-capture run --upstream https://api.openai.com/v1 -- python my_app.py
```

That one is a developer affordance, not a topology. It writes to `./traces` on
whichever machine ran it, which is perfect for one box and useless for a
cluster.

## Using it: tokens mode

Same shape, plus the model and the renderer.

```python
capture = CaptureService(
    upstream=router_url,
    upstream_type="skyrl",              # the router's wire
    model="Qwen/Qwen3-8B",              # tokenizer inferred; policy model for this run
    max_model_len=32768,
    project="swe-bench-qwen3",
    run=job_id,
    db={"path": "file://./traces"},
)
capture.start()
```

In a rollout worker, the only addition is `step`:

```python
with client.trajectory(task_id=task.id, step=global_step) as t:
    await agent.run(api_base=t.base_url, api_key=t.api_key)
```

The agent runs unmodified, in text space. The proxy renders the prompt, calls
the engine with token IDs, parses the sampled IDs back into a message, and
commits the turn to the graph before responding. Then:

```bash
skyrl-capture export ./traces --format token-samples -o train.jsonl
```

which carries `input_ids`, `loss_mask`, `rollout_logprobs`, and
`rollout_expert_indices`, plus the trajectory and branch identifiers.

In SkyRL this should be a config block rather than code:

```yaml
capture:
  enabled: true
  db: file://./traces
```

## What you get either way

Both modes, the branching graph, all four export formats (`graph`, `replay`,
`text-samples`, `token-samples`), the CLI, the SDK, the viewer, the upstream
plugin registry, and the test suite. The viewer is open source and works against
both backends: a dashboard you cannot point at your own data would make the
whole thing feel like a demo.

---

# What Anyscale adds

Not features. **Scope, durability, and zero setup.**

If you run on Anyscale, the two storage arguments disappear:

```python
capture = CaptureService(
    upstream=router_url,
    upstream_type="skyrl",
    model="Qwen/Qwen3-8B",
    max_model_len=32768,
)
capture.start()
```

The environment already knows where the record goes. Concretely:

| | What it is |
| --- | --- |
| **A provisioned database** | One per workspace, in the customer's own cloud. We run migrations, backups, retention, and scaling. Connection details are injected into the job. |
| **Injected provenance** | Job ID, cluster, image, and user are tagged onto every trajectory automatically. "Which job, on which image, produced the tokens that trained this checkpoint" answers itself — and cannot be faked or reconstructed afterwards. |
| **An always-on viewer** | Already running, already pointed at your workspace, already authenticated. Nothing to start and nothing to expose. |
| **Everything, not just this run** | Cross-run search and comparison, regression detection between steps and between runs, dataset lineage from an export back to the trajectories that produced it. |
| **Team scope** | RBAC, SSO, audit, and spend attribution — the same governance surface Anyscale already sells. |

The tokens never leave the customer's network: managed capture is an Anyscale
Service running in their own cloud. Routing prompts and completions out to a
SaaS to reach an engine sitting inside the VPC would be wrong on latency, egress
cost, and governance simultaneously.

Nothing here is withheld from the open-source build. A team that wants to run
Postgres and the viewer themselves can, forever. What they cannot easily get is
the part that is genuinely operational: a database they did not set up,
provenance they did not have to remember to record, and one place that holds
every run the team has ever done.

---

# What needs to be done

## Open source

1. **Drop target multi-tenancy.** One service, one upstream. This removes the
   `targets` table, `store/targets.py`, the target CRUD in `control/app.py`, and
   row-level `tenant` throughout — roughly 250 `tenant` and 270 `target`
   references. The target is not deleted so much as **promoted into the run**:
   upstream, model, tokenizer, and `max_model_len` are facts about a run.
   Isolation moves from a column to a database boundary.
2. **The file-backed record.** A `file://` db with the layout above, plus a
   second `TrajectoryView` constructor that reads it. Because `load_view()` is
   the only SQL in the export path, this makes all four formats and the viewer
   work off a directory. Do **not** port the store layer to SQLite — the SQL is
   meaningfully PostgreSQL-specific and two dialects would diverge exactly where
   this system is fragile.
3. **Hold the graph in the writing process.** Tokens mode already does: the
   trace is complete in memory when the turn commits. Text mode needs the same —
   a per-trajectory `GraphIndex` held in the worker and flushed into the record
   on finish, rather than `load_index` against SQL. Two builders, one
   persistence story. This should not add any overhead to the request path, because the graph is built asynchronously.
4. **Version the file format.** Three things read it — the viewer, the
   exporters, and eventually the upload path into a managed database — so it is
   a public format, not a convenience dump. It needs partitioning by trajectory,
   a per-run index, and a version header. `SCHEMA_VERSION` and
   `SPOOL_FORMAT_VERSION` are the precedent.
5. **The hierarchy.** A `runs` table and record; `run_id`, `task_id`, and `step`
   as first-class indexed fields on a trajectory.
6. **Collapse the setup API.** `start_capture()` plus `ensure_target()` becomes
   one constructor. `CaptureService.start()` publishes `CAPTURE_ENDPOINT`;
   `CaptureClient()` reads it. No URL plumbing in user code.
7. **The viewer.** Trajectory tree, token inspection, and the task × step grid,
   against both backends.
8. **Split the install.** `skyrl-capture` and `skyrl-capture[tokens]`.
9. **Rename and publish.** `skyrl-capture` → `skyrl-capture`, as its own
   package in the SkyRL family, alongside `skyrl-train`, `skyrl-gym`, and
   `skyrl-tx`.
10. **Publish numbers.** Per-turn overhead, prefix-reuse rate, and a
    reproducible parity benchmark against Miles. The harness exists; the
    published results do not.
11. **A runtime comparator, and a GPU verification layer** — the two things
    `vs_miles.md` concludes are worth borrowing.

Items 1–6 are the substance. Item 10 is what makes the argument land publicly.

## Anyscale, later

1. **Provision a database per workspace** — per workspace rather than per user,
   because teams are the buying unit and per-user databases fragment exactly the
   cross-run view being sold.
2. **Short-lived, append-only credentials.** A raw DSN in every job environment
   is a long-lived fully-privileged secret. Issue scoped credentials that can
   `INSERT` and `SELECT` but not `UPDATE` or `DELETE`. Append-only is the right
   semantics for a system of record anyway, so the security fix and the data
   model agree. Only the driver needs it; workers speak HTTP to the proxy and
   never touch storage.
3. **Schema negotiation.** With a bare connection string there is no server in
   the middle. The client must read `schema_migrations` on connect and refuse
   clearly on mismatch; the managed side owns running migrations.
4. **Quotas and accounting.** One runaway job with long contexts fills a
   database. Per-run byte accounting at minimum.
5. **The hosted viewer**, pointed at the workspace database, inside the existing
   Anyscale console rather than as a second dashboard to log into.
6. **Environment injection** — storage connection details plus job, cluster,
   image, and user identity.
7. **Cross-run surfaces**: search, step-over-step regression detection, and
   lineage from an export artifact back to the trajectories that produced it.

---

# Non-goals

- **This is not LLM observability.** We capture for training. The bar is
  fidelity and replayability, not dashboards, eval scores, or alerting. There
  are a dozen good tools for that and none of them can hand a trainer exact
  tokens.
- **Not a trainer.** Exports are files; anything can read them.
- **Not tied to SkyRL.** It ships in the SkyRL family and SkyRL is the reference
  integration, but `upstream/plugins.py` means another trainer's wire is that
  project's to register. `skyrl-gym` and `skyrl-tx` already work this way.

# Open questions

- **Licence.** Permissive to maximise adoption, or source-available to stop a
  competitor hosting it? Leaning permissive: it matches Ray and SkyRL, and a
  re-hoster would still have to build the history and governance from scratch.
- **Does speculative-decoding data change the token wire?** Today we store
  selected-token logprobs only; `top_logprobs` is accepted and returned empty.
  Draft-model training generally wants the target's distribution. This blocks
  freezing the record format, so it needs an answer early.
- **Does `file://` ever mean object storage?** Several drivers writing records
  to one bucket would cover much of the multi-run case without Postgres. It is
  also how you reinvent a database badly. Start local-only.
- **Credential bootstrap for workers in the files option.** Trial workers on
  other machines need to reach the driver's proxy. The per-trajectory lease is
  the right primitive; how the worker gets its first token needs an answer that
  is not "it is a private network."
- **Who builds this, and by when.** Items 1–10 above are all open source and all
  adoption. There is no platform business until there are runs to manage.
