# skyrl-capture

Trajectory-scoped inference capture, in the SkyRL family alongside
`skyrl-train`, `skyrl-gym` and `skyrl-tx`. Point an existing LLM workload at a
per-trial base URL, run it unchanged, and get back a lossless record of every
model call: a chronological timeline with request, first-token,
and response timings, a context graph showing shared prefixes and branch
points, and deterministic exports for replay and benchmarking.

The workload does not change. No SDK patching, no tracing headers, no
per-request instrumentation — the route identity is the correlation key.

```python
from skyrl_capture.sdk import capture

with capture(project="terminal-bench", labels=["task-17"]) as trajectory:
    run_client()                          # existing application, untouched
    trajectory.annotate(rlvr_reward=1.0)  # the reward reaches the export
```

Or wrap any command:

```bash
skyrl-capture run --project terminal-bench --tag task-17 \
  -- uv run harbor run --dataset terminal-bench@2.0 --agent terminus-2
```

One capture process captures **one inference server**, named when it starts.
There is no target to register, no sink to declare, and no control key to
present: creating a trajectory is the only setup call there is.

## What it gives you

| | |
| --- | --- |
| **Timeline** | Per-call latency, time to first token, streaming chunk cadence, retries, errors, and the signed wait before each call — measured from the call it continued from, not from whichever call arrived before it. |
| **Context graph** | One node per message, one parent per node. A shared prefix is stored once — where shared means the same messages under the same tools and model — a node with more than one child is a fork, and `author` records whether the model produced a message or a client replayed it. |
| **Metadata** | Labels (bare tags) and annotations (free-form key/value) on the trajectory, mutable for its life. One reward covers a whole branched tree, which is what a fan-out of sub-agents needs. |
| **Hierarchy** | project / run / trajectory, with the task each attempt attempted and the step that produced it. Indexed columns, so the task × step grid is one query. |
| **Exports** | Four formats, one per use case: `graph` (lossless tree), `replay` (traffic at a scale factor), `text-samples` (distillation), `token-samples` (RL on exact tokens). Deterministic, versioned, and every trajectory carries its own capture-integrity block. Finishing a trajectory returns its rows directly, in whichever format you ask for. |
| **Durability** | Every trajectory is a file, written as it runs. A crash loses at most the appends text capture had queued; a token response does not close until its exact exchange is on disk. What capture cannot vouch for, it says. |
| **Tokens mode** | The proxy owns the renderer and calls a token-in/token-out endpoint, storing the exact prompt and completion token IDs, per-token sampling masks, logprobs, and MoE routed experts. |

## Quickstart

Needs Python 3.11+. Nothing to set up first: there is no database and no
schema. `--record-dir` is where the run goes, and it is required — every
trajectory is persisted as it runs, so a process with nowhere to write refuses
to start.

```bash
pip install skyrl-capture            # text mode: no ML dependency at all
pip install "skyrl-capture[tokens]"  # adds the renderer and a tokenizer

UPSTREAM_API_KEY=$OPENAI_API_KEY skyrl-capture serve --record-dir ./traces \
  --upstream-type openai --upstream-url https://api.openai.com/v1
```

The split matters because capture installs *beside* a trainer, in the same
environment, and a trainer pins its own `transformers`. Text-mode capture
tokenizes nothing, so the base install carries no ML dependency; only token
mode needs one.

Then, in another shell:

```bash
skyrl-capture run --project my-project -- python my_agent.py
skyrl-capture list --project my-project
skyrl-capture view
```

The API is at `/v1` and health at `/healthz`. The [viewer](docs/viewer.md) is
a separate Node program — `skyrl-capture view` — and because it only ever
reads `/v1`, the same page serves a run in progress and a finished
[record directory](docs/record.md):

```bash
skyrl-capture view --api http://127.0.0.1:8080   # live
skyrl-capture view --record ./traces             # offline, no database
```

Full walkthrough: **[docs/quickstart.md](docs/quickstart.md)** — it runs
end to end against a bundled mock provider, so it needs no API key.

## Documentation

**Start here**

| Document | What it covers |
| --- | --- |
| [quickstart.md](docs/quickstart.md) | Working end to end in a few minutes, no API key needed |
| [examples/agents/](examples/agents/) | Runnable examples: one per graph behaviour, linked from [graph.md](docs/graph.md), plus [tokens.py](examples/agents/tokens.py) for token capture end to end |

**What the system does** — the behaviour you build against

| Document | What it covers |
| --- | --- |
| [graph.md](docs/graph.md) | How the context graph is derived, what counts as a branch, and what is deliberately not inferred |
| [exports.md](docs/exports.md) | The four export formats, their schemas, and what each is for |
| [record.md](docs/record.md) | The file-backed record: a run as a directory, read with no database and no server |
| [viewer.md](docs/viewer.md) | The viewer: the task × step grid, the trajectory tree, and the loss mask over decoded text |
| [verification.md](docs/verification.md) | Two layers: the classified prefix audit, and re-feeding a record's prompts to the engine |
| [tokens.md](docs/tokens.md) | Token-in/token-out capture: exact token IDs, sampling masks, logprobs |

**How it is built** — implementation and operation

| Document | What it covers |
| --- | --- |
| [architecture.md](docs/architecture.md) | How the data plane, the write path, and the record fit together, and why |
| [operations.md](docs/operations.md) | Deployment, scaling, tuning, backup |
| [benchmarks.md](docs/benchmarks.md) | Measured overhead, methodology, and how to run it yourself |
| [design/benchmarks-rework.md](docs/design/benchmarks-rework.md) | The write-path rework measured before and after, and why the harness's own ratio is noise at small sizes |
| [design/record-format.md](docs/design/record-format.md) | The record's layout and byte format, as a cross-language contract |
| [design/durability.md](docs/design/durability.md) | Per-trajectory persistence: the design this is built to |
| [design/token-capture-parity.md](docs/design/token-capture-parity.md) | 14 token capture requirements the PRD omits, and where each lives |
| [design/tokens_log.md](docs/design/tokens_log.md) | Token capture: what was tested for correctness, and what it costs at a million tokens |
| [design/skyrl-integration.md](docs/design/skyrl-integration.md) | Running an RL harness unmodified against the token proxy |
| [LOG.md](docs/LOG.md) | Chronological build log: what was built, what broke, what was decided |

**Reference** — the surfaces, enumerated

| Document | What it covers |
| --- | --- |
| [api-reference.md](docs/api-reference.md) | The complete `/v1` control-plane surface |
| [cli-reference.md](docs/cli-reference.md) | Every `skyrl-capture` command |
| [sdk.md](docs/sdk.md) | The Python SDK |

## How it works

```
workload ──► /route/{trajectory_id}/v1/...  ──► upstream provider
                     │
                     │  raw ASGI data plane: four clock reads and a list append
                     │  on the way out. No JSON parsing, no database, no
                     │  framework, no disk wait on the request path.
                     ▼
        after the reply: parse it, plan the graph change, commit it
                     │
                     ▼
        the trajectory's own aggregate — at once, so the next turn matches
                     │
                     ▼
        record/active/<shard>/<id>.capture — behind a bounded queue
                     │  finish: compile once, rename atomically
                     ▼
        record/committed/<shard>/<id>.json.zst — viewer-ready, read by anyone
        record/committed/<shard>/<id>.head.json — the listing, plain JSON
```

Capture is asynchronous and fail-open: if the upstream answered, the workload
gets that answer, even when capture cannot keep up. Missing calls are counted
and surfaced on `/healthz`, on the trajectory's record, and in the `integrity`
block each trajectory carries into an export — so incomplete capture is visible
rather than silent.

Tokens mode is the deliberate exception. There the proxy *produced* the tokens, so
a turn whose tokens cannot be attributed exactly fails instead of returning a
response that would enter training as if it were exact. See
[design/token-capture-parity.md](docs/design/token-capture-parity.md).

## Measured overhead

Measured by replaying **391 real Claude Code sessions** — SemiAnalysis's AgentX
traces, median context ~100k tokens, 111,026 requests — through the proxy on a
10-core laptop:

| | **Text capture** | **Token capture** |
| --- | --- | --- |
| Cost at 96 concurrent sessions | **none measurable** | **4% of throughput** |
| Cost at 192 sessions | **none measurable** | 15% |
| Ceiling, one process | **none found** (to 7.4M input tok/s) | ~75 turns/s, ~3.9M input tok/s |
| What degrades first | — | the tail: p99 7.2 s → 37 s past saturation |
| Scaling out | not needed here | a 2nd replica is worth 1.6x once proxy-bound |
| Routing | the load balancer must hash the trajectory id | same |

Above one replica the load balancer has to pin a trajectory to one of them, in
both modes — see
[operations.md](docs/operations.md#routing-pinning-a-trajectory-to-a-replica).

**Text capture is free** at every level this harness can reach. **Token capture
is free to ~100 concurrent sessions and cheap to ~190** — per-turn work is
O(delta), not O(context), so what costs is a cold start (a session's first turn,
or one whose cached trace was evicted) rather than a long conversation.

The capture path itself costs about **6.5 µs of CPU per request**, measured
directly. [docs/benchmarks.md](docs/benchmarks.md) has the full sweep, the
process ablation, and what the numbers still do not cover.

Two things the benchmark work found and fixed, both worth knowing:

- The forward path, not capture, was the throughput limit. With `httpx` the
  proxy reached 27% of the no-proxy ceiling; the lean HTTP/1.1 transport, now
  the default, reaches **88%** — 2388 rps against a 2715 rps ceiling, p99
  3.9 ms instead of 29.6 ms. On concurrent token traffic the gap is 10×, where
  httpx collapses rather than plateaus.
- Deriving a trajectory rewrote its whole timing on every batch, capping one
  worker at 17 exchanges/sec. That work is O(one exchange) now, it happens in
  the serving process after the response has gone out, and the queue in front
  of the disk absorbs what is left: at 950 rps nothing was dropped.

See [benchmarks.md](docs/benchmarks.md) for methodology and caveats, and for
why criterion 13's 10k-rps run needs real hardware rather than a claim.

## Development

```bash
uv sync --all-extras
uv run pytest                   # real sockets, no network, no API keys
uv run ruff check src tests
```

A fresh checkout is one `pytest` away from a green run: there is nothing to
provision, and nothing a run can touch outside its own temporary directories.

## Status

All four PRD delivery phases are implemented. 12 of the 13 acceptance criteria
are met and covered by tests; the thirteenth — a sustained 10,000-rps
comparison — has its harness, methodology, and thresholds implemented, and
needs real multi-node hardware to execute. [docs/LOG.md](docs/LOG.md) maps
every criterion to the tests that cover it.

**311 tests**, run against both upstream transports (`UPSTREAM_TRANSPORT=httpx` and `lean`).
