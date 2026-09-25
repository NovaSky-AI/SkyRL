# Trajectory-Scoped SkyRL Capture

> **This is the original brief, kept as written.** Parts of it have been
> superseded deliberately and are left standing rather than edited, because a
> requirements document that quietly tracks the implementation stops being
> evidence of what was asked for. The current description of the system is
> `README.md` and `docs/`.
>
> What changed, and where to read the reasoning:
>
> | The PRD says | It is now | Why |
> | --- | --- | --- |
> | Tenant-scoped control keys, a `targets` table, target and sink CRUD | One process, one upstream, fixed at startup; no control credential | [architecture.md](docs/architecture.md), [api-reference.md](docs/api-reference.md#authentication) |
> | Node annotations with revision history | Trajectory-scoped labels and annotations, latest value only | earlier review batch; see [LOG.md](docs/LOG.md) |
> | `project` as the only grouping | project / run / trajectory, with `task_id` and `step` | [api-reference.md](docs/api-reference.md#the-hierarchy) |
> | PostgreSQL as the only record | PostgreSQL, plus a portable file-backed record | [record.md](docs/record.md) |
>
> **Scope:** Text-space capture for existing inference workloads, with a
> separate TITO mode backed by token-in/token-out inference.

## Summary

Build a trajectory-scoped inference proxy that captures what an LLM server sees
when an existing workload runs. The workload must remain unchanged at the
source-code and per-request level. A caller creates a trajectory, configures
the workload with a trajectory-specific provider base URL and credential, runs
the existing command, and finishes the trajectory with final labels and an
optional leaf reward. Node annotations may be edited post-hoc with revision
history.

The system stores lossless model request/response events and derives two views:

1. A chronological model-call timeline, including request/response latency,
   streaming timing, errors, retries, and gaps between calls.
2. A context graph showing shared message prefixes, branch points,
   context growth, and inferred relationships between model calls.

The system does not attempt to observe arbitrary tool, terminal, or human
activity. It measures the resulting wait intervals between model calls. This
keeps the product focused on inference behavior while remaining useful for
agentic workloads.

## Goals

- Capture existing OpenAI-, Anthropic-, and compatible workloads without code
  changes inside the workload and without per-request tracing headers.
- Bind every request received through a trajectory-scoped route to one
  trajectory automatically.
- Preserve the original text-space payloads sufficiently for exact replay.
- Record request start, first-byte/first-token when available, and completion
  timestamps using both wall-clock and monotonic clocks.
- Make sequential calls, concurrent calls, inferred forks, and idle gaps
  visible in a usable UI and query API.
- Represent the context as a message graph where each node stores one newly
  introduced message and its delta, while retaining the raw model exchange.
- Derive shared-prefix relationships from ordered message histories while
  retaining the evidence for every derived edge.
- Ingest asynchronously and remain horizontally scalable; capture failure must
  not fail or materially delay inference.
- Sustain at least 10,000 aggregate inference requests per second across
  concurrent trajectories with effectively zero capture-path overhead.
- Export captured data to replay and benchmarking formats, initially including
  raw payload JSONL, multi-turn JSONL, and AIPerf DAG JSONL.
- Attach immutable final labels and versioned human, model-based, or
  programmatic annotations to any message node. Numeric annotations can be used
  as step or terminal rewards; the common RLVR case annotates a leaf assistant
  node.
- Reuse the graph, timing, and annotation primitives in a later TITO capture
  path where the downstream inference endpoint receives and emits token IDs.

## Non-goals

- HTTP-client monkey-patching, language-runtime interception, or TLS MITM as a
  required integration mechanism.
- Capturing or reconstructing every tool call, terminal command, human wait,
  or internal application operation.
- Claiming that an inferred prefix edge is authoritative application causality.
- Building a general-purpose agent framework or replacing OpenTelemetry.
- Requiring OpenInference, MLflow, Langfuse, Phoenix, or a specific agent SDK.

## User experience

### Programmatic lifecycle

The preferred interface uses a trajectory handle. The existing client or
command remains unchanged; only its environment/configuration is changed.

```python
trajectory = create_trajectory(
    project="terminal-bench",
    target="openai-prod",
)

try:
    with trajectory.environment():
        run_client()  # existing application; no tracing changes
finally:
    finish_trajectory(
        trajectory,
        labels={"task": "task-17"},
    )
```

The implementation may expose an equivalent launcher for opaque commands:

```bash
trace run \
  --project terminal-bench \
  --target anthropic-prod \
  -- uv run harbor run ...
```

`project` groups trajectories for querying and export. `target` names a
runtime-managed upstream configuration. Every call to `create_trajectory`
creates a fresh, single-trial route/credential lease. The route identity,
rather than a client-supplied session header, is the primary correlation
mechanism. The lease is revoked at finish or expiry and can never be reused.
Labels supplied at finish are immutable; an idempotent retry must provide the
same values.

### Lifecycle semantics

- `created`: route and credential issued; no model event yet.
- `active`: first request accepted; events may be appended.
- `finished`: caller has ended capture; final labels and optional leaf reward
  are accepted and late events are marked late.
- `aborted`: caller ended with an error or lease expired.
- `expired`: safety timeout closed an abandoned trajectory.

Finishing must be idempotent. Each trial must receive a fresh lease; a finished,
aborted, or expired lease cannot accept a second trial.

## Integration contract

### Provider routing

The capture endpoint must accept the client protocol declared by the selected
target and forward requests to that target's configured upstream. Targets are
added and removed at runtime. At minimum support:

- OpenAI Chat Completions;
- OpenAI Responses where practical;
- Anthropic Messages;
- streaming and non-streaming responses;
- provider-specific request fields preserved in the raw envelope.

The target owns its upstream URL, type, model/rendering configuration when
needed, and encrypted upstream credential. Target secrets are never returned
after creation or written into capture records. The trajectory credential
authenticates the capture route and authorizes forwarding.

### Correlation

Every accepted exchange receives a server-generated `exchange_id`. The
trajectory route supplies `trajectory_id` automatically. If the provider
request contains a native request ID, previous-response ID, or equivalent,
preserve it as metadata rather than treating it as the primary correlation key.

No per-request session or path header is required. Capture must remain correct
without application spans or tracing headers.

## Capture data model

### Trajectory

Required fields:

- `trajectory_id`
- project name and a snapshot of the selected target identity
- creation, first-event, finish, and expiry timestamps
- destination/provider configuration (with secrets removed)
- final labels, sealed at finish
- current node annotations and their revision history
- capture status and dropped-event counters
- schema version

### Model exchange

Each exchange contains:

- `exchange_id` and `trajectory_id`;
- provider, model, endpoint kind, and request method/path;
- raw request body and response body when available;
- selected request/response headers, with an explicit redaction policy;
- protocol-parsed messages and content blocks;
- provider request/response IDs;
- wall-clock and monotonic start, first-byte/first-token, and end times;
- streaming chunk timing summary and optional chunk records;
- HTTP status, provider error, retry attempt, and completion reason;
- prompt, completion, cached, and reasoning usage when supplied by provider;
- source/runtime metadata supplied by the launcher;
- matched input-prefix node, newly committed input-node IDs, and output-node ID;
- `previous_exchange_id` only when explicitly known, never guessed silently.

The raw envelope is the replay source of truth. Parsed fields are derived and
may evolve independently.

### Graph node

Each graph node represents one message introduced into a trajectory. The raw
model exchange and its message nodes are linked but stored separately: the
exchange is the replay source of truth, while the nodes form the incremental
context representation used for graph analysis and export.

Required node fields are:

- `node_id`, `trajectory_id`, `parent_node_id`, and introducing `exchange_id`;
- one protocol-parsed message delta, including role and content blocks;
- origin: `request` for an input message or `model_output` for the sampled
  assistant message;
- message hash, insertion timestamp, and output timing when applicable;
- derivation evidence and schema version.

For the first model call, the proxy commits one node for each request message,
then one node for the assistant message returned by the model. For a later
call, it matches the longest exact message prefix, commits only the new request
message tail, then commits the new assistant message. Every root-to-leaf path
therefore reconstructs one complete text trajectory, and shared prefixes are
stored once. Two distinct messages with the same parent form a fork. Each node
has exactly one parent; joins are outside the MVP.

### Labels and node annotations

Labels describe the trial and are sealed at finish. An annotation is a mutable,
versioned value attached to a `node_id`; changing it updates the current view
while retaining prior revisions for provenance. Annotations never mutate the
captured request, response, or node delta. A single-leaf trajectory may use the
`reward` shorthand at finish; general post-hoc annotation uses a node ID and
annotation name.

Each record contains:

- stable annotation name, `node_id`, and revision number;
- `name`, scalar or structured value, optional categorical label, and
  explanation;
- source and annotator kind: `human`, `llm`, `code`, `verifier`, or `system`;
- rubric, evaluator, or reward-function version;
- creation/update timestamps, author/service identity, and provenance metadata;
- optional confidence, calibration, and privacy/redaction status.

A numeric annotation is a reward. Any message node may receive one; for RLVR,
the normal case is a verifier-produced reward on the evaluated leaf assistant
node. Updating the same `(node_id, name)` replaces its current value and creates
a new revision. Changing the target node or annotation name creates a distinct
annotation.

### Derived relations

Relations are separate records, not mutations of the raw exchanges. Each
relation includes:

- source and target node IDs;
- relation type: `prefix_extension`, `temporal_predecessor`, `explicit_parent`,
  or `concurrent_sibling`;
- evidence: common message/block hashes, provider IDs, timing overlap, or
  explicit context;
- shared-prefix length in messages/content blocks;
- wait interval in milliseconds where applicable;
- confidence and derivation version.

## Prefix and timing analysis

The first implementation uses deliberately strict text-space rules:

1. Preserve the raw request and extract its ordered message/content-block list
   without cross-provider canonicalization.
2. Hash each complete parsed message and walk the existing graph to find the
   longest exact request-message prefix.
3. Commit one node per unmatched request message, in order, then commit one
   `model_output` node for the observed assistant response.
4. If a history was compacted or rewritten, branch at the last unchanged
   message; if no message matches, branch from the trajectory's dummy root.
5. Treat an identical retry separately at the exchange layer but do not commit
   duplicate message nodes or mistake it for a branch.
6. Detect overlapping request intervals independently of graph parentage. Do
   not infer joins or uncertain causal edges in the MVP.
7. For the workload-level distribution, compute the gap between consecutive
   model exchanges as `next.request_start - previous.response_end`. Retain the
   signed observed value and clamp it to zero only when a replay format requires
   a non-negative delay.

The UI must distinguish observed facts (timestamps, payloads, provider IDs)
from derived facts (exact-prefix edges and branch labels).

## UI and query requirements

The trajectory view must provide:

- a chronological waterfall of exchanges;
- request, first-token, response, and inter-request-gap durations;
- filters for provider, model, status, retry attempt, and labels;
- expandable raw and protocol-parsed payloads;
- a context-prefix tree view with branch points and shared-prefix lengths;
- a distinction between sequential and overlapping calls;
- token/cost/cache summaries when providers report them;
- trajectory labels and node-annotation summaries with evaluator filters;
- per-node annotation/reward overlays on the timeline;
- a timeline export and a machine-readable relation export.

The first UI does not need to render arbitrary application/tool spans. It should
show the time occupied by those activities as gaps between model exchanges.

## Replay and export

Exports must be deterministic and versioned.

### Initial formats

- **Raw payload JSONL:** one complete request body per exchange, preserving
  provider-specific fields for same-provider replay.
- **AIPerf multi-turn JSONL:** linear model-exchange chains with incremental
  request messages and observed inter-turn delays.
- **AIPerf DAG JSONL:** exact context branches and observed per-turn delays,
  using the mapping below.

### AIPerf DAG mapping

The MVP populates only fields supported by direct capture evidence:

- each maximal linear exchange segment becomes one AIPerf conversation line;
- each exportable root exchange becomes a root AIPerf `session_id`;
- each model exchange becomes one AIPerf turn; message nodes do not each become
  turns;
- the root exchange's complete request-message path becomes `turn.messages`;
- a child exchange's newly committed `request` message nodes become
  `turn.messages`, because AIPerf inherits the parent history and replayed
  parent response;
- captured model, maximum output tokens, and tools map to `model`,
  `max_tokens`, and `tools`; remaining supported wire parameters go in `extra`;
- the observed non-negative wait from the parent exchange's response end to the
  child exchange's request start becomes `delay` in milliseconds;
- exchange siblings whose requests share the same parent `model_output` node
  become `forks`, with each child path assigned its own session ID.

Only a branch that matches through a parent exchange's exact `model_output`
node is exported as an AIPerf fork. Compaction branches, rewritten histories,
and unmatched roots become independent root sessions. The exporter does not
infer AIPerf `spawns` or joins in the MVP. Ambiguous relations remain available
in native capture records but are not encoded in AIPerf DAG JSONL; no sidecar
is required for the MVP export.

### TITO capture mode

TITO is a separate capture path, not retokenization performed during export.
The external client may still use a text-space OpenAI-compatible API, but the
proxy owns the model-aware renderer and calls a token-in/token-out inference
endpoint. Each exchange stores the exact prompt token IDs sent to inference and
the exact completion token IDs and logprobs returned by inference.

The message graph has the same one-node-per-message structure as text mode. A
TITO node additionally stores the exact token, sampling-mask, and logprob delta
introduced by its message. Request/environment messages normally introduce
non-sampled token deltas; the assistant `model_output` node introduces the new
sampled completion tokens and their logprobs.

On later calls, the proxy finds the longest matching message prefix, recovers
the exact token deltas already committed for that prefix, and renders only the
new message tail. Before reusing the prefix, it verifies that the concatenated
node token deltas exactly match the prompt IDs sent to inference. A mismatch
forces a full render and a new branch rather than silently mixing tokenizations.

Text and TITO modes share trajectory IDs, message nodes, graph edges,
timestamps, leases, and annotations. Text mode treats message/content deltas as
authoritative; TITO additionally treats endpoint-produced token, mask, and
logprob deltas as authoritative. Tokenizer/model identity, token timing, and
cache-block metadata are stored when available.

## Reliability and performance

- The request path must not depend synchronously on the analytics database.
- Use an in-process or sidecar bounded buffer plus durable local/object-storage
  spooling before remote ingestion.
- Backends must support horizontal ingestion and partition by trajectory and
  exchange ID.
- Retries must be idempotent using exchange IDs and an ingestion revision.
- Capture outages must return the provider response whenever forwarding
  succeeded; they must surface dropped-event counts and health status.
- Measure added p50/p95/p99 latency and throughput overhead separately for
  non-streaming and streaming requests.
- Benchmark capture-enabled operation against the same proxy and upstream with
  capture disabled. At a sustained 10,000 aggregate requests per second, the
  initial success thresholds are at least 99% of baseline throughput, no more
  than 1 ms of additional p99 proxy-processing or streaming-chunk delay, and no
  more than 0.01 percentage points of additional request errors.
- Under that load, a correctly provisioned and healthy ingestion backend must
  record every exchange without drops. Backend failure remains fail-open and is
  evaluated separately from the healthy-system overhead benchmark.
- Support configurable payload sampling only as an explicit mode; full capture
  is the default for a trajectory created for replay.

## Security and privacy

- Encrypt payloads in transit and at rest.
- Keep provider credentials out of persistent capture records by default.
- Provide field-level redaction for messages, headers, tool arguments, and
  provider metadata.
- Support tenant-scoped authorization for trajectory creation, viewing,
  finishing, export, and deletion.

## Acceptance criteria

The first production-capable milestone is complete when:

1. An unchanged OpenAI- or Anthropic-compatible workload can run using only a
   trajectory-scoped base URL and credential.
2. Every exchange in the route is assigned to exactly one trajectory and has a
   durable exchange ID.
3. A failed analytics write never changes a successful provider response.
4. Raw payloads and response metadata can reproduce a same-provider request.
5. The UI shows a correct chronological timeline and inter-request gap
   distribution for a multi-step workload.
6. A workload with two exact divergent message histories renders a visible
   prefix branch with its prefix evidence.
7. A linear trajectory exports to AIPerf multi-turn format, and exact branches
   export using AIPerf `forks` and per-turn `delay`.
8. A trajectory can be finished twice without duplicate finalization or data
   corruption.
9. Payload redaction and tenant isolation are covered by automated tests.
10. Every trial uses a fresh lease; finishing revokes it and seals its labels.
11. Node annotations are editable post-hoc, provenance-aware, and queryable;
    every change preserves a revision. A leaf assistant node can carry a
    numeric RLVR reward annotation.
12. All finished trajectories in a project can be exported together with a
    manifest identifying the selected trajectory IDs.
13. A sustained 10,000-QPS capture-on benchmark meets the throughput, p99
    latency, error-rate, and zero-drop thresholds defined under Reliability and
    performance when compared with the same proxy running capture-off.

## Delivery phases

### Phase 1: capture and replay foundation

Implement trajectory leases, provider forwarding, raw exchange envelopes,
stream timing, durable asynchronous ingestion, finish/expiry semantics, and
raw payload replay.

### Phase 2: text context graph

Implement one-message-per-node storage, exact message-prefix matching, branch
and overlap detection, wait-time distributions, and the trajectory UI.

### Phase 3: benchmark adapters

Implement AIPerf raw, multi-turn, and DAG exports; add validation fixtures and
replay schedule tests.

### Phase 4: TITO extension

Add a model-aware rendering route backed by token-in/token-out inference. Store
authoritative prompt/completion token IDs at exchange scope and token, mask,
and logprob deltas at message-node scope while reusing the same trajectory,
graph, lease, and annotation primitives.

## MVP decisions

- A fresh, single-use route/credential lease is required for every trial.
- Labels are sealed at finish; annotations are mutable, versioned, and
  node-scoped.
- Numeric node annotations represent rewards; the evaluated leaf assistant
  normally carries the RLVR signal.
- Text graph edges and AIPerf forks require exact prefix evidence. The MVP does
  not infer joins, spawns, or ambiguous AIPerf relationships.
- Training-example schemas, provider-wide canonicalization, spool-exhaustion
  policy, and optional application-span correlation are outside this PRD.

## Appendix A: API and CLI surface

### API conventions

The control plane uses `/v1` JSON APIs authenticated by a tenant-scoped control
key. The data plane is the trajectory-specific proxy URL and credential returned
by trajectory creation. Resource payloads are intentionally shallow; optional
configuration can be added without changing the lifecycle.

`project` is initially a tenant-scoped string rather than a separately managed
resource. It is created implicitly when first used and provides the grouping
key for listing and bulk export. Target names are also unique within a tenant.

The complete MVP control-plane surface is:

| Resource | Operations |
| --- | --- |
| Targets | create, list, get, remove/drain |
| Trajectories | create, list, get, finish, delete |
| Capture data | list exchanges, get message graph |
| Annotations | put current value, list current values, list revisions |
| Exports | create project/trajectory export, get status/download |
| Dataset sinks | create, list, get, remove export destinations |

### Targets

A target describes an upstream inference service. Targets can be added and
removed while the capture service is running.

Create a text target:

```http
POST /v1/targets
Authorization: Bearer <control-key>

{
  "name": "openai-prod",
  "type": "openai",
  "url": "https://api.openai.com/v1",
  "api_key": "sk-..."
}
```

Create a TITO target:

```http
POST /v1/targets
Authorization: Bearer <control-key>

{
  "name": "glm-tito",
  "type": "tito",
  "url": "http://skyrl-router/generate",
  "model": "glm-5.2",
  "tokenizer": "zai-org/GLM-5.2"
}
```

The response omits secrets:

```json
{
  "name": "glm-tito",
  "type": "tito",
  "status": "ready"
}
```

Read and remove targets:

```http
GET    /v1/targets
GET    /v1/targets/{name}
DELETE /v1/targets/{name}
```

Deletion immediately prevents new trajectories from selecting the target.
Existing leases continue against their snapshotted target configuration. The
target reports `draining` until those leases finish or expire, after which its
stored secret can be deleted. Updating targets and forced deletion are outside
the MVP; a changed configuration uses a new target.

### Trajectory lifecycle

Create one fresh lease for one trial:

```http
POST /v1/trajectories
Authorization: Bearer <control-key>
Idempotency-Key: <caller-generated-trial-key>

{
  "project": "terminal-bench",
  "target": "glm-tito"
}
```

```json
{
  "id": "tr_123",
  "base_url": "https://capture.example.com/tr_123/v1",
  "api_key": "trk_abc",
  "expires_at": "2026-09-09T02:00:00Z"
}
```

The returned URL accepts the target's client-facing protocol. For an
OpenAI-compatible target, the unchanged workload sends requests such as:

```http
POST /tr_123/v1/chat/completions
Authorization: Bearer trk_abc

{
  "model": "glm-5.2",
  "messages": [{"role": "user", "content": "Solve this task"}]
}
```

Finish the trajectory:

```http
POST /v1/trajectories/tr_123/finish
Authorization: Bearer <control-key>
Idempotency-Key: <caller-generated-finish-key>

{
  "labels": {
    "task": "task-17",
    "outcome": "success"
  },
  "reward": 1.0
}
```

`reward` is optional shorthand for an `rlvr_reward` annotation on the only leaf
assistant node. It is rejected when the graph has multiple leaves. The finish
request closes the lease to new traffic, drains already accepted exchanges,
and returns `202 Accepted` while finalization is in progress:

```json
{
  "id": "tr_123",
  "status": "finalizing"
}
```

Trajectory reads are:

```http
GET    /v1/trajectories?project=terminal-bench
GET    /v1/trajectories/{id}
GET    /v1/trajectories/{id}/exchanges
GET    /v1/trajectories/{id}/graph
DELETE /v1/trajectories/{id}
```

The list and event endpoints use cursor pagination. `GET /graph` returns the
message nodes, their parent edges, leaf IDs, and current annotations. Deleting a
trajectory deletes its exchanges, graph, annotations, and generated exports
according to the service's retention policy.

### Mutable node annotations

An annotation is addressed by node ID and name. `PUT` creates it or changes its
current value:

```http
PUT /v1/nodes/n3/annotations/rlvr_reward
Authorization: Bearer <control-key>

{
  "value": 0.75
}
```

```json
{
  "node_id": "n3",
  "name": "rlvr_reward",
  "value": 0.75,
  "revision": 2
}
```

Every changed value creates a revision; writing the existing value is
idempotent. Actor identity and timestamps are inferred from the authenticated
request. The initial API accepts any JSON-compatible `value`; evaluator and
rubric metadata can be added later.

```http
GET /v1/nodes/{node_id}/annotations
GET /v1/nodes/{node_id}/annotations/{name}/revisions
```

### Exports

Exports are asynchronous snapshots. A project export selects all trajectories
that are finished when the request is accepted:

```http
POST /v1/exports
Authorization: Bearer <control-key>

{
  "project": "terminal-bench",
  "format": "tito_jsonl"
}
```

A single trajectory can be selected instead:

```json
{
  "trajectory": "tr_123",
  "format": "aiperf_dag_jsonl"
}
```

Exactly one of `project` or `trajectory` is required. Initial formats are
`native_jsonl`, `raw_payload_jsonl`, `tito_jsonl`,
`aiperf_multi_turn_jsonl`, and `aiperf_dag_jsonl`.

```json
{
  "id": "exp_123",
  "status": "pending"
}
```

```http
GET /v1/exports/{id}
```

When ready, the response includes a temporary download URL. Every group export
contains a manifest with project name, schema version, creation time, selected
trajectory IDs, and capture completeness/drop counters.

### CLI

Manage targets without restarting the service:

```bash
trace target add glm-tito \
  --type tito \
  --url http://skyrl-router/generate \
  --model glm-5.2 \
  --tokenizer zai-org/GLM-5.2

trace target add openai-prod \
  --type openai \
  --url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY

trace target list
trace target show glm-tito
trace target remove glm-tito
```

Run one unchanged command as one trial:

```bash
trace run \
  --project terminal-bench \
  --target glm-tito \
  --label task=task-17 \
  -- uv run harbor run --dataset terminal-bench@2.0 \
     --agent terminus-2 --model glm/GLM5.2 --n-concurrent 1
```

`trace run` creates the trajectory, injects the returned base URL and credential
using environment variables appropriate for the target's client protocol,
runs the child command with normal stdin/stdout/stderr, finishes the trajectory,
and exits with the child's exit code. It traps normal termination signals and
finishes with an aborted outcome when possible; lease expiry is the fallback.

The command may include a reward known at finish:

```bash
trace run --project terminal-bench --target glm-tito \
  --label task=task-17 --reward 1 -- python run_agent.py
```

Inspect and update captured data:

```bash
trace list --project terminal-bench
trace show tr_123
trace exchanges tr_123
trace graph tr_123
trace annotate tr_123 --leaf --name rlvr_reward --value 0.75
trace annotate tr_123 --node n3 --name critique --value "incorrect tool"
trace delete tr_123
```

Download all finished trajectories in a project:

```bash
trace export --project terminal-bench --format tito-jsonl \
  --output terminal-bench-traces.tar.gz

trace export --trajectory tr_123 --format aiperf-dag-jsonl \
  --output tr_123.jsonl
```

The export command creates the export job, polls it, and downloads the result.
`trace export status exp_123` exposes the polling operation separately.

One `trace run` invocation defines one trajectory lease and therefore must
represent one trial. If a wrapped command internally runs several trials, their
boundaries cannot be recovered from inference requests alone; the caller must
split the command or use a future multi-trial launcher.

## Appendix B: Storage and dataset sinks

### Storage model

The service separates transactional metadata from large capture payloads and
datasets:

```text
Inference proxy
    |
    +-- durable asynchronous capture queue
          +-- object store: lossless payloads and datasets
          +-- PostgreSQL: lifecycle, indexes, graph, and annotations
```

PostgreSQL is the authoritative store for control-plane state and searchable
metadata. An object store is the authoritative store for large or lossless
request, response, message, token, and export data. A dataset sink is an
optional destination for completed exports; it is never part of the inference
request path.

### Deployment profiles

| Deployment | Metadata | Payload and dataset storage | Analytics |
| --- | --- | --- | --- |
| Local self-hosted | PostgreSQL | Local filesystem | DuckDB over exported Parquet |
| HA self-hosted | PostgreSQL | S3-compatible object storage | PostgreSQL initially |
| Managed | Managed PostgreSQL | S3 or GCS | PostgreSQL initially; optional analytical store later |

The MVP supports PostgreSQL only, including for local development. It does not
add SQLite as a second transactional implementation. This keeps migrations,
locking, idempotency, and production behavior consistent across deployments.

### Local self-hosted setup

A local installation runs PostgreSQL and mounts a directory for capture
objects:

```yaml
services:
  postgres:
    image: postgres:18
    environment:
      POSTGRES_DB: traces
      POSTGRES_USER: traces
      POSTGRES_PASSWORD: traces
    volumes:
      - trace-postgres:/var/lib/postgresql/data

  capture:
    image: skyrl-capture:latest
    environment:
      DATABASE_URL: postgresql://traces:traces@postgres:5432/traces
      OBJECT_STORE_URI: file:///data/objects
    volumes:
      - trace-objects:/data/objects
```

There is no schema and no migration step: the state is a map the serving
process owns, and results land in the record directory.

For an HA self-hosted deployment, `OBJECT_STORE_URI` changes to an `s3://`
location and PostgreSQL is supplied as an external replicated service. The
capture and ingestion workers remain stateless apart from their bounded local
durable queues.

### PostgreSQL responsibilities

PostgreSQL stores small records needed for transactions, filtering, graph
navigation, and UI queries:

```text
targets
trajectories
leases
exchanges
message_nodes
annotation_current
annotation_revisions
exports
dataset_sinks
ingestion_batches
```

Important fields use ordinary typed columns, including tenant, project,
trajectory status, timestamps, target name, node parent, and object URI.
Flexible labels and annotation values use `jsonb`. Raw payload bodies and TITO
token arrays are not stored in PostgreSQL.

Representative logical records are:

```text
trajectory:
  id, tenant, project, target, status, labels, timestamps, dropped_event_count

exchange:
  id, trajectory_id, request_uri, response_uri, timing,
  input_leaf_node_id, output_leaf_node_id

message_node:
  id, trajectory_id, parent_node_id, exchange_id, role, message_hash,
  payload_uri, token_count

annotation_current:
  node_id, name, value, revision, updated_at

annotation_revision:
  node_id, name, revision, value, actor_id, created_at
```

### Object-store responsibilities

Object storage holds:

- lossless provider request and response bodies;
- optional streaming chunks;
- parsed message bodies;
- TITO token, mask, and logprob deltas;
- finalized native, raw, TITO, and AIPerf exports;
- group-export archives and manifests.

The initial representation uses compressed JSON for capture objects because it
preserves irregular provider payloads. Export jobs may materialize compressed
JSONL or Parquet. A simple object layout is:

```text
tenants/{tenant}/
  projects/{project}/
    trajectories/{trajectory_id}/
      manifest.json
      exchanges/{exchange_id}.request.json.zst
      exchanges/{exchange_id}.response.json.zst
      nodes/{node_id}.json.zst
    exports/{export_id}/
      manifest.json
      data.jsonl.zst
```

Object keys use stable generated IDs, not user-provided labels. Project and
tenant values in paths must be encoded or replaced with internal IDs to prevent
path traversal and naming collisions.

### Asynchronous write path

Capture persistence proceeds as follows:

1. The proxy forwards the upstream response to the workload.
2. It appends the capture envelope to a local durable ingestion queue.
3. A background worker uploads the request and response objects.
4. After object upload, it inserts or updates the exchange index in PostgreSQL.
5. Graph processing writes message-node metadata and node payload objects.
6. Finish closes the lease and waits for accepted records to drain before the
   trajectory becomes complete.

Every stage is idempotent by exchange, node, or export ID. Uploading objects
before committing their database pointers prevents committed rows from
referencing missing data. A database failure after upload may leave an orphan
object, which a later retry can adopt or garbage collection can remove.

### Supported dataset sinks

The initial sink types are:

| Type | URI example | Intended use |
| --- | --- | --- |
| Local filesystem | `file:///data/exports` | Local self-hosted development |
| S3-compatible | `s3://bucket/prefix` | AWS and self-hosted object storage |
| Google Cloud Storage | `gs://bucket/prefix` | Managed or self-hosted on GCP |
| Managed download | No URI | Temporary signed download from managed storage |

The deployment's primary object store always receives capture data first.
Named sinks receive completed export artifacts asynchronously. A slow or
unavailable customer sink cannot delay inference or make the primary capture
incomplete.

Direct dataset writes to PostgreSQL, ClickHouse, Snowflake, BigQuery, Kafka, or
another customer database are outside the MVP. Standard JSONL and Parquet in
object storage provide an integration point for those systems. A columnar
analytics database may later consume the canonical event stream as a derived
index, but it must not become the source of truth.

### Dataset sink API

Create a named sink:

```http
POST /v1/sinks
Authorization: Bearer <control-key>

{
  "name": "team-s3",
  "type": "s3",
  "uri": "s3://my-bucket/inference-traces"
}
```

The initial API relies on deployment credentials or workload identity for
object-store access. Explicit customer credential and role-assumption fields
can be added later without changing the sink identity.

```http
GET    /v1/sinks
GET    /v1/sinks/{name}
DELETE /v1/sinks/{name}
```

Deleting a sink prevents new export jobs from selecting it. Existing jobs may
finish; deletion does not remove artifacts already written into the external
destination.

Select a sink when creating an export:

```http
POST /v1/exports
Authorization: Bearer <control-key>

{
  "project": "terminal-bench",
  "format": "tito_jsonl",
  "sink": "team-s3"
}
```

If `sink` is omitted, the export remains in the primary object store and is
made available through the managed-download response. Export status reports
the selected trajectory snapshot, output URI, byte count, checksum, and any
sink-delivery error.

### Sink CLI

```bash
trace sink add team-s3 \
  --type s3 \
  --uri s3://my-bucket/inference-traces

trace sink list
trace sink show team-s3
trace sink remove team-s3

trace export --project terminal-bench --format tito-jsonl \
  --sink team-s3
```

For local self-hosted use, the configured `file://` primary store is sufficient;
creating a named sink is optional.

### Implementation references

- [PostgreSQL JSON types and `jsonb` indexing](https://www.postgresql.org/docs/current/datatype-json.html)
- [Apache Arrow partitioned dataset API](https://arrow.apache.org/docs/python/api/dataset.html)
- [DuckDB querying Parquet directly](https://duckdb.org/docs/current/guides/file_formats/query_parquet)
