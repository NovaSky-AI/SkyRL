# Per-Trajectory Persistence and Unified Viewer Refactor

## Summary

Replace the global event log, `LiveState` reducer, and process-wide `Recorder`
with:

- One hot `ActiveTrajectory` aggregate per locally used trajectory.
- One append-only active journal per trajectory.
- One compiled, viewer-ready `TrajectoryRecord` per finished trajectory.
- One `RecordReader` that merges active and committed records.
- A minimal lifecycle API on every proxy.
- An optional viewer/query/export API, enabled by default and removable with
  `--disable-viewer`.

Persistence is mandatory. Disk supports multiple capture processes sharing a
local directory or compatible RWX PVC, under the external invariant that
consistent-hash routing provides exactly one writer per trajectory.

The agent loop remains unchanged:

- No capture-specific inference headers.
- No custom request fields.
- No client middleware.
- Existing OpenAI/Anthropic-compatible request bodies continue to work.
- The trajectory-specific base URL remains the correlation mechanism.

There is no ownership lease, global sequence, global reducer, active-log
compaction, automatic trajectory expiration, memory-only mode, or legacy-record
compatibility.

### Product guarantees

- Recovery occurs at completed-exchange boundaries.
- Finished records are compiled artifacts, not replayed event histories.
- The viewer reads persisted state rather than private proxy memory.
- Partial streaming output is never persisted or displayed.
- TITO prioritizes exact token correctness and waits for durability before
  completing a response.
- Text prioritizes transparent pass-through and captures asynchronously.
- Finished trajectories are removed from hot proxy memory.
- Unfinished crash-orphan journals remain until resumed or explicitly finished.
- Routing, not storage, guarantees one writer per trajectory.

## Public APIs and Domain Types

### Lifecycle API

Always mount:

```http
POST  /v1/trajectories
POST  /v1/trajectories/{id}/finish
PATCH /v1/trajectories/{id}/metadata
GET   /healthz
GET   /readyz
GET   /metrics
```

Remove:

```http
DELETE /v1/trajectories/{id}
PATCH  /v1/runs/{run_id}/metadata
```

Runs become read-only groupings derived from trajectory records.

### Trajectory creation

Make `trajectory_id` required.

- The SDK generates the ID before sending the create request.
- Repeating creation with the same ID and identical body is idempotent.
- Reusing the ID with a different body returns `409`.
- Persist trajectory creation before returning success.
- Remove server-generated trajectory IDs, process-local create idempotency,
  `expires_at`, and trajectory TTL configuration.

The response continues to provide the trajectory-specific inference base URL.

### Trajectory finish

Extend the request:

```json
{
  "labels": ["scored"],
  "annotations": {"reward": 1.0},
  "command_result": "success",
  "format": "graph",
  "options": {}
}
```

- `format` defaults to `graph`.
- Supported names are `graph`, `replay`, `text-samples`, and `token-samples`.
- Reuse existing export option validation.
- Wait for in-flight requests and pending exchange commits.
- Compile and atomically persist the canonical `TrajectoryRecord`.
- Render the requested per-trajectory export from that committed record.
- Return HTTP `200`:

```json
{
  "id": "tr_...",
  "status": "finished",
  "format": "graph",
  "records": []
}
```

Persist a hash of the finish payload:

- An identical retry returns the committed trajectory rendered in the requested
  format.
- A conflicting retry returns `409`.
- A retry after a crash completes or discovers the same deterministic committed
  artifact.

The synchronous SDK retries connection failures, timeouts, and `502`, `503`, or
`504` up to three total attempts. It reuses the same trajectory ID, body, and
idempotency key with short exponential backoff and jitter.

### Metadata after finish

Keep trajectory metadata updates.

For an active trajectory:

- Validate the update.
- Apply it to the hot aggregate.
- Append `MetadataUpdated` to its journal.

For a committed trajectory:

- Load the compiled record.
- Apply the update.
- Increment its revision.
- Rebuild metadata-dependent fields.
- Atomically replace the committed file.

Previously returned finish exports and previously generated bulk artifacts
remain historical snapshots. Later reads and exports use the updated metadata.

### Core types

Introduce:

```python
@dataclass
class ActiveTrajectory:
    header: TrajectoryHeader
    graph: ConversationGraph
    exchanges: list[ExchangeSummary]
    sequence: int
    status: str
    integrity: CaptureIntegrity
    in_flight: int
    pending_commit: Future | None
```

```python
@dataclass(frozen=True)
class TrajectoryRecord:
    record_version: int
    schema_version: int
    derivation_version: int
    revision: int
    finish_request_hash: str
    trajectory: TrajectoryDocument
    exchanges: tuple[CapturedExchange, ...]
    nodes: tuple[GraphNode, ...]
    node_order: tuple[str, ...]
```

```python
@dataclass
class CaptureIntegrity:
    complete: bool
    calls_missing: int
    recovery_uncertain: bool
    delivery_uncertain_exchange_ids: list[str]
    errors: list[str]
```

`delivery_uncertain` describes whether capture knows the complete response was
handed back to the client. It is separate from whether the upstream response
and exact tokens were successfully captured.

## Persistence and Capture Flows

### Disk layout

Use:

```text
record/
├── manifest.json
├── active/
│   └── <hash-prefix>/
│       └── tr_....capture
├── committed/
│   └── <hash-prefix>/
│       └── tr_....json.zst
└── exports/
    ├── jobs/
    └── artifacts/
```

- Compute the shard prefix from the first two hexadecimal characters of a
  stable hash of the trajectory ID.
- `manifest.json` contains only static format/schema information and
  credential-free upstream provenance.
- Multiple processes may race to create the manifest atomically; later openers
  validate compatibility.
- Remove global segment ranges, sequence numbers, completion flags, and dynamic
  manifest rewrites.
- Require coherent append/read behavior, meaningful `fsync`, and atomic
  same-directory rename from the filesystem or PVC.

### Storage interfaces

```python
class ActiveStore(Protocol):
    async def create(self, header: TrajectoryHeader) -> None: ...
    async def append(self, trajectory_id: str, record: ActiveRecord) -> None: ...
    async def recover(self, trajectory_id: str) -> ActiveTrajectory | None: ...
    async def remove(self, trajectory_id: str) -> None: ...
```

```python
class CommittedStore(Protocol):
    async def put(self, record: TrajectoryRecord) -> None: ...
    async def get(self, trajectory_id: str) -> TrajectoryRecord | None: ...
    async def update_metadata(
        self,
        trajectory_id: str,
        update: MetadataUpdate,
    ) -> TrajectoryRecord: ...
```

```python
class RecordReader(Protocol):
    async def get(self, trajectory_id: str) -> TrajectoryView | None: ...
    async def list(self, query: TrajectoryQuery) -> TrajectoryPage: ...
```

A future PostgreSQL implementation must implement these product operations
rather than exposing generic SQL repositories to the capture path.

### Active journal

Reuse length-prefix and CRC framing concepts, but define a new trajectory-local
format and magic.

Supported record types:

- `TrajectoryCreated`
- `ExchangeCommitted`
- `ExchangeDeliveryConfirmed`
- `CaptureGap`
- `MetadataUpdated`
- `FinishRequested`
- `TrajectoryPoisoned`

`ExchangeCommitted` contains:

- Per-trajectory sequence.
- A diagnostic request fingerprint derived from the standard request body.
- Derived exchange metadata.
- Exact graph delta.
- Filtered headers.
- Raw request and response bodies when retained.
- Completed chunk and timing capture.
- Exact TITO token IDs, token text, offsets, masks, and logprobs.
- Initial delivery state.

The request fingerprint is diagnostic only. It must not deduplicate requests
because repeated identical prompts may be intentional resampling.

There are no partial-stream records, global events, run events, export events,
snapshots, or compaction.

A reader stops at the first truncated or CRC-invalid record. Before a
replacement writer resumes, it truncates the torn tail and continues from the
last valid per-trajectory sequence.

### Canonical committed record

The committed record contains:

- Schema, record, and derivation versions.
- Record revision and finish-request hash.
- Public trajectory metadata and integrity.
- Ordered exchanges and captured payloads.
- Materialized graph nodes and node ordering.
- Exact TITO token/text associations.
- Enough structure for viewer paths and exporters without replaying lifecycle
  events or invoking a tokenizer.

Write it by:

1. Serializing and compressing off the event loop.
2. Writing a temporary file in the committed directory.
3. Flushing and `fsync`ing it.
4. Atomically renaming it to the deterministic trajectory path.
5. `fsync`ing the directory.
6. Removing the active journal only after the committed file is visible.

Readers prefer committed data whenever both forms exist.

### Hot state and pending commits

Replace `LiveState` with:

- `TrajectoryRegistry`, holding only locally hot aggregates.
- `CommitCoordinator`, owning bounded background persistence.
- Per-trajectory ordered commit futures.

The registry resolves a trajectory as follows:

1. Return its hot aggregate if present.
2. Otherwise recover it lazily from its active journal.
3. Reject inference if only a committed record exists.
4. Return `404` if neither exists.

Do not replay every trajectory at startup.

Raw bodies and large token payloads leave hot memory after their journal record
becomes durable. Finished aggregates and TITO session state are evicted after
finalization.

### Text request flow

1. Resolve or recover the trajectory.
2. Forward the request without waiting for prior capture commits.
3. Capture response bytes and timings while forwarding.
4. After the final content/SSE chunk, enqueue derivation and persistence.
5. Complete the downstream response without waiting for persistence.
6. Serialize background commits per trajectory to preserve graph and journal
   order.

Text capture is fail-open:

- Queue saturation, parsing failure, and disk failure never alter a successful
  upstream response.
- Mark an in-memory capture gap and persist it if the store recovers.
- Continue serving later requests.
- Mark the final record incomplete when a known gap exists.
- If a text trajectory is recovered by a replacement process, set
  `recovery_uncertain=true`, because an exchange may have been returned during
  the previous process's asynchronous commit window.
- `finish()` returns a retryable failure if it cannot write the canonical
  committed artifact.

### TITO request flow

TITO requires no custom request ID or header.

1. Resolve or recover the trajectory and exact token trace.
2. Serialize turns under the trajectory lock.
3. Render the standard OpenAI-compatible request into exact input tokens.
4. Run token inference.
5. Construct the complete exact exchange.
6. Begin derivation, append, and durable flush as soon as the full outcome
   exists.
7. Stream or send the synthesized response while persistence runs.
8. Before closing the downstream response, await the durable
   `ExchangeCommitted`.
9. Close the response.
10. Enqueue `ExchangeDeliveryConfirmed` after the ASGI send completes.
11. Permit the next TITO turn after the durable exchange commit.

This preserves TTFT and content-chunk cadence while adding only residual
persistence time before response close.

TITO is fail-closed:

- A persistence failure prevents a clean response close.
- Attribution failure poisons the trajectory.
- No later turn may silently retokenize missing model output.
- A process death before durable commit causes a failed connection; the
  ordinary client retries the unchanged request.
- A cleanly completed response always has its exact exchange durably stored.

### TITO delivery ambiguity without request IDs

Persistence and network delivery cannot be atomic. The remaining narrow
failure is:

1. `ExchangeCommitted` becomes durable.
2. The process dies before or during downstream close.
3. The client retries the same standard request.
4. Capture cannot distinguish that retry from intentional resampling.

Handle this conservatively:

- A durable exchange initially has `delivery_confirmed=false`.
- Append `ExchangeDeliveryConfirmed` after the response send completes.
- When recovering a TITO trajectory, an exchange without confirmation is
  marked `delivery_uncertain`.
- If a later request's standard message history exactly includes that
  exchange's assistant output, treat that as evidence the client used it,
  append delivery confirmation, and reuse its exact token path.
- Otherwise retain the exchange for diagnosis but treat it as an
  uncertain/orphan branch.
- Graph viewing includes uncertain exchanges with an explicit flag.
- Training exports mark uncertain branches untrainable.
- Replay exports exclude uncertain exchanges.
- Do not automatically suppress a later identical request; repeated sampling
  remains valid.

The delivery confirmation means "the server handed the complete response to
the downstream transport," not proof that the remote application consumed it.

### Finish flow

1. Set the aggregate to `finalizing` so new turns are rejected.
2. Append `FinishRequested` with final metadata and request hash.
3. Wait for accepted in-flight turns.
4. Wait for all pending exchange commits.
5. Recover the authoritative full trajectory from its active journal.
6. Apply final metadata and compile `TrajectoryRecord`.
7. Atomically persist the committed artifact.
8. Render the requested per-trajectory export.
9. Evict hot state and TITO trace state.
10. Delete the active journal asynchronously.
11. Return the export envelope.

Crash behavior:

- Before committed rename: retry reconstructs and writes again.
- After committed rename: retry discovers and returns the existing record.
- Before active deletion: readers prefer committed and cleanup removes the
  redundant journal.

## Viewer, Exports, and Runtime Composition

### Lifecycle and viewer separation

Split the current control-plane application into:

- `LifecycleApp`: create, finish, trajectory metadata, health, readiness, and
  metrics.
- `ViewerApp`: read/query routes and bulk exports.

The capture ASGI application always mounts the lifecycle app. It mounts the
viewer app only when enabled.

Add:

```text
skyrl-capture serve --viewer          # default
skyrl-capture serve --disable-viewer
```

`serve` does not host browser assets. The existing Node-based `view` command
remains the UI host.

### Unified viewer reader

There is no live/offline reader distinction:

```python
def get(identifier):
    return committed.get(identifier) or active.replay(identifier)
```

- Active trajectory details are replayed from their local journal.
- Committed details are loaded directly from `TrajectoryRecord`.
- Listings deduplicate by ID and prefer committed state.
- Active files are incrementally replayed from their last valid byte offset.
- Committed detail records use an LRU cache.
- The permanent viewer index retains summaries, not all graphs and payloads.

Only one viewer/indexer process per record root is supported. Multiple browser
clients may connect to it.

### Progressive indexing

Do not block startup on a complete scan.

- Start the viewer API immediately.
- Discover active and committed files in the background.
- Return available first pages while indexing continues.
- Include `indexing`, `indexed_trajectories`, and nullable `total` in listing
  responses.
- Treat pagination as provisional while indexing.
- Reset the first-page cursor when newly discovered records affect it.
- Once indexing finishes, return stable filtered totals.
- Manual and automatic refresh fetch both new active and new committed records.
- Refresh open trajectory detail as well as run and trajectory listings.
- Remove the current assumption that a filesystem record never changes.
- Do not add WebSockets, server-push, or a global change feed.

### Bulk exports

Keep existing asynchronous bulk export behavior in the viewer service.

- Project and run exports select committed trajectories only.
- Persist export jobs as atomic files under `exports/jobs`.
- Persist compressed artifacts under `exports/artifacts`.
- One viewer process owns export job execution; no distributed claiming is
  needed.
- Finish-time rendering calls the same pure formatters without starting the
  bulk export worker.

### Runtime and configuration

Make `CAPTURE_RECORD_DIR` or `--record-dir` mandatory. Refuse startup when
absent.

Remove:

- Memory-only capture.
- `TRAJECTORY_TTL_SECONDS`.
- Automatic expiry.
- Global segment rotation and event sequence configuration.
- Global recovery replay.
- Run metadata writes.
- Trajectory deletion.
- Global recorder lag metrics.

Retain a bounded commit capacity:

- Text refuses capture work when full, records a gap, and keeps proxying.
- TITO waits for commit capacity before closing the response.

On graceful shutdown:

1. Stop accepting new requests.
2. Wait for in-flight requests.
3. Drain pending commits and delivery confirmations.
4. Leave unfinished active journals intact.
5. Close storage and transport resources.

### Monitoring

Expose:

- Pending commit count and high-water mark.
- Oldest pending age.
- Append, compression, and `fsync` latency.
- Text capture gaps and commit failures.
- Text recovery-uncertain count.
- TITO response-close persistence wait.
- TITO delivery-uncertain exchange count.
- TITO poisoned trajectories.
- Hot trajectory count.
- Lazy recovery count and latency.
- Viewer indexing progress when enabled.

Document explicitly that routing must consistently hash create, inference,
finish, and metadata requests by trajectory ID. Storage does not protect
against split-brain writers.

## Implementation Sequence and Verification

### Implementation order

1. Add `ActiveTrajectory`, `TrajectoryRecord`, integrity types, storage
   protocols, new journal codec, and disk implementation.
2. Add `TrajectoryRegistry` and `CommitCoordinator`; refactor exchange
   derivation away from global `LiveState`.
3. Move text capture to ordered asynchronous commits.
4. Move TITO capture to durable-before-close commits and delivery confirmation.
5. Rewrite create, metadata, and finish around the new registry/store.
6. Update the SDK for client-generated trajectory IDs and three-attempt finish
   retry.
7. Split lifecycle and viewer applications; add `--disable-viewer`.
8. Implement the unified progressive `RecordReader` and update viewer refresh
   behavior.
9. Move bulk export job persistence out of trajectory events.
10. Delete the global events/reducer, `Recorder`, segmented sink, global replay
    reader, expiry loop, run mutation, delete route, and obsolete configuration.
11. Bump record/log/schema versions and rewrite architecture and record-format
    documentation.
12. Regenerate demo records, fixtures, and goldens in the new format.

### Storage and recovery tests

- Active journal round-trip.
- CRC failure and torn-tail detection.
- Safe truncation and continuation after a torn write.
- Atomic committed write at every crash point.
- Committed-over-active deduplication.
- Two store instances writing distinct trajectories into one shared root.
- One reader observing appends made by multiple capture processes.
- Lazy per-trajectory recovery without startup-wide replay.
- Finished aggregate and token-trace eviction.
- Post-finish metadata rewrite and revision visibility.
- Unsupported overlapping writers documented and tested as an external
  invariant.

### Text tests

- Streaming and non-streaming responses complete without waiting for delayed
  persistence.
- Same-trajectory background commits remain ordered.
- Queue saturation and disk failure do not alter the upstream response.
- Known failures produce capture gaps.
- Recovery marks text integrity uncertain.
- Finish refuses success when the canonical artifact cannot be persisted.
- Later recovery of the disk allows an incomplete final record to be committed.

### TITO tests

- Standard OpenAI-compatible clients work without custom fields or headers.
- Exact token IDs, token text, offsets, masks, and logprobs survive recovery.
- Persistence overlaps synthesized response delivery.
- Response close waits for durable `ExchangeCommitted`.
- Commit failure prevents clean completion and blocks later turns.
- Mid-exchange process death followed by an ordinary client retry produces a
  recoverable new attempt.
- Delivery confirmation is appended after successful close.
- Recovery flags a committed exchange lacking confirmation.
- A later request containing the uncertain assistant output confirms and
  reuses its exact token path.
- A retry that does not continue from the uncertain output produces a separate
  branch.
- Viewer graph includes uncertain exchanges.
- Training exports mark uncertain branches untrainable.
- Replay exports omit uncertain exchanges.
- Identical intentional prompts remain valid repeated samples.

### API, SDK, viewer, and export tests

- Create requires a client-generated trajectory ID.
- Repeated identical create succeeds; conflicting create returns `409`.
- Finish defaults to graph and supports all four formats and options.
- Finish is idempotent across process restart.
- SDK performs exactly three attempts for retryable finish failures.
- Active and committed metadata updates are visible.
- Removed run-metadata and delete routes return `404` or `405`.
- `--disable-viewer` removes read and bulk-export routes while lifecycle and
  capture continue working.
- Progressive indexing returns an early partial page and later converges.
- Active-to-committed transition never duplicates or temporarily hides a
  trajectory.
- Manual and automatic refresh update runs, pages, statuses, and open detail.
- Bulk exports run only through the viewer service.
- Full text/TITO integration suites, security checks, packaging boundaries,
  exporter goldens, viewer tests, lint, and type checks pass.

## Assumptions and Compatibility

- No released record format requires migration; old global event logs are
  rejected.
- Agent inference requests must remain standard and unchanged.
- The routing layer guarantees one live writer per trajectory.
- Shared storage provides coherent reads, durable `fsync`, and atomic rename.
- Multi-replica deployments enable the viewer on one replica or run one
  standalone viewer; other replicas use `--disable-viewer`.
- Partial streaming visibility and mid-generation recovery are out of scope.
- TITO provides exact recovery for cleanly completed responses, with a
  documented narrow delivery ambiguity around process death during response
  close.
- No automatic cleanup or retention policy is included. Finished active
  journals are removed; unfinished orphan journals remain.
- PostgreSQL later implements the same active append, recovery, final record,
  metadata rewrite, and reader contracts.
- Preserve unrelated worktree changes during implementation.
