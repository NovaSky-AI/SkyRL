# Build log

A chronological account of what was built, in what order, what broke, and what
was decided — written for review rather than as a changelog.

Every commit below is self-contained and has its tests passing. Where a
decision departed from the PRD, or where a test found a real bug, that is
called out rather than smoothed over.

**Totals:** 495 tests, CI green. The last section is the per-trajectory
persistence rework of September 2026, which replaced the global event log with
one journal and one compiled record per trajectory.

The last twenty-seven commits are a review batch — see [Review batch](#review-batch--september-2026)
at the end. It removed more than it added: derived relations, configurable
redaction, annotation revisions and their provenance columns, one export
format, `--step-wise`, and a stored `gap_ms` that was wrong for any branched
trace.

---

## Before writing code: two inputs

**The PRD**, and — on request — **SkyRL PR #2143**, the reference TITO
implementation. Reading the reference implementation before designing the TITO
path turned out to matter: it is materially more precise than the PRD in places
where imprecision silently corrupts training data. Fourteen requirements the
PRD does not state were extracted into
[design/token-capture-parity.md](design/token-capture-parity.md) before any TITO code was
written, among them:

- The renderer must return **per-token message attribution**. Without it there
  is no way to split a rendered prompt into per-message nodes — this is *the*
  mechanism that makes one-node-per-message possible in token space, and the
  PRD does not mention it.
- An assistant node holds the chat template's **generation scaffold** before
  its sampled suffix, so `sampled_start` is a boundary, not a formality.
  Getting it wrong trains on tokens the model never produced.
- **Tools participate in prefix identity**, because templates render tool
  schemas into the prompt.
- Prefix reuse must **anchor on a completed inference boundary** and be
  **verified byte-for-byte**, forcing a full re-render on mismatch.
- On export, each sampled node must be **trainable at most once** across
  branches, or forks double-count their shared ancestors.

Two decisions were taken to the user before starting: an all-Python data plane
with the spool specified as a **cross-language byte contract** so a Rust
implementation can replace the forward path later, and **full TITO** rather
than a deferred phase.

---

## 1. Foundation — `ddfdad6`

Identifiers, configuration, the spool format, object storage, credentials, and
the complete PostgreSQL schema.

The load-bearing decision here is that **the spool is specified as a byte
format, not a Python API** ([design/disk-queue-format.md](design/disk-queue-format.md)).
It is the seam at which the data plane can be replaced without touching
ingestion, so a conforming writer needs nothing from this repository but that
document.

Details worth noting:

- **Bodies are raw length-prefixed blobs, not base64.** At capture time a body
  is already an opaque byte string; base64 would add ~33% size plus an
  avoidable encode on the request path.
- **A torn tail is expected.** A crashed writer leaves a partial record. The
  reader stops at the first short or CRC-mismatched record and reports
  truncation rather than raising. Verified against a deliberately truncated
  segment.
- **Two kinds of secret, stored differently.** A target's upstream credential
  must be recoverable, so it is AES-256-GCM encrypted with `tenant/name` as
  associated data — a ciphertext moved to another target fails to decrypt
  rather than silently authenticating as a different upstream. A lease
  credential never needs recovery, so only its SHA-256 hash is stored.
- **Object keys are sanitized** because tenant and project names reach them.

Verified as built: segment round-trip, torn-tail tolerance, resume-from-offset,
envelope round-trip, drop accounting under a full ring, key-traversal
rejection, credential owner binding, and migrations applying idempotently
against a real PostgreSQL 18.6.

## 2. Store layer and the data plane — `1167846`

One module per resource family, plus the hot path.

The data plane is a **hand-written ASGI application** with no router, no
validation, no JSON parsing, and no database access. It holds references only
to the ring, the lease cache, and the HTTP client, so a regression that adds a
synchronous write is a type error rather than a latency incident.

Design decisions made here:

- **Acceptance accounting is in-memory**, flushed by a background task. Finish
  semantics need to know how many exchanges were accepted, but the request path
  cannot write to PostgreSQL — so it does two dictionary increments and a
  background task folds them in. Drops are counted the same way, because a drop
  happens precisely when the ring is full and therefore cannot be reported
  through the ring.
- **`LISTEN/NOTIFY` for lease revocation.** A periodic refresh alone would
  leave a window in which a finished trajectory still accepted traffic.
- **A negative cache for unknown tokens**, so a hostile client cannot turn a
  bad credential into one database query per request.
- **`accept-encoding` is pinned to `identity`.** Capturing a gzipped body would
  leave the stored payload unparseable without a decompress on every read and
  would break streaming chunk attribution. The client's original value is
  preserved in the envelope so replay can reproduce the original request.
- **Errors are returned in each provider's own error shape**, so an unchanged
  SDK parses them normally instead of surfacing an opaque exception.

## 3. Phase 1: capture and replay — `6554d1c`

The ingestion pipeline, the control plane, the UI, the SDK, and the CLI.
15 tests.

Two decisions here are the ones I would most want a reviewer to check:

**Redaction hashes the original message but stores the redacted payload.** If
redaction ran before hashing, two messages differing only inside a redacted
field would collide and the context graph would claim a shared prefix that does
not exist. Hashing first keeps prefix matching exact while never persisting the
sensitive text; the hash is one-way, so it leaks nothing. Credentials are
dropped unconditionally, not by policy.

**Anthropic's top-level `system` field is materialized as a leading system
message**, recorded in the node's derivation evidence. Without it, two requests
differing *only* in system prompt would appear to share a prefix. This is the
only structural normalization in the system — there is deliberately no
cross-provider canonicalization beyond it.

### A testing decision that changed the harness

The first test suite used `httpx`'s in-process `ASGITransport`. The streaming
test failed with `chunk_count == 1`, and the cause was the transport: it
**coalesces a response body into a single chunk**, so streaming chunk timing —
one of the things this system exists to measure — cannot be exercised through
it at all.

The fixtures now run the real server over **real sockets through uvicorn**. It
costs about 50 ms per test and buys coverage of actual HTTP framing, header
handling, and chunked transfer. This also caught two production-only bugs later
(see §7).

### Fail-open, tested three ways

Acceptance criterion 3 is "a failed analytics write never changes a successful
provider response". One test is not enough for that, so there are three:

| Test | Failure simulated |
| --- | --- |
| Ingestion stopped, response still `200`, no row exists yet, restart picks it up from the spool | The analytics database is unreachable |
| Spool writer stopped and the ring filled to capacity for real | Back-pressure |
| `encode_envelope` patched to raise | A bug in capture code |

The middle one started as a stub that patched `ring.offer`, which failed —
`CaptureRing` uses `__slots__`. Filling a real ring with the writer stopped is
the faithful scenario anyway.

## 4. Phase 2: the context graph — `90aa4b8`

Exact message-prefix matching, one node per message, one parent per node, and
derived relations that carry their evidence. 25 tests including lifecycle.

The matching rule is four steps, and the interesting cases fall out of it
rather than needing special handling: a compacted history branches at the last
unchanged message, an unmatched history branches from the root, an identical
retry commits nothing and is flagged at the exchange layer, and a re-sampled
reply is a genuine fork.

Relations carry **honest confidence**: an exact prefix match is 1.0, wall-clock
adjacency is 0.5 and says so in its own evidence — *"ordering evidence only;
not a claim of application causality"*. Presenting temporal adjacency as
authoritative would be the easiest way for this system to mislead someone.

### Two bugs the tests found

**Concurrent ingestion of the same batch.** The background ingestion loop and
an explicit `drain()` (which `finish` calls) shared one spool reader with no
mutual exclusion. Both could be handed the same batch before either committed
its checkpoint. Idempotency meant no data was duplicated — but the exchange was
*re-derived*, and the second derivation found the nodes already present and
recorded the exchange as having introduced nothing, overwriting the correct
attribution. Symptom: an `input_prefix_node_id` pointing at the exchange's own
node. Fixed by serializing batch processing.

**Replay re-derived attribution.** The same root cause in a different guise: a
spool replay after a restart re-derived the graph and overwrote the original
attribution. Now a replay is a true no-op. Graph attribution is a fact about
the moment an exchange was *first* ingested, not about what a re-derivation
would conclude later.

### A definition that was wrong

`branched` was defined as "the request had messages beyond the matched prefix" —
which is true of **every ordinary continuation**, so a plain linear
conversation reported `"branched": true`. Corrected to "the matched leaf
already had a child before this exchange committed".

This has a consequence worth understanding: when two continuations fork from
one reply, exactly *one* carries `branched: true` — the first extended a leaf
with no children, the second made it diverge. The test asserts exactly that.
The fork itself stays visible in `branch_points` regardless of arrival order.

## 5. Phases 3 and 4: exports, TITO, security — `6b9faa0`

Five export formats, the full TITO path, and the security surface. 53 tests.

**Exports.** The AIPerf DAG mapping only exports a fork where the child's
context matched exactly through the parent's `model_output` node — the only
relationship AIPerf can actually replay, since it inherits the parent history
*and the replayed parent response*. A branch at a request node has no parent
response to replay from, so it becomes an independent root session. `spawns` and
joins are never inferred. Capture keeps the signed gap; only the replay format
clamps it, and each turn carries the unclamped observation alongside.

**TITO.** All 14 parity requirements are verified end to end through the real
proxy, not just against the trace data structure. The trace algorithm was
validated in isolation first — tampered prefix rejected, compaction branch,
identical retry reusing nodes, tools change preventing bridging, stale turn
rejected, length mismatches rejected, misaligned routed experts rejected — before
being wired into a route.

The deliberate divergence from the PRD: **TITO capture is fail-closed.** In
text mode the proxy observes tokens someone else produced, and a gap in an
observation is a gap. In TITO mode the proxy *produced* them, and a response
whose tokens could not be attributed exactly would enter training as if it were
exact. Verification is pure computation, so this costs no I/O; durability is
still asynchronous through the same spool.

**Security.** Redaction at rest with field-level paths, credentials dropped
unconditionally, and tenant isolation across every read and write path —
trajectory, exchange, payload, graph, relations, node, annotation, revision,
target, sink, export, and download.

### Bugs found

- **The export runner's `drain()` could return while a job was still running.**
  The background loop and `drain()` could both claim a job. Execution is now
  serialized and `drain()` waits on the same lock.
- **Object keys retained a literal `..` inside a segment.** Harmless as a
  filesystem path segment, which is why it passed a first look — but these keys
  also appear inside `file://`, `s3://`, and signed HTTP URLs, where a client
  may normalize a path. Dot runs are now collapsed.
- **A test steered the mock upstream with a request header** that the TITO
  route correctly does not forward to the token endpoint. The mock is now
  steered by the payload the proxy actually sends. The test was wrong; the
  proxy was right.

### One semantic corrected during end-to-end use

Running the real CLI against the real server showed exchanges flagged `late`
that had been sent *before* finish. `late` had been defined as "written after
finish was requested" — but ingestion is asynchronous, so almost everything
lands after finish, and a normal trajectory's whole tail was being flagged.

Corrected to "**accepted** after finish was requested". Since finish revokes
the lease, a genuinely late event only occurs when a proxy acted on a stale
cached lease — which is exactly the case the flag should surface. The test now
simulates that race explicitly rather than asserting the wrong semantic.

### A dev-setup hazard, fixed

The test suite truncates every table, and it was pointing at whatever
`DATABASE_URL` named — so a test run destroyed the local development
deployment's data mid-walkthrough. Tests now use a separate `traces_test`
database, created on demand by a session fixture, and deliberately **do not**
consult `DATABASE_URL`: that variable names a running deployment.

## 6. Benchmark — `de43a7d`

The first benchmark reported **~190 ms of "added p99 capture latency"**. That
was wrong three separate ways, and fixing it found two real scalability bugs.

### The measurement was wrong three ways

**The load generator was the bottleneck, not the proxy.** One Python event loop
driving many concurrent requests peaks near 1.7k rps on this machine and
*loses* throughput above about 8 concurrent requests — 1521 rps at concurrency
8, then 239 rps at 32. Four processes at concurrency 8 aggregate to ~4200 rps
against the same upstream, so the collapse was the generator. It was reporting
its own queueing as proxy latency. Load is now generated by several processes,
aggregating exact samples.

**The harness ceiling was unstated.** Every run now begins with a calibration
arm that drives the upstream with no proxy in the path, so a reader can tell a
proxy limit from a harness limit. It is what makes the finding in §"the proxy
costs more than capture" possible to state at all.

**Latency was measured at saturation, where it means queueing.** A
microbenchmark put the capture path at **6.5 µs of CPU per request** — which
cannot produce milliseconds of latency. Re-running open-loop below the ceiling
gave the real answer: **+0.09 ms added p50, 100.3% of baseline throughput, zero
drops.** The harness now marks latency thresholds `n/a` for a closed-loop run
rather than reporting a verdict its methodology cannot support.

### Two scalability bugs the benchmark exposed

**Ingestion was quadratic in trajectory length.** It recomputed whole-trajectory
timing *and* ran an O(n²) concurrency self-join **once per exchange**. A
4000-exchange benchmark did not finish in ten minutes. A long agent trajectory
would have hit the same wall in production. Now done once per batch, with
concurrency derivation incremental over the batch.

**Pairwise concurrency relations are inherently quadratic** in simultaneously
open calls, and for a heavily concurrent trajectory nearly every pair overlaps
— a thousand parallel calls is half a million relations that say little
individually. The pair count is now capped; the linear per-exchange
`overlapping` flag keeps concurrency visible regardless.

### A finding worth acting on

The capture-off baseline reaches only **28% of the harness ceiling** (725 rps
against 2590), so forwarding through the proxy costs 3.6× before capture is
switched on. That is `httpx`'s per-request overhead, not the capture path. A
buffered forward path for non-streaming requests bought ~5%; the rest is inside
the client library. This is the first thing to fix for absolute throughput, and
it is exactly what the cross-language spool contract exists to enable.

## 7. Documentation and end-to-end use — `fd6b78e`, `9a671c2`, `222a7c3`

Ten documents, then running the real CLI against a real server, which found two
things the test suite had not.

**`trace serve` did not start at all.** The runtime built its asyncpg pool in
one event loop and then handed the app to `uvicorn.run`, which creates a new
one — a pool is bound to its creating loop, so the first query failed. The
tests never saw it because they call `runtime.start()` directly rather than
going through ASGI lifespan. `CaptureServer` now takes a builder and constructs
the runtime inside lifespan startup. Separately, the `LISTEN/NOTIFY` listener
held a pooled connection indefinitely, which collided with the pool's
reset-on-release; it gets its own connection now.

**A rejected reward shorthand left a successful trial to expire.** On a
branched trajectory, `finish` applied the finish and *then* returned `409`
because the reward could not pick a leaf among three — so `trace run` believed
the finish had failed. The reward is validated after the finish because the leaf
count is only knowable once accepted exchanges have drained, so the rejection is
now a `reward_error` field on the `202`, with the candidate node IDs, and the
launcher prints the `trace annotate` command for each.

Also fixed: `trace list` truncated the trajectory ID, which is the one field a
reader always needs to copy.

Then 17 **UI contract tests**. The UI is deliberately build-free — no bundler,
no type checking — so nothing would catch a renamed response field until
someone opened the page. These read `app.js`, extract the endpoints it calls,
and assert every field each view reads is still present. One asserts the UI
calls only documented `/v1` endpoints, so it cannot grow a private API.

## 8. Making the proxy fast — `78ee2db`, `7791d42`

The benchmark's own finding was that the *forward path*, not capture, was the
throughput limit: with `httpx` the proxy reached 27% of the no-proxy ceiling.
`httpx` does redirect handling, cookie and auth flows, content negotiation, and
event hooks — all work a transparent proxy must not do anyway.

So the data plane now depends on a small transport protocol (send a request
with built headers and a serialized body; read the response whole or chunk by
chunk) rather than on `httpx`'s API. Two implementations sit behind it: the
`httpx` one, and a lean HTTP/1.1 client over asyncio streams with per-origin
keep-alive pooling.

| | `httpx` | `lean` |
| --- | --- | --- |
| Throughput, capture off | 697 rps | **2388 rps** |
| p50 / p99 | 9.2 / 29.6 ms | **2.7 / 3.9 ms** |
| Fraction of the no-proxy ceiling | 27% | **88%** |

`httpx` stays the **default**: this repository cannot exercise the lean
transport against a real provider's TLS endpoint, and the default on the
inference path should be the battle-tested client until someone validates the
alternative against real traffic. The lean transport is tested over a real TLS
handshake with a generated certificate — including that an *untrusted*
certificate is refused — and the whole suite runs green under both. Neither
transport can be configured to skip verification, and a test asserts no such
switch exists: the proxy holds the upstream credential, so an unverified
connection is exactly the case where it could be handed to the wrong server.

21 transport tests cover what the lean client has to get right on its own:
content-length and chunked bodies, chunk extensions, trailers, a chunk larger
than one socket read, keep-alive reuse, the retry required when a server closes
an idle pooled connection, `connection: close`, bodyless `204`, a truncated
body surfacing as an error rather than a short response, and malformed status
lines and chunk sizes.

### A second bottleneck, found in the same run

With the proxy faster, the split-ingestion arm exposed that one worker caught up
at only **17 exchanges/sec**. The cause was in code I had already "fixed" once:
moving whole-trajectory recomputation from per-exchange to per-batch removed the
quadratic *per exchange*, but each batch still **rewrote every row** of the
trajectory — O(n) writes per batch, O(n²/batch) overall.

Now the window function still runs over the trajectory, because ordering
requires it, but the `UPDATE` touches only rows at or after the earliest new
arrival, and overlap is evaluated for the batch's rows plus a monotone
propagation to rows they overlap. **63 exchanges/sec**, 3.7× better.

The number that matters more: at 950 rps the in-memory ring never went deeper
than **11** entries and nothing was dropped. The worker was ~3600 exchanges
behind when load stopped and caught up in 57 seconds. That is the durable spool
doing its job — ingestion does not have to match the capture rate, it has to
catch up before the spool fills the disk. `operations.md` now sizes shards from
that measurement.

---

## Acceptance criteria

| # | Criterion | Status | Covered by |
| --- | --- | --- | --- |
| 1 | Unchanged OpenAI/Anthropic workload runs on only a base URL and credential | met | `test_unchanged_openai_workload_is_captured`, `test_anthropic_workload_is_captured` |
| 2 | Every exchange assigned to exactly one trajectory, durable exchange ID | met | `test_unchanged_openai_workload_is_captured`, `test_every_trial_gets_a_fresh_lease` |
| 3 | A failed analytics write never changes a successful provider response | met | `test_response_does_not_depend_on_the_analytics_write`, `test_full_ring_drops_and_counts_without_failing_the_request`, `test_capture_exception_is_contained` |
| 4 | Raw payloads reproduce a same-provider request | met, via `replay` | `test_replay_reproduces_the_request`, `test_replay_payload_is_the_request_and_nothing_else` |
| 5 | Correct chronological timeline and gap distribution | met | `test_gap_is_measured_from_the_parent_not_the_previous_arrival`, `test_concurrent_siblings_have_no_gap_and_are_both_marked_overlapping`, `test_streaming_records_chunk_timing`, UI Timeline tab |
| 6 | Two divergent histories render a visible prefix branch | met | `test_divergent_histories_render_a_visible_branch`, plus the four repair cases in `test_graph.py` |
| 7 | A linear trajectory exports as a replayable chain; a fork splits | met, via `replay --shape` | `test_replay_export_carries_arrival_and_delay`, `test_replay_shapes_differ_only_on_a_fork` |
| 8 | A trajectory can be finished twice without duplicate finalization | met | `test_finish_is_idempotent`, `test_metadata_stays_editable_after_finish` |
| 9 | Credentials never persisted, and tenant isolation, under automated test | met | 9 tests in `tests/test_security.py`. Configurable redaction was **removed** on review (R1); the credential invariant it shared a module with was kept |
| 10 | Fresh lease per trial; finishing revokes it | met | `test_every_trial_gets_a_fresh_lease`, `test_finish_revokes_the_lease_immediately`, `test_expiry_closes_an_abandoned_trajectory`. Label **sealing** was removed on review (R3): metadata stays editable |
| 11 | Metadata editable post-hoc and queryable; RLVR reward | met, reshaped | 8 tests in `tests/test_annotations.py`. Provenance fields and revisions were **removed** on review (R3), and the reward moved from the leaf node to the trajectory so a branched run can carry one |
| 12 | All finished trajectories in a project export together | met, reshaped | `test_a_project_export_selects_a_snapshot_and_records_it`. The **manifest** was dropped on review (R13): it duplicated the export job row, and per-trajectory integrity travels inside the data |
| 13 | Sustained 10,000-QPS capture-on benchmark meets thresholds | **harness complete, full-scale run pending hardware** | `docs/benchmarks.md`; all thresholds pass at the rates one machine sustains |

Criterion 13 is the one that is not met, and the reason is hardware rather than
software. The comparison harness, calibration arm, open- and closed-loop modes,
zero-drop verification, and threshold evaluation with the PRD's exact numbers
are all implemented; the ceiling with no proxy in the path on this machine is
2590 rps. [benchmarks.md](benchmarks.md) states what a real run requires.

## Delivery phases

| Phase | Status |
| --- | --- |
| 1 — Leases, forwarding, envelopes, stream timing, durable async ingestion, finish/expiry, raw replay | complete |
| 2 — One-node-per-message storage, exact prefix matching, branch and overlap detection, wait distributions, trajectory UI | complete |
| 3 — AIPerf raw, multi-turn, and DAG exports; validation fixtures; replay schedule tests | complete |
| 4 — Model-aware rendering route over token-in/token-out inference, token/mask/logprob deltas at node scope | complete |

## Known gaps and next steps

Ordered by what I would do next.

1. **Validate the lean transport against a real provider and make it the
   default.** It is 3× the throughput of `httpx` and passes the whole suite
   including a real TLS handshake, but nothing here can exercise it against an
   actual provider endpoint, so it ships opt-in.
2. **Run criterion 13 on real hardware.** The harness is ready.
3. **Batch ingestion's database work.** One worker does ~63 exchanges/sec, and
   most of that is round trips: a per-exchange transaction, several statements
   inside it, and a separate settle transaction per batch. Batching the inserts
   and uploading node objects concurrently is the obvious next win, and the
   durable spool means it is a throughput question rather than a loss one.
4. **Benchmark with large payloads.** Envelope framing is linear in body size
   and the CRC is the only whole-body pass, but that scaling is unmeasured.
5. **Re-derivation tooling.** `DERIVATION_VERSION` is stamped on every derived
   record so parsed fields and relations *can* be recomputed from the raw
   envelope, but there is no command to do it yet.
6. **Spool exhaustion policy.** Currently a full ring drops and counts. Spilling
   to a secondary location, or applying back-pressure as an explicit opt-in,
   would be a real choice for a deployment that prefers latency to loss. The
   PRD lists this as out of scope.
7. **A UI view across trajectories.** The UI is per-trajectory; comparing a
   project's trials means using the API.
8. **Anthropic streaming tool-call arguments** are reassembled from
   `input_json_delta` fragments and, when a stream is truncated mid-tool-call,
   stored as `_unparsed_partial_json` rather than being dropped. Correct, but
   an exporter consuming that field should know it exists.

## Deliberate divergences from the PRD

Each of these is a decision, not an oversight, and each is documented where it
lives.

| Divergence | Why |
| --- | --- |
| **TITO capture is fail-closed**, while text capture is fail-open | The proxy produced those tokens. A response whose tokens cannot be attributed exactly would enter training as if it were exact — worse than no record. Verification is CPU-only, so it costs no latency. [token-capture-parity.md](design/token-capture-parity.md) |
| **Anthropic's `system` field becomes a leading message** | Otherwise two requests differing only in system prompt appear to share a prefix. Recorded in the node's derivation evidence. [graph.md](graph.md) |
| **Compaction branches *are* exported as AIPerf forks** when they match through an assistant node | The PRD groups them with unmatched roots. But replaying to that reply and then sending the rewritten tail is faithful, so exporting it as a fork loses nothing. A branch at a *request* node still becomes a root, since there is no parent response to replay. [exports.md](exports.md) |
| **`aiperf_multi_turn_jsonl` emits complete root-to-leaf paths**, not maximal linear segments | Segments are the DAG format's job. Repeating shared prefixes is what makes each multi-turn line independently replayable. [exports.md](exports.md) |
| **Pairwise concurrency relations are capped** | Quadratic in simultaneously open calls, with low information per pair. The linear `overlapping` flag preserves the observation. [graph.md](graph.md) |
| **Latency thresholds are only evaluated open-loop** | At saturation a latency delta is queueing, not overhead. Reporting a pass or fail there would be reporting a number the methodology cannot support. [benchmarks.md](benchmarks.md) |

---

# Review batch — September 2026

A batch of review feedback taken in one pass. Each item below removes or
reshapes something the first implementation got wrong, or built before its
purpose was clear. Nothing here has shipped, so the initial migration is edited
in place rather than accumulating migrations that describe states no database
was ever in.

## R1. Redaction removed

Configurable field-level redaction is gone: `capture/redact.py`, the
`REDACT_JSON_PATHS` setting, the `_redacted` body envelope, the node-payload
redaction pass, and the `redaction` block written into each exchange's
`source_metadata`. Bodies are now stored exactly as they were sent.

**What was deliberately kept.** `redact.py` conflated two things: a
configurable redaction policy, and the unconditional dropping of credentials.
The second is a security invariant, not a feature — without it the client's
lease token and the upstream provider key would be persisted in every captured
exchange. So the header allowlist and `ALWAYS_DROP_HEADERS` survive as
`capture/headers.py`, and `test_credentials_are_never_persisted` still guards
them. Only the configurable half was removed.

Five tests went with it; the credential test was kept and its docstring
corrected, since "whatever the policy says" no longer refers to anything.

## R2. Derived relations removed

`node_relations` is gone entirely: the table, `ingest/relations.py`, both
derivation passes, `store.list_relations`, the `/v1/trajectories/{id}/relations`
endpoint, the `relations` key on the graph endpoint, `trace relations`, the UI
panel, and the `relation` record type in the native export.

The table held four edge types, and none of them was load-bearing:

- `prefix_extension` restated `message_nodes.parent_node_id`. Its one unique
  field was `shared_prefix_messages`, which is the depth of the fork node and
  so is recoverable from the tree.
- `temporal_predecessor` restated timestamp ordering at confidence 0.5, next to
  exact matches at 1.0, in a table where the confidence column was the only
  thing separating a fact from an adjacency guess.
- `explicit_parent` restated `exchanges.previous_exchange_id`, which is still
  set from `previous_response_id`.
- `concurrent_sibling` was pairwise and quadratic, capped at 5000 per batch by
  a setting that now has nothing to cap. The linear per-exchange `overlapping`
  flag was always the useful form and is untouched.

Nothing read them: AIPerf fork attribution comes from `parent_output_node_id`
on the exchange, and `trace graph` walks node parents. Verified before deleting.

**A field that went with them.** `CommitOutcome.branched` existed only to be
written into relation evidence — it recorded which of two continuations
*arrived* first, not anything about the resulting shape. With relations gone it
had no reader, so it is gone too, and `docs/graph.md`'s "what branch means"
section is now the one-line definition: a node with more than one child is a
fork.

Six tests were rewritten to assert against the tree rather than relation
evidence — fork depth from the path length, overlap from the per-exchange flag,
explicit parentage from `previous_exchange_id`. One
(`test_temporal_relation_is_lower_confidence_than_prefix`) was deleted, since
what it guarded no longer exists.

## R3. Labels and annotations collapsed into one mutable surface

`annotation_current` and `annotation_revisions` are gone. Labels and
annotations are now two columns on `trajectories`: `labels TEXT[]` for bare
string tags, `annotations JSONB` for a free-form key/value document. Both are
mutable for the life of the trajectory, both hold only their latest value, and
`labels_sealed` no longer exists.

**Trajectory scope, not node scope.** This is the change that motivated the
rest. A branched run — sub-agents, a re-sample, a repaired turn — yields
several training samples from one trial, and they share one reward. Attaching
that reward to a node cannot express it, which is why `finish(reward=...)`
previously refused whenever the graph had more than one leaf assistant node.
That whole mechanism is gone: the leaf-count check, the `reward_error` field
on the `202`, the `leaf_assistant_node_ids` hint, the SDK's `annotate_leaf`,
and roughly forty lines in `trace run` that retried the finish without the
reward so a rejected annotation could not leave a successful trial to expire.
`reward` is now sugar for `annotations["rlvr_reward"]` and cannot fail.

**No predeclared fields.** `annotator_kind` and its `CHECK`,
`evaluator_version`, `rubric_version`, `confidence`, `calibration`, `source`,
`label`, `explanation`, `privacy_status`, `provenance`, and `numeric_value` are
all gone. Filtering reads the document directly —
`(annotations->>'reward_A')::float > 0.4` — which costs no schema change for a
new key; GIN indexes cover both columns, and a deployment running one numeric
filter constantly can add an expression index for that key alone.
`numeric_value` was the single-slot, hardcoded version of exactly this idea.

**No revisions.** Deliberate, and the one real loss: an overwritten reward
leaves no trace that a human overrode a judge. Accepted for simplicity, with
the reasoning that current-state metadata and an audit trail are different
problems and only the first is wanted now.

**API.** `PUT /v1/nodes/{id}/annotations/{name}`, its listing, and its
revisions endpoint are replaced by `GET`/`PATCH
/v1/trajectories/{id}/metadata`. Annotations merge rather than replace, so two
writers touching different keys do not clobber each other.

**CLI.** `--label k=v` was already key/value, so the split is a real break and
was made loudly rather than by leaving a flag that means something new:
`--tag NAME` takes a bare string, `--annotate k=v` takes a pair, on both
`trace run` and `trace annotate`. A tag containing `=` is rejected with a
message pointing at `--annotate`. `trace revisions` is gone.

`tito_jsonl`'s reward lookup changed with it: it read `rlvr_reward` from a
per-node annotation on the leaf, and now reads the trajectory's annotations, so
every trainable branch of a branched trace carries the same value — which is
the semantics the change was for.

## R4. Upstream providers are pluggable

`type` was a closed enum of three, hardcoded in the migration, the request
model, and the store, and branched on in a dozen more places. It is now a name
in a registry: a provider is one class and one `register()` call, with no
migration, because the database stores only the name.

The split is two protocols, composed rather than inherited:

- **`WireFormat`** — route classification, parsing, error envelopes.
- **`UpstreamServer`** — environment variables, client route suffix, upstream
  URL join, credential header, mode, tokenizer requirement.

Composition is what lets two servers share a parser. Classification is
path-based and parsing dispatches on the endpoint kind, so both formats share
`BaseWireFormat` for those and differ only in the error envelope.

The built-ins are the test of whether the split is right. `AnthropicServer` is
three members and one method; `TokensServer` is two members. If either had
needed more, the responsibilities would be in the wrong place.

**Fail closed.** `registry.get` raises `UnknownUpstream` rather than falling
back. Three silent `snapshot.get("type", "openai")` defaults are gone, from
`lease_cache`, `trajectories`, and the SDK's environment injection. With a
closed enum the fallback was harmless; with an open set an unregistered
Anthropic target would have got Bearer auth and OpenAI parsing instead of an
error. The proxy maps the exception to 502 — not 503, which the error tables
map to `overloaded_error`, and this is misconfiguration rather than load.

One place keeps the OpenAI shape as a genuine fallback: reporting the error
needs an envelope, and the format we failed to resolve is the one that would
choose it. The same applies before a lease exists at all.

**Removed with it.** `proxy/errors.py` (the shape tables moved onto the wire
formats), `forward.build_upstream_url` and `forward.endpoint_kind_for`,
`sdk.PROTOCOL_ENVIRONMENT`, `targets.VALID_TYPES`, `Target.is_tito`, the
`Literal` on the request model, and both `CHECK` constraints — `targets.type`
is now free text validated by the registry.

`endpoint_kind_for` took a `target_type` parameter it never read; that dead
parameter is gone with the function.

**Naming.** The `tito` type and mode became `tokens`, and
`TITO_CAPTURE_STRICT` became `TOKENS_CAPTURE_STRICT`, because `tito` means
nothing outside this repository. The internal `tito/` package and
`design/tito-parity.md` kept the name for now — the first is an implementation
module, the second is explicitly about parity with SkyRL's TITO implementation.
(R27 finished the job: both were renamed too, and only SkyRL citations keep it.)

`tests/test_upstream.py` is the acceptance criterion: it registers a
`gemini-compat` provider in fifteen lines, creates a target and a trajectory
against it, and captures a graph — proving no migration and no edit outside
the provider's own definition.

## R5. Exports rebuilt around four use cases

Five formats became four, named for what they are for rather than for the tool
that consumes them. Each use case gets exactly one, and none is a superset of
another.

| Was | Is |
| --- | --- |
| `native_jsonl` | `graph` |
| `raw_payload_jsonl` | absorbed by `replay`, which inlines the payload |
| `aiperf_multi_turn_jsonl`, `aiperf_dag_jsonl` | one `replay` with `--shape` |
| `tito_jsonl` | `token_samples` |
| — | `text_samples`, **new** |

`text_samples` filled the real gap: distillation and speculative-decoding
training had **no** format. The dedup logic they need existed only inside
`tito_jsonl`, in token space.

### `graph`

One line per trace. An exchange stopped being a record type: a model call
produces exactly one model-authored node, so the call attaches to that node,
and a node with more than one child is the whole representation of a fork.
That removed the repetition that made the old format tiring to read — in an
11-record file, the two version fields appeared 11 times and `trajectory_id`
10 times, and every node paid for an `annotations` array whether or not it had
one.

Dropped as noise: the `payload` envelope around each message, character and
block counts, the monotonic clock fields, headers, `source_metadata`, `depth`,
`sequence`, and the node-linkage columns the tree already encodes.

### `replay`

Gained `arrival_ms`, the one genuinely new capture-side field in the set.
Without it an Nx replay can scale pacing inside a session but not the arrival
distribution, which is most of what makes an hour of captured traffic behave
like an hour of traffic.

Mooncake is documented as a *projection* off `replay`, not adopted as a
format: it carries token lengths and block hashes instead of text, so it is
tokenizer-bound and cannot serve replay against a different model — the
opposite of what was wanted from it.

### The `trainable` contract

Both sample formats share one envelope and one rule. **Rows are never
dropped**; `trainable` carries every filtering decision:

```
trainable = author is model
            AND (no --dedup-nodes    OR this row first claims the node)
            AND (no --mask-abandoned OR the row is not abandoned)
```

The row count therefore means the same thing under every flag combination, and
a fully masked row stays a legible record of a branch the agent walked away
from. `node_id` and `author` are always present, so a consumer can ignore
`trainable` and recompute any policy.

`token_samples` gave up three behaviours to fit: it dropped paths with nothing
trainable, deduped unconditionally, and had no notion of an abandoned branch.

**`abandoned` cost a correction.** The obvious definition — a childless
`model` node — flags *every* path, since every path ends in one. The real
signal is that the branch lost a race: the leaf's parent has other children.
One definition then covers both the discarded original of a repair and the
unpicked samples of a best-of-N.

**`--step-wise` and `is_last_step` are gone.** Step-wise guarded against
trainers double-counting a branched trajectory, but `--dedup-nodes` already
guarantees each sampled node is trainable at most once, and both branches of a
genuine fan-out are real generations worth training on. `is_last_step` was
positional over the emitted rows — on a two-branch fan-out it marked whichever
row happened to be last — so it described nothing once a trajectory had more
than one path.

### Schema changes underneath

- `message_nodes.origin` → `author`, with values `client`/`model`.
- `message_nodes.context_hash`, chained as
  `sha256(parent context_hash ‖ delta_hash)`. Equal hashes mean equal
  conversations *across* trajectories, which the `(trajectory, parent, delta)`
  identity tuple cannot express. `message_hash` stays beside it.
- Sampling parameters, `max_output_tokens`, `tools` and `tools_hash` were
  parsed and then buried in the exchange's `source_metadata` grab-bag. They are
  columns now: a training row cannot be reproduced without them.
- `trajectories.payload_capture` → `bodies`, `outcome` → `command_result`,
  `dropped_event_count` → `calls_missing`, `late_event_count` →
  `calls_after_close`. The old names described the pipeline; the new ones
  answer the question a consumer actually has, which is whether the trace can
  be trusted.
- `exchanges.tito_uri` → `tokens_uri`, `payload_sampled_out` →
  `bodies_omitted`, and the node payload key `tito` → `tokens`.

**`exchanges.gap_ms` is gone as a column** and is derived at read time instead
(`exchanges.GAP_MS_SQL`). The stored value was a window function over arrival
order, and the moment a trajectory branches the previous arrival is a sibling
rather than a predecessor — so the number was wrong for exactly the traces
worth measuring. It is now read from the parent exchange via
`parent_output_node_id`. On the sub-agent fan-out both sibling calls now report
their wait from the shared orchestrator call; before, the second reported a
wait on the first.

That change also fixed the concurrency test's premise. Two overlapping,
unrelated calls used to report a negative gap; they now report *no* gap,
because neither followed the other, while the per-exchange `overlapping` flag
still records that they ran at once.

### Also

`parse_exchange` no longer takes a `provider` argument it never read, and the
ingest worker now parses through `registry.get(...).wire.parse(...)` — so the
`WireFormat` abstraction is load-bearing rather than decorative.

Migration `002_exchange_input_leaf` was folded into `001`. It described a state
no database was ever in, and every item in this batch edits `001` in place.

## R6. The repair case is tested

A client that edits a sampled assistant message and replays it was untested in
both modes. Every existing branch test replayed the assistant *verbatim*, so
the whole suite agreed on a case that never distinguishes what the model
produced from what a client claimed it produced.

It also produces a structurally different graph from anything previously
asserted. Every other branch case forks at a `model` node —
`test_graph.py` and `test_tito.py` each asserted exactly that. A repair forks
at the **user** node above it, with two assistant siblings of differing
author, and leaves the sampled reply as a leaf.

Added to `test_graph.py`:

- the fork lands on a user node, and the two siblings are one `model` and one
  `client` with different message hashes;
- the sampled original survives as a leaf whose sibling carried on — the
  "abandoned" signal, asserted as *the parent has another child that
  continued* rather than as childlessness, which every path has;
- `parent_output_node_id` is `None`, so the exporter cannot treat it as a
  fork, and no wait can be attributed either;
- a **structured** repair that edits `tool_calls[].function.arguments` and
  nothing else still forks, since real tool repair does not edit message text.

That last one needed a fixture that did not exist. The first draft passed for
the wrong reason: the mock has no tool-call support, so the "sampled" message
was plain text and the "repaired" one was a `tool_calls` message — they
differed trivially rather than by one character inside a nested field. The
mock upstream now takes `x-mock-tool-arguments`, so the test samples a
malformed tool call and repairs only its arguments.

Added to `test_tito.py`: the same fork shape in token space, that the client's
substitute carries no sampled tokens and therefore no `sampled_start`, that
its tokens never appear in a `loss_mask`, and that an exact replay stays a
continuation rather than becoming a fork.

## R7. CLI renamed to `icap`, and it fails like a tool rather than a defect

**`trace` shadowed macOS's `/usr/bin/trace`**, so the command is now `icap`.
The entry point, the Typer app name, every doc, the launcher's own output
prefix, and the `launcher` value recorded in `source_metadata` all moved.
`PRD.md` was deliberately left alone: it is an input document, not a
description of the current tool.

**An unreachable control plane printed a stack trace.** The transport
exception escaped `CaptureClient` before `_handle` saw a response, so the most
common way to use the tool wrong — the service is not running, or `--endpoint`
is wrong — looked like a bug in the tool. Every verb now goes through one
`_request` that turns `httpx.RequestError` and `httpx.TimeoutException` into
`CaptureError`, and `app_main` catches `CaptureError` for a one-line message
and exit 1:

```
error cannot reach the control plane at http://127.0.0.1:9999: [Errno 61]
Connection refused. Is `icap serve` running, and is --endpoint right?
```

The message names the endpoint it tried and what to check, since those are the
two things the operator needs and neither was in the traceback.

## R8. Sweep

Verified by grepping the whole repository for every removed and renamed term.
Three findings were real rather than cosmetic:

**A test was passing for the wrong reason.**
`test_tenant_cannot_annotate_another_tenants_node` asserted `404` on
`/v1/nodes/{id}/annotations` and its revisions endpoint — routes that no longer
exist. It would have passed against *any* implementation, including one with no
tenant isolation at all. Rewritten against the trajectory metadata endpoint,
and it now also asserts the rejected write did not land under the owning
tenant.

**A repair fixture was passing for the wrong reason**, described in R6.

**`docs/design/export-formats.md` was deleted.** It was a change proposal
written before the work, so it carried a "Removed / was → is" table that
duplicated `docs/exports.md` while describing states the code no longer has.
Its one unique section, the shared conventions, moved into `exports.md`, which
is now the single spec.

The acceptance-criteria table above was updated where this batch changed what a
criterion means: 4 and 7 are met through `replay` rather than the formats they
named, 9 no longer covers configurable redaction, 10 no longer seals labels,
and 11 is met in reshaped form without provenance or revisions.

The quickstart was then run verbatim against a fresh database — every command,
the agent snippet, both exports, and both SQL filter examples.

## R9. A quadratic read the derived gap would have introduced

Deriving the gap instead of storing it (R5) traded a write for a read, and the
read had no index to use. The lookup finds a call's parent *by that parent's
`output_node_id`*, but the only index covering it was
`(trajectory_id, parent_output_node_id)` — the wrong column — so Postgres
scanned the trajectory's index entries and filtered, once per row. Reading one
trajectory was quadratic in its length.

Measured on a synthetic 5,001-exchange trajectory:

| | Time |
| --- | --- |
| Reading the old stored `gap_ms` column | 11.5 ms |
| Derived, no usable index | 3,236 ms |
| Derived, with `exchanges (trajectory_id, output_node_id)` | **26.8 ms** |

The index turns a filter into an index condition on both columns, which is the
280× between the second row and the third.

**It does not restore parity, and it was wrong to imply otherwise.** Reads are
~2.3× more expensive than they were with a stored column. What pays for that:
the old value needed a window function and an `UPDATE` on every ingested
batch, measured at ~105 ms per run on the same data, and that is gone
entirely. Cost moved from the write path — the measured bottleneck at 63
exchanges/sec — to a read that happens when somebody looks.

And the old number was wrong on any branched trajectory, which is the actual
reason for the change. The performance shift was a consequence, not the goal.
If 26 ms ever becomes a problem the fix is to materialize the value again,
derived from the parent chain rather than from arrival order.

Worth recording because it is the third time in this project that moving work
around introduced a quadratic — per-exchange derivation in §6, per-batch row
rewrites in §8, and now a per-row parent lookup. The pattern each time was the
same: the new shape was correct, and nothing checked whether the access path
still existed.

## R10. Exports are readable without a decompressor

A single-trajectory export was unconditionally zstd-compressed, which is right
for the object store and useless at a terminal — `head`, `jq`, and `cat` all
fail on it, and macOS has no `zstd` by default. The first thing anyone does
with an export is look at it.

`icap export --compression none|gzip|zst` now decides what `--output` gets,
defaulting to `none`:

| | Single trajectory | Project |
| --- | --- | --- |
| `none` (default) | `.jsonl` | `.tar` |
| `gzip` | `.jsonl.gz` | `.tar.gz` |
| `zst` | `.jsonl.zst` | `.tar.zst` |

This is a local-download concern only. The **stored** artifact and any sink
delivery stay compressed, which is worth keeping — measured on a 60-call
trajectory:

| Format | Raw | Compressed | Ratio |
| --- | --- | --- | --- |
| `graph` | 136,846 B | 14,145 B | 9.7× |
| `text_samples` | 81,053 B | 2,792 B | 29× |
| `replay` | 37,629 B | 482 B | 78× |

A project export bundles a manifest beside the data, so it is always a tar and
only the wrapper varies. An `--output` name whose extension disagrees with
`--compression` is **refused rather than written**, so a file never lies about
its contents.

## R11. The versions had not moved

Writing the compression flag exposed it: a `replay` line said
`"schema_version": 1` while the documentation showed 2. The record shape and
the export set changed incompatibly across this whole batch and
`SCHEMA_VERSION` had never been bumped, so every export was claiming to be the
old shape.

`SCHEMA_VERSION` is now 2, with a comment naming what changed.
`DERIVATION_VERSION` is 2 as well, because the wait before a call is now
measured from its parent rather than from the previous arrival — which changes
every gap on a branched trajectory, and that is exactly what a derivation
version is for. The migration's column defaults were moved in step, so a
hand-inserted row is not stamped as an older shape than the code writes.

## R12. Documentation sweep

Checked mechanically rather than by reading, because the batch renamed enough
that eyes would miss things. Four checks, each run against the live service:

**Every documented command executed verbatim.** Extracted every `icap …` from
the docs and the README, substituted a real trajectory ID, and ran it. This
found three commands that would have been **refused**: `--output
tr_123.jsonl.zst`, `--output terminal-bench-traces.tar.gz` and `--output
dataset.tar.gz` had never been updated for `--compression`, so the guard added
in R10 would have rejected the docs' own examples.

**Every documented flag and route exists.** 37 flag mentions across 13 command
forms, and every `/v1` route in both directions — documented-but-gone and
exists-but-undocumented are both empty.

**Every export schema matched against real output.** Exported all four formats
and diffed the actual keys against `exports.md`: top level, node, `call`,
`turn`, and message. Nothing undocumented.

**Env vars reconciled with `config.py`.** Four were readable but undocumented,
predating this batch: `UPSTREAM_CONNECT_TIMEOUT`, `MAX_STREAM_CHUNK_RECORDS`,
`SPOOL_COMPRESS`, `INGEST_UPLOAD_CONCURRENCY`. Now documented.

Also fixed: `schema_version: 1` in two doc examples after the bump to 2;
`command_result` present in an example but never explained, so its distinction
from `status` is now written down; `/v1/policy` described in prose with no key
names; and the totals at the top of this log.

The failures the harness reported that were *not* real: inline `# comments`
that the extractor did not strip, example names like `glm-tokens` that do not
exist in a scratch database, and — for a while — zsh not word-splitting an
unquoted variable, which made three working commands look broken.

## R13. The manifest is gone, and so is the archive

A project export was a `tar.gz` bundling `manifest.json` beside `data.jsonl`.
That made `--output turns.jsonl` fail on a project and succeed on a single
trajectory, for a reason the caller could not see — and the error said only
that `--compression none writes '.tar'`, which reads as a claim about
compression rather than about archiving.

**The archive existed only to carry the manifest.** With the manifest gone,
every export at every scope is plain JSONL: a project export is a
concatenation of its trajectories, not a bundle.

**The manifest was almost entirely a denormalized copy of the export job row.**
`options`, `selected_trajectory_ids`, `record_count`, `byte_count`,
`checksum`, `format`, `project` and `created_at` are already columns on
`exports`. What it held that was not a column — `capture_integrity` — is
per-trajectory information that already travels *inside* the data, in each
trajectory's `integrity` block, which is the level you want it at: a
dataset-wide count says something was lost without saying where.

One thing genuinely did surface only through the manifest: `options`, the
flags that produced the export. Dropping the manifest without noticing would
have lost the flag provenance that R10 relies on, since the sample formats
never drop rows and a dataset that trains on nothing is explained by its
flags. `options` is now on the export job record.

Removed with it: the `manifest` column, the separately stored `manifest.json`
object, `keys.manifest_key`, the `_tar_gz` helper, and the `--manifest` flag
that had existed for about ten minutes.

**Acceptance criterion 12 changes meaning.** It read "All finished
trajectories in a project export together with a manifest". The first half
still holds and is still tested; the manifest half is deliberately dropped.
The criterion is now met as "…export together, with the selection and flags on
the job record and capture integrity on each trajectory".

Two earlier fixes in this area, folded in here since they never stood alone:
the output-name check now runs **before** the export instead of after, because
doing the work and then declining to write it wasted the run and read like a
failure of the export rather than of the name.

## R14. Two defects in `replay`'s session IDs, found by a question

Asked what `-s0000` means, and the answer turned out to be "two different
things, depending on a flag."

**The shapes collided.** Unifying `aiperf_multi_turn_jsonl` and
`aiperf_dag_jsonl` into one `replay` format with `--shape` (R5) collapsed their
session-ID suffixes too. The originals used `-t` for a turn chain and `-s` for
a segment; both became `-s`. So exporting one trajectory in both shapes gave
`tr_…-s0000` meaning "plan + sub A" in one file and "plan" in the other.
Restored: `-t` for `multi-turn`, `-s` for `dag`, with a test asserting the two
sets are disjoint.

**`dag` had become lossy.** The original emitted `parent_session_id` and
`forks`, linking each segment to where it came from and what continued from
it. My rewrite dropped both. A segment carries no context of its own — that is
the point of the shape — so without the links the pieces cannot be put back
together, and `dag` was not a compact encoding of the tree but a discarding of
it. `docs/exports.md` claimed it "splits at every fork and repeats nothing",
which was true and beside the point.

Both restored, and `exports.md` now has a Sessions section that answers the
question directly: a session is one replayable unit, a trajectory produces
more than one only when it branches, and here is what the two shapes do to the
same fan-out.

## R15. A runnable branching workload

`scripts/subagent_pipeline.py` produces the shape the graph work was designed
around, on demand: an orchestrator plans, three sub-agents inherit that plan
verbatim and do multi-turn work, then the orchestrator synthesizes. Seven model
calls fanning out from one reply, and no tracing code — it reads
`OPENAI_BASE_URL` and `OPENAI_API_KEY` and nothing else.

It exists because the interesting behaviour in this system only appears when a
trajectory branches, and every other way of producing one was ad hoc.

Running it surfaced something worth recording about `replay`. On this workload
the two shapes are not equivalent:

| shape | sessions | turns | arrivals (ms) |
| --- | --- | --- | --- |
| `multi-turn` (default) | 4 | **10** | `0, 0, 0, 0` |
| `dag` | 5 | **7** | `0, 102, 270, 437, 560` |

The workload made **7** calls. `multi-turn` repeats the shared prefix, so it
re-runs the orchestrator's plan in all four sessions — a 43% inflation — and
because every root-to-leaf path begins at the same root call, all four sessions
arrive at zero, collapsing a 560 ms fan-out into a simultaneous burst.

`dag` reproduces the call count exactly and preserves the arrival spread. For a
format whose purpose is replaying traffic, that is the faithful shape whenever
a trajectory branches; `multi-turn`'s repetition buys standalone lines, which
is a different goal. Whether the default should flip is an open question, not a
decision — it is recorded here and in `scripts/README.md` rather than acted on.

## R16. `replay` scheduling made relative, and `multi-turn` removed

Two changes from one observation: absolute arrival offsets bake a single
rollout's latencies into the artifact.

### `multi-turn` is gone

Its documented justification was that repeating shared prefixes "makes each
line independently replayable". That was false — `exchange_path_messages` is
called for *both* first-turn cases in the code, so a `dag` segment rooted at a
fork already sends its complete observed context. Both shapes were
self-contained; only one was faithful.

On the sub-agent pipeline, against 7 real calls: `multi-turn` emitted **10
turns** and reported all four sessions arriving at `0`, because every
root-to-leaf path begins at the same root call. `dag` emitted 7 with the real
spread. The one thing `multi-turn` uniquely offered — a session per complete
conversation — is what `text_samples` now does, with trainability attached.

So `replay` has one shape, `--shape` is gone, and a session is a maximal linear
run of calls between forks.

### Scheduling is relative

| Field | Measured from | Present when |
| --- | --- | --- |
| `delay_ms` | end of the parent session's last turn | there is a `parent_session_id` |
| `arrival_ms` | first call in the **export** | always |

A replayer starts a child at `parent_end + delay_ms`, a root at
`trace_start + arrival_ms`. On the pipeline that reads:

```
session   arrival   delay  parent
-s0000          0       -  -        (the plan)
-s0001        102      81  -s0000   (handlers)
-s0002        270     248  -s0000   (tests)
-s0003        437     416  -s0000   (deps)
-s0004        560     539  -s0000   (synthesize)
```

Measuring `arrival_ms` across the whole export rather than per trajectory also
fixed a third instance of the same bug: a project export restarted every
trajectory at zero, so replaying it fired them all at once. Two runs 1.4s apart
now export as arrivals `0` and `1414`.

### The limit, stated rather than papered over

The fan-in is invisible, and deliberately so. `-s0004` synthesizes all three
sub-agents' findings, but its context matched the prefix through the fork, so
the graph sees a fourth child of `-s0000` and its `delay_ms` of 539 ms silently
contains "the three siblings took 439 ms in this rollout" plus "the
orchestrator thought for 121 ms" — the script's own `think=0.12`.

The fan-in *is* observable: each sibling's sampled output appears verbatim in
the synthesizing call's context, while the `deps` sub-agent — which merely ran
after `tests` because of a `for` loop — contains neither. So content matching
would separate real data dependencies from scheduling artifacts.

It was considered and rejected. Deriving causality from content would mean
guessing at an execution graph the proxy cannot see, and it fails the moment a
framework summarizes results rather than pasting them. The consequence is
accepted and written into `exports.md`: replaying against a much slower model
can issue a synthesizing call before its inputs would be ready. The same
reasoning keeps `graph.md` from inventing joins — the system measures the gap,
it does not explain it.

## R17. The replay record has a stable shape

`parent_session_id`, `delay_ms` and `forks` were emitted only when they had a
value, so a root session had no `delay_ms` key at all and a consumer had to
test for each field before reading it.

They are now always present: `null` for a root's parent and delay, `[]` for a
session nothing forks from. This is the convention the turn level already
used — the first turn in a session carries `delay_ms: null` rather than
omitting it — so the two levels now read the same way. A test asserts every
record has an identical key set.

## R18. A sample row says what produced it

Two problems in `text_samples`, one raised and one it exposed.

**A message was flattened beside itself.** Each carried `role` and `content`
*and* a `message` object holding both. The copy was lossy: for a tool call the
flattened `content` is `null`, so a consumer reading it would have seen nothing
while the real message carried the call — the exact structured-repair case R6
added tests for. A message record is now `node_id`, `author`, `trainable`, and
`message`: what capture added, beside the message as captured.

**A row did not say what it was sampled under.** No model, no tools, no
sampling parameters — so a row could not be reproduced, and could not be read
correctly either, because a chat template renders tool schemas into the prompt.
The same messages under different tools are a different data point; that is
also why tools participate in prefix identity (parity requirement 6). Both
sample formats now carry `model`, `tools`, `tools_hash` and `sampling`.

These are per call, and a path can span several. Where they disagree — an agent
that changes tools mid-run — the row sets `conditions_vary: true` and leaves
the four fields null, rather than stating one of several and lying about the
rest. `graph` carries the per-call detail.

Fixing this also corrected `token_samples`: it took `model` from the target's
configuration, which is frequently unset, overwriting nothing useful. It now
comes from the call that produced the tokens. `tokenizer` stays a target
property.

## R19. Training each generation once is the default, and it was silently off

`--dedup-nodes` named a mechanism rather than a decision, and its defaults
disagreed between the two sample formats: `text_samples` repeated targets,
`token_samples` did not. The CLI then sent the key **explicitly on every
export**, so `options.get("dedup_nodes", True)` in the `token_samples` branch
received the CLI's `False` — and every command-line export of `token_samples`
silently lost parity requirement 12, *each sampled node is trainable at most
once*, the one thing that file exists to guarantee.

Making it the default fixes that structurally rather than by correcting a
number. The flag is now `--allow-repeated-targets`, off by default and named
for the unusual thing it does: letting one sampled message be a training target
in several rows.

On the sub-agent pipeline, which makes 7 calls:

| | plan is a target in | total targets |
| --- | --- | --- |
| default | 1 of 4 rows | **7** |
| `--allow-repeated-targets` | 4 of 4 rows | 10 |

The opt-in is a real choice — an RL setup may want a shared prefix weighted by
how many outcomes it led to — but not a default worth having.

`masked_reason` gains the matching vocabulary: `duplicate_node` is now
`repeated_target`. Help text for all three sample flags now names the formats
they apply to, rather than saying "Samples:".

## R20. `git add -A` was committing the user's working files

Noticed while checking the docs: `scripts/` contained files nobody meant to
track. Blanket staging had swept the operator's scratch into commits whose
messages describe unrelated work — export dumps (`ms-turns-dag.jsonl`,
`graph.jsonl`, `turns.jsonl`, `project.tar.gz`, `manifest.json`,
`data.jsonl`), their later deletion, an `agent.py`, and the deletion of a
`scripts/README.md`.

This is the second time in this batch. The first was `playground/`, where the
same command committed three throwaway scripts; the lesson was written down and
then not applied.

Generated export artifacts are now in `.gitignore` — they are data written
beside the scripts that produce them, not source — and
`scripts/text-samples.jsonl` is untracked. Staging is explicit from here.

## R21. Tools belong in node identity, and the sample envelope shrinks

A capture bug, found by asking what a training sample should carry.

**Tools did not participate in prefix identity in text mode.** Parity
requirement 6 says they must — a chat template renders tool schemas into the
prompt, so the same messages under a different tool set are a different
context — but it was implemented only in the token path. Demonstrated with two
calls sending identical messages, receiving an identical reply, differing only
in tools:

```
before:  2 nodes, 1 leaves, 0 branch points     <- one node, two tool sets
after:   4 nodes, 2 leaves, 0 branch points     <- two roots, one per tool set
```

The `exchanges` rows showed different `tools_hash` values resolving to the same
`output_node_id`: the graph reported one generation where there were two.

`delta_hash` now covers the message *and* the tool set, and `context_hash`
chains that rather than the bare message hash — which is what the migration
comment already claimed it did. A tools change branches from the **root**, not
from the call that changed them, because that is where the rendered context
first differs; the token path does the same, since a tools mismatch forces a
full re-render (requirement 8).

**That let three fields leave the sample envelope.**

- `sampling` decides what a *replay* sends, and `replay` carries it. A training
  row is the text and the conditions that shaped the prompt; temperature shaped
  neither.
- `tools_hash` is redundant beside `tools`.
- `conditions_vary` is unnecessary: tools are now uniform along a path by
  construction, since a change branches instead. Where two tool sets exist they
  produce two rows — the same philosophy the format already applies to every
  other divergence, rather than a flag hedging inside one row.

`model` stays, and is the one field that can still have no single value: unlike
tools it is not part of the context, so it does not branch, and a workload that
switches model mid-path gets `null` rather than one of two answers.

## R22. A runnable demonstration of the tools case

`scripts/tool_change.py` is the workload R21 was found with, as an agent loop
rather than a one-off check: the agent asks with read-only tools, the model
says it needs to write, so the agent widens the tool set and asks *the same
question again*.

The replies are pinned identical on purpose, so nothing except the tool set can
account for a divergence:

```
4 nodes, 2 leaves, 0 branch points
- user          26ch  QRX521AF
  * assistant     31ch  P3AGYWCS
- user          26ch  XRMH8ZMM
  * assistant     31ch  YWG4N362
```

Two roots. The two user nodes carry the *same* `message_hash` (`7bffbeb628`)
and different `delta_hash` (`c57e850202`, `6311a0d7a5`) — the message is
identical, the context is not. Before R21 this was one node and one reported
generation.

Exporting gives two `text-samples` rows, one per tool set, neither hedging
about which tools produced it.

## R23. The model is part of node identity too

Same argument as tools, one step further. Node identity has to cover
everything that makes two generations different, and the model that produced
one qualifies: two models answering the same prompt the same way are still two
samples.

Before this, a teacher and a student replying identically merged into a single
node and a single row:

```
teacher: It is an off-by-one.
student: It is an off-by-one.

4 nodes, 2 leaves, 0 branch points
- user          15ch  EFZ9CR67        -p0000  model=big-model
  * assistant     20ch  FJD2J8ZE
- user          15ch  S7HGG8AW        -p0001  model=small-model
  * assistant     20ch  KH9PSEAP
```

Two roots, two rows, filterable by `model` — which is what a dataset meant for
distilling a large model into a small one, or for training on one of them,
needs.

The model differs from tools in *why* it branches. Tools change the rendered
prompt, so the context genuinely differs. The model does not; it branches
because nothing on the old path was produced by it. Both land at the root, and
between them they make `model` and `tools` single-valued on every path — which
is what removed `conditions_vary` in R21 and now removes the last case where
`model` could be null.

## R24. A runnable distillation workload

`scripts/model_distill.py` is R23 as a workload: the same task put to a teacher
and a student, each working through it in a short loop, which is the shape a
distillation dataset is gathered in.

Turn 1 is pinned to the same reply from both models on purpose — the turn that
would have collapsed into a single node — so nothing except the model can
account for a divergence. Later turns differ as the two models would.

```
teacher-70b: The token expiry check is inverted.
student-7b:  The token expiry check is inverted.

8 nodes, 2 leaves, 0 branch points     -p0000  model=teacher-70b  targets=2
                                       -p0001  model=student-7b   targets=2
```

Turn 1's user *and* assistant nodes share a `message_hash` across the two
models (`5a4e12d7a0`, `d311a9ea14`) and differ in `delta_hash` — word-for-word
the same message, two identities. Both replies are trainable, one per row;
merged, there would have been one generation to train on instead of two.

The export is filterable on `model`, which is the whole point:

```
jq -r 'select(.model == "teacher-70b") | .path_id' samples.jsonl
```

## R25. Asking for tokens from text capture is refused

`--format token-samples` against a text-mode trajectory wrote a valid,
**empty** file and exited zero. That reads like "nothing matched a filter"
rather than "this cannot be produced" — the same class of quiet answer this
batch has been removing everywhere else.

It is now a `400` naming what was asked for, what was found, and what to use
instead:

```
token_samples needs capture in 'tokens' mode, and none of the 1 selected
trajectories is (1 in 'text'). Exact token IDs are recorded only for a target
of type 'tokens'; use text_samples for text capture.
```

**The test is the capture mode, not the record count**, and that distinction
was the whole design question. Zero rows is legitimate on its own: a trajectory
that made no calls exports zero of anything, and a rule keyed on emptiness
would reject that too. A project export mixing modes still succeeds and exports
the `tokens` trajectories.

It is refused where the snapshot is chosen rather than where it is executed, so
no job row is created to go and fail later — the same move as checking
`--output` before running the export.

## R26. Second documentation sweep

The first sweep (R12) checked what could be executed — commands, flags, routes,
schemas. It could not check prose, and prose is where this batch left the most
wreckage. Found by reading, then confirmed against the running system:

**Stale semantics still described as current.** `quickstart.md` explained
`gap ms` as "the observed wait from the previous call's response end", which
has been wrong since R16 made it measure from the *parent* call.
`architecture.md` still described gaps as recomputed with a window function
after each batch — a mechanism that no longer exists, since the value is
derived at read time.

**A claim about where a reward lands.** The quickstart showed
`rlvr_reward=1.0` hanging off a leaf node in `icap graph` output and said the
reward "landed on the leaf assistant node". Both untrue since R3: annotations
are trajectory-scoped, and `icap graph` prints no per-node reward. Verified by
running it.

**A sentence my own earlier edit had mangled.** The README read "in each
trajectory's record, and in every export trajectory's own integrity block" —
a collision between the old text and a replacement, pushed and unnoticed.

**Counts.** "132 tests" in two places (160), "five formats" in the module map
(four).

**Two `trace` invocations the rename missed.** `trace [COMMAND] --help` and
`trace health`: the R7 pass substituted `trace <subcommand>` from a list, and
neither `[COMMAND]` nor `health` was in it.

**Stale field names in examples.** `"tito"` as a node payload key (it is
`tokens`), and `"outcome"` used as an example *annotation* key, which is legal
but reads as the field that was renamed to `command_result`.

**One thing genuinely undocumented rather than stale:** `derivation_version`
is emitted on every `graph` record and appeared in no example.

Also qualified a claim that tools-in-identity made imprecise: "a shared prefix
is stored once" is now "stored once, where shared means the same messages under
the same tools and the same model".

## R27. Finishing the `tito` rename

R14 renamed the user-facing `tito` type and mode to `tokens` but stopped at the
repository's own surface: the `tito/` package, `docs/tito.md`,
`docs/design/tito-parity.md`, `tests/test_tito.py`, the `TitoNode` /
`TitoService` / `TitoError` classes, the `KIND_TITO_EXCHANGE` spool kind, and
the `tito` extra in `pyproject.toml` all still carried a name that means
nothing outside this repository. Half a rename is worse than none: the config
key and the module that reads it no longer matched.

All of it is renamed now. Only `docs/design/token-capture-parity.md` still
says TITO, and only where it names SkyRL's implementation — a citation, not
our vocabulary. A header note says so.

The rename found a real bug. `ui/static/app.js` read `payload.tito` from the
node payload:

```js
if (payload.tito) sections.push(["tito", payload.tito]);
```

The API has returned `tokens` under that key since R14's payload rename, so the
token panel in the UI had been silently empty for every tokens-mode trajectory
since then. No test covered it, because the UI tests assert on the API
responses rather than on what the page renders from them.

Two stale links in this log pointed at `design/tito-parity.md` and would have
404'd on GitHub. Fixed, along with the R14 paragraph that claimed the package
and the parity doc "keep the name" — true when it was written, false now.

## R28. A quickstart that is one, and a documentation index with an order

The quickstart had accreted into a reference manual: 298 lines, with the full
`replay` record schema, the session/`arrival_ms`/`delay_ms` scheduling model,
the derivation of `gap_ms`, the credential-encryption requirements, and SQL for
filtering annotations. All of it correct, none of it *quickstart* — a reader
trying to get running had to read past four explanations of why something is
the way it is.

It is 252 lines now, and every removed passage was checked to exist in the doc
that owns it before it was cut:

| Cut from quickstart | Lives in |
| --- | --- |
| `replay` schema, sessions, `arrival_ms` / `delay_ms` | `exports.md` |
| `gap_ms` derivation and sign | `graph.md#timing`, `api-reference.md` |
| What a repair looks like in the tree | `graph.md`, `exports.md` |
| `CAPTURE_SECRET_KEY`, credentials at rest | `operations.md#credentials` |
| `/healthz` `degraded` | `operations.md`, `api-reference.md` |
| Annotation filtering SQL | `api-reference.md` |
| Aborted trials and lease expiry | `cli-reference.md` |

What replaced the export section is a four-row table naming each format and
what it is for, which is the one thing a new reader actually needs at that
point. Every command and flag in the result was re-checked against `--help`.

**The documentation index had no order.** It listed `api-reference` third and
`graph` seventh — reference before the behaviour it is a reference *to*. It is
now four groups, ordered by what a reader needs when: **start here**
(quickstart, scripts), **what the system does** (graph, exports, tokens),
**how it is built** (architecture, operations, benchmarks, design notes, this
log), **reference** (api, cli, sdk). The surfaces are an appendix, because
that is what they are.

A link checker over every Markdown file confirms all 60-odd relative links and
their anchors resolve.

## R29. A runnable compaction workload

`scripts/` had fan-out, a tool change, and two models. It was missing the one
thing every long-horizon agent does: compaction. `scripts/compaction.py` runs a
seven-turn migration task with two compactions — summarize the transcript,
throw it away, carry on from the system prompt plus the summary.

Compaction is not a special case in this system and is not labelled as one,
which is the point of having it as an example. Ran it and checked the shape
rather than describing it:

```
21 nodes, 3 leaves, 1 branch points
- system        55ch  EF8JRFWW <branch>
  - user          64ch  CQ050FWC        <- the original opening
  - user         208ch  CG8J6HJH        <- rebuilt from summary 1
  - user         158ch  K1BWZE7T        <- rebuilt from summary 2
```

The rebuilt context still starts with the same system message, so the longest
exact prefix is one node deep and the run forks at the root. A long-horizon
agent produces a fan of shallow branches, not a deep spine.

The summary lands in the graph twice and the two copies differ in kind: the
model-authored assistant node that ends one branch (`180ch`) and the
client-authored user node that opens the next (`208ch`, the same text under a
`Summary of the work so far:` header). That is what `author` is for — the same
distinction that separates a repaired assistant message from a second sample.

`text-samples` gives three rows, one per branch, with every model-authored
message trainable and no generation trained twice:

```
msgs=9 trainable=[F,F,T,F,T,F,T,F,T]   authors=[client,client,model,...]
msgs=8 trainable=[F,F,F,T,F,T,F,T]
msgs=6 trainable=[F,F,F,T,F,T]
```

The two paths that end in a summary generation are summarization training data,
which falls out for free rather than needing a format for it.

`replay` gives three sessions, and this is where the honest limit shows:

```
session=s0000 parent=None arrival_ms=0    turns=4
session=s0001 parent=None arrival_ms=169  turns=3
session=s0002 parent=None arrival_ms=325  turns=2
```

No parent session, because a rebuilt context's prefix ends on the *system*
node, which the client wrote — there is no parent model output to hang it from.
So they schedule by `arrival_ms` from the start of the trace. The system knows
when the compacted segment started, not that it started *because* the previous
one ended. Deriving that would mean inferring causality from adjacency, which
R11 ruled out deliberately.

## R30. The graph doc, rewritten around runnable cases

`graph.md` explained the matching rule and then asserted, in a seven-row table,
what it does in seven situations. The assertions were right, but a reader had
no way to check them and no way to see what any of those shapes looks like.

Each row is now a section with a runnable script behind it, in a new
`examples/agents/`, and the tree printed under it is the real output of running
that script under capture rather than a tree written by hand:

| Case | Example |
| --- | --- |
| Ordinary next turn | `next_turn.py` |
| Identical retry | `retry.py` |
| Re-sampled reply | `resample.py` |
| Repaired reply | `repair.py` |
| Compacted history | `compaction.py` |
| Nothing in common | `new_system_prompt.py` |
| Changed tool set | `tool_change.py` |
| Changed model | `model_change.py` |

They share a `_capture.py` that is one `chat()` call and two tool schemas, so
each example is only the story it tells — the longest is twenty lines. The
richer workloads in `scripts/` stay where they are, and the three cases that
have one link to it.

Running them was worth it for more than the trees. Two claims in the doc were
wrong and would not have been caught by reading:

- The retry section said both exchanges "resolve to the same
  `input_leaf_node_id`". True, and they also share `output_node_id`, which is
  the stronger statement and the one that makes the detection exact. Checked
  against the API rather than asserted.
- A sentence I wrote in this same pass claimed `icap graph --json` inlines each
  node's message. It does not — it returns `has_payload` and leaves the message
  in the object store, which is deliberate, since walking a large graph should
  not drag every message with it. The message comes from
  `GET /v1/nodes/{node_id}/payload` or the `graph` export.

Three things were stale rather than new:

- **`gap_ms` was still documented as `next.request_start -
  previous.response_end` in observed order.** It has been measured from the
  call a request *continued from* since R9, which is the entire point of the
  sibling-versus-predecessor distinction the rest of the page makes. The
  paragraph describing gaps being "recomputed for the whole trajectory with a
  window function after each ingested batch" described a mechanism R9 deleted;
  the value is derived at read time now.
- **"See the confidence table above"** pointed at a table R2 removed with
  derived relations. Replaced with the actual rule, and with a pointer to the
  compaction example, where the limit is most visible: the second segment
  plainly happened because the first ended, and the graph will not say so.
- **`design/tito-parity.md` and "TITO mode"**, which R27 renamed.

Two references to internal requirement numbers ("this is parity requirement 6",
"requirement 8") came out. A reader of this page has not read the parity
document and should not have to in order to learn why tools are part of node
identity; the reason is that chat templates render tool schemas into the
prompt, and that reason now stands on its own.

Added: a field table for what a node actually holds, which the page had never
given, and a note on which surface returns which fields.

## R31. The identity section, and five references R27 broke

**Node identity was the hardest section on the page to read.** It opened by
saying `delta_hash` covers the message, tools and model, then four paragraphs
later said "in text mode `delta_hash` is the message hash" — which contradicted
the opening and was simply wrong: text mode has hashed message, tools and model
together since R23. It also interleaved text mode and tokens mode throughout,
so neither was stated plainly, and spent a paragraph on cached-prefix re-render
behaviour that belongs in the token capture document.

Rewritten around the thing the section is actually about: there are three
hashes because there are three recurring questions, one per hash. A first
cut of this called `delta_hash`'s question "is this the same step?" and titled
the subsection after it; "step" was a word the page never defined and did not
need, so both now say what they mean.

| Hash | Covers | Answers |
| --- | --- | --- |
| `message_hash` | One message, on its own | "Is this the same message?" |
| `delta_hash` | That message, and the tools and model it was seen with | "Is this the same message under the same conditions?" |
| `context_hash` | Every delta from the root down to here | "Is this the same conversation?" |

Then one short subsection each: why a step is more than a message (tools and
model, each linking to the worked example above rather than re-arguing it), why
the conversation needs a hash of its own (the identity tuple names a parent node
ID, and node IDs are local to a trajectory — a content hash is not), and what
changes in tokens mode. The chain is spelled out with a worked three-node
example, and the formula was checked against stored data rather than read off
the source:

```
chain formula matches stored context_hash for every node: True
```

**Five broken references, mine.** R27's rename substituted `tito` -> `tokens`
in prose as well as code, which silently turned `design/tito-parity.md` into
`design/tokens-parity.md` inside five docstrings — a file that has never
existed, since the document was renamed to `design/token-capture-parity.md` in
the same pass. In `ingest/parse.py`, `tokens/types.py`, `tokens/service.py`,
`store/graph.py` and `tests/test_tokens.py`. The R27 link check only covered
Markdown files, so it saw none of them; references inside source comments are
checked now too.

## R32. The tool-change case the examples were hiding

`tool_change.py` changed the tool set on the very first message, where starting
a second root is unremarkable — there was no history to share. Reading the page
from that example alone, "a changed tool set starts a new root" sounds like a
statement about first messages.

The case that actually surprises people is a change **partway through**: one
turn under `search`, a second turn continuing that history under `search` and
`edit`. Measured rather than reasoned about:

```
6 nodes, 2 leaves, 0 branch points
- user          20ch  G2KV2ZN9
  * assistant     23ch  YNPD6YY6
- user          20ch  8CTDCR1Y        <- the same two messages, committed again
  - assistant     23ch  KCP780GS
    - user          11ch  MDGHT16E
      * assistant     23ch  MDNF1WK1
```

Still a second root, and the whole prefix is duplicated, because `tools_hash` is
folded into *every* message of a call rather than only the new ones. Now in the
page with its own example, `tool_change_midway.py`, and with the two
consequences spelled out: the prefix is stored twice, and the replayed
assistant on the new root is `author: client`, so that generation still trains
exactly once — from the branch it was produced on.

**The alternative was considered and rejected.** Forking at the turn where the
tools changed, rather than re-rooting, needs the tool set kept out of the
prefix walk and folded only into the model-output node's identity — which means
*matching* has to become asymmetric, because otherwise a replayed assistant
message hashes differently from the node it is replaying and every ordinary
continuation forks. Putting the tools on the preceding user message has the
same defect and additionally forks a level too high, duplicating that user
message. The invariant underneath: a node's identity has to be computable from
the node alone and come out the same every time the message reappears, so a
per-call value is either in every node of every call or out of the walk
entirely. Today's behaviour stands; the cost is prefix duplication, and the
benefit is that `context_hash` alone remains a valid prefix-cache key.

`test_changing_tools_starts_its_own_branch` was renamed to
`..._starts_its_own_root`. It asserts `len(roots) == 2` and always did; the
name was the thing suggesting otherwise.

**`## Anthropic's system prompt` read like a design note**, leading with the
bug it avoids rather than the behaviour it provides. Rewritten as `## Context
that is not a message`: the graph's unit is a message, some providers do not
send the whole context as messages, and anything occupying a position in the
context gets a node anyway. The four `materialized_from` markers
(`system_field`, `instructions`, `input_item`, `response_output`) are now
listed — the section had shown one of them and implied it was the only one.

## R33. A flaky test that was a real race, and an export runner that could die

`test_events_accepted_before_finish_are_not_late` failed once in a full run and
passed in isolation every time. The temptation with a test like that is to add
a retry and move on. It was pointing at a genuine hole in `drain_ingestion`.

**The spool writer takes records off the ring before it writes them.**

```python
records = ring.drain(config.flush_records)      # ring is now empty
await asyncio.to_thread(self.write_batch, records)   # ...but nothing is on disk
```

`drain_ingestion` waited for the ring to empty and then read the spool:

```python
for _ in range(3):
    if not len(self.ring):
        break
```

Between those two lines the records are off the ring and not yet in a segment —
in neither place a reader can see. The window is short, which is why the test
usually passed, and it is real, which is why it sometimes did not. It is not
only a test problem: `finish()` calls `drain_ingestion` precisely so that a
caller who finishes a trajectory and immediately reads it sees a complete
graph, and this could return one exchange short.

`SpoolWriter` now counts records it has taken but not written, and
`drain_ingestion` waits on the ring *and* that count, with a deadline instead of
three fixed turns. Fifteen consecutive runs of the test pass, though the
mechanism is the evidence rather than the count — it passed in isolation before
too.

**`IngestWorker.start()` after `stop()` was a no-op.** `stop()` sets
`_stopping` and nothing ever cleared it, so a restarted worker's `_run` loop
exited on its first check. The worker looked alive and ingested nothing. No
test caught it because the tests that restart a worker then call `drain()`,
which runs `run_once()` directly and does not consult the flag.

**The export runner could die and take every future export with it.** The loop
was:

```python
try:
    identifier = await asyncio.wait_for(self._queue.get(), self._poll_interval)
except TimeoutError:
    row = await export_store.claim_pending(self._runtime.db)   # <- unguarded
```

An exception from `claim_pending` — a transient database error is enough —
raises out of the `except TimeoutError` handler, past the whole `try`, and out
of `_run`. The task ends, and because nothing awaits it the traceback is never
retrieved. The only symptom is exports sitting in `pending` forever.

This was observed, not theorised: the development database had thirty exports
`ready` and then three stuck in `pending` from 04:53 onward, with the proxy,
ingestion and graph queries all healthy. The loop now guards its whole body and
sleeps a poll interval before continuing. A failed export is recorded and the
runner carries on; a dead runner is silent, which is far worse.

Restarting the service confirmed the diagnosis: the three exports that had been
`pending` since 04:53 were claimed and completed within seconds of the new
process coming up, leaving only the two `token_samples` jobs that R25 fails on
purpose. Capture, graph and export were then re-checked end to end against the
restarted service.

**One flake is still open, and it is not this one.**
`test_tenant_cannot_annotate_another_tenants_node` failed twice today — once in
a full suite, once standalone — with a different symptom: a 404 on the
trajectory itself, not a missing exchange. Twenty-five standalone runs produced
one failure, and eight runs at the previous commit produced none, which is far
too few at a rate near four percent to tell whether the ingestion changes here
are involved. Recorded rather than guessed at; the cause is not known yet.

## R34. Two timestamps are the whole record of when a call happened

`overlapping` was a stored boolean saying whether a call's interval met another
call's. It was maintained by three `UPDATE` statements running on every ingest
batch — one to evaluate the new rows, one to propagate overlap onto existing
rows the new arrivals touched, one to re-evaluate a whole trajectory on demand.

It followed entirely from `request_start_at` and `response_end_at`, both of
which are already stored. R9 made exactly this argument about `gap_ms` and
removed the stored column; the same argument applies here and had been missed.

The column is gone, the three statements are gone, and `overlapping` is an
`EXISTS` subquery evaluated when an exchange is read, next to `GAP_MS_SQL`.
There is already an index for it — `exchanges_trajectory_time_idx` on
`(trajectory_id, request_start_at)`. `ingest/timing.py` is 58 lines instead of
145, and what remains is honest to its name: monotonic durations and streaming
cadence, with nothing about the relationships between calls.

The API is unchanged — an exchange still carries `overlapping` — so
`SCHEMA_VERSION` stays at 2. No export ever carried the field, and the only
thing that moved is where the value is computed.

`_settle_trajectory` used to do gap recomputation, overlap detection and
counters. R9 took the gaps, this takes the overlap, and what is left is the
denormalized counters, so it lost its `exchange_ids` argument too: there is no
longer a write set to bound.

**The documentation point is the one worth keeping.** `graph.md`'s Concurrency
section used to justify not deriving pairwise concurrency records on the
grounds that they are quadratic. That is a design-review answer. The
reader-facing statement is that two timestamps per call are recorded and
everything about *when* follows from them — whether calls overlapped, what the
wait was, how many were open at once. Those are queries, not columns, and a
question the shipped fields do not answer needs a query rather than a schema
change.

`## What is deliberately not inferred` was removed on request. Its one
load-bearing idea — that adjacency in time is not evidence of causation, so no
edge is ever drawn from it — moved into Concurrency, where it belongs, since
that section is now about the difference between time and context.

Schema change applied in place to `001_init.sql` rather than as a second
migration, and both databases recreated. The repository has one user and no
deployment to migrate.

## R35. Concurrency and timing leave the graph document

R34 rewrote `graph.md`'s Concurrency section and folded the causality point
into it. Both that section and Timing are now gone: the page is about how
context is derived into a graph, and how long a call took is a different
subject that was only ever sitting next to it.

What the page is left with reads as one argument end to end — one node per
message, the matching rule and its cases, what identifies a node, and the
providers that do not send a flat message list.

`quickstart.md` pointed at `graph.md#timing` to explain why `gap_ms` is
measured from the call a request continued from rather than the previous
arrival. With nowhere to point, the quickstart now says it in a clause: once a
trajectory branches the previous arrival is a sibling rather than a
predecessor, and the wait between siblings never happened. `api-reference.md`
carries the field-level definition, which is where a reader looking up a field
would go anyway.

## R36. exports.md, rewritten for someone using the library

The page was written as a design document: it argued for each decision before
saying what the format was. "An absolute offset bakes one rollout's latencies
into the artifact, so the two fields answer different questions." "That is a
deliberate trade." "Rows are never dropped" as a principle before the reader
knew what a row was.

Restructured around what a reader is trying to do. A table that maps a goal to
a format, then one section per format that opens with the record and follows
with how to read it, then one section on what trains and what does not, then
output and delivery, then checking a finished export. Rationale survives only
where it changes a decision — why each generation trains once by default, why
sampling parameters are in `replay` rather than in a training row.

The examples are from a real export of `scripts/subagent_pipeline.py` rather
than hand-written, and a check confirms every field the page shows appears in
that output. 399 lines to 300.

**One section documented a feature that does not exist.** `### Mooncake`
described `replay` as emittable in Mooncake trace format "when a tokenizer is
named". There is no Mooncake anywhere in `src/`, and `--format` takes exactly
`graph | replay | text-samples | token-samples`. It was a note about a format
we chose *not* to adopt — that is its status in `TODO_1.md` and in R16 — which
had drifted into the reference as a capability. Removed.

Two smaller corrections while checking the page against the code:
`masked_reason` is absent rather than null when a row has trainable messages,
and the three values it can take are confirmed as `abandoned`,
`repeated_target` and `context_length`.

## R37. Every export snippet cut to what distinguishes it

The `graph` example showed a complete record: fifteen trajectory-level fields
and two fully populated nodes, including `sampling`, `usage`, `timing` and body
URIs. Everything in it was accurate and none of it helped, because a reader
meeting the format for the first time cannot see the shape through the
inventory. The same was true of the other three.

Each snippet is now the smallest thing that shows what the format *is*, and
the rest of the fields moved to a list under it.

`graph` is a trajectory that asked one question and sampled the answer twice —
three nodes, four fields each, and `nd_1` carrying two `children`. That fork is
the one thing `graph` has that the other three formats flatten away, and it is
now the first thing on screen.

`replay` is one session with two turns, enough to show requests-only and that
the second turn carries just its tail. `text-samples` is two messages, one
`trainable` and one not. `token-samples` is the arrays and the mask.

The data is from real exports of `examples/agents/resample.py` and
`next_turn.py`, and a check parses every JSON block in the page.

**Two errors this caught.** The `token-samples` mask read `[0, 0, 1, 1, 1]`,
which cannot occur: `loss_mask` is `trainable[first:]` where `first` is the
index of the first sampled token, so it always begins with 1. And a turn delay
was written as a plausible-looking number rather than a measured one; the real
value from the sub-agent workload is 85 ms, which is its 80 ms think pause.

## R38. The token-samples row says what it means

Three things in the sample row were saying something other than what they were.

**A reward field that was really an annotation.** `token_samples` rows carried
`reward`, read from `annotations["rlvr_reward"]` — while the shared envelope
already carried the whole `annotations` map, for both sample formats. So the
value was in the row twice, one copy under a hard-coded key, in a system whose
stated rule is that no field of an annotation is assumed. `text_samples` never
had it. Removed; a consumer reads whatever key it annotated with.

**`--reward` went with it**, along with `reward=` on `finish_trajectory`,
`capture()` and `POST /finish`. Shorthand that privileges one key name is the
same mistake at the input end. `--annotate rlvr_reward=1` is two characters
longer and makes no claim about what a reward is called. One behaviour is lost
with it: `icap run --reward` applied the value only when the child exited zero.
An `--annotate` is applied whatever happens, and `command_result` already
records whether the run succeeded.

**`prompt_token_ids` / `response_ids` was a single-turn idea.** The split was
at the first trainable token:

```python
first = trainable.index(True)
prompt_token_ids = token_ids[:first]
response_ids     = token_ids[first:]
loss_mask        = trainable[first:]
```

On a multi-turn path that names nothing. "Response" holds every later user
message and tool result, masked to zero and sitting between generations, and
"prompt" is only the prefix before the first one — not any call's prompt. The
row is now one `input_ids` sequence with a `loss_mask` of the same length, and
`rollout_logprobs` and `rollout_expert_indices` aligned to the same index. The
tests got shorter: three of them had been reassembling
`prompt_token_ids + response_ids` and re-padding the mask before they could
assert anything.

The per-call `prompt_token_ids` in `tokens/` is untouched. That one is the wire
protocol to the endpoint and is a real prompt.

## R39. `abandoned`, defined

The flag was used in four places and described in one sentence. The rule is
exact:

```python
leaf.author == "model" and len(siblings(leaf)) > 1
```

What was imprecise is what it *means*. Its own docstring asked "did this path
lose a race with a sibling?", and it does not detect that — it detects having
siblings, which is true of every sibling including the one that was kept.
Confirmed against real exports:

```
best-of-N (resample.py)  ->  abandoned = [True, True]
repair    (repair.py)    ->  abandoned = [True, False]
```

So `abandoned` marks one of several endings from the same point, not a
discarded branch. Capture sees two continuations from one context and has no
evidence of which the agent used. A repair is asymmetric and the flag lands
where intuition expects; best-of-N is symmetric and every sample carries it.

That matters before reaching for `--mask-abandoned`, which on a best-of-N
trajectory masks the whole dataset. `exports.md` now states the rule, both
shapes, and that limit. The docstring says the same.

The name still over-promises — it suggests a judgement the system cannot make.
Left as is rather than renamed, since it is in the export schema.

## R40. tokens.md, rewritten for someone wiring up an endpoint

Same pass as `graph.md` and `exports.md`. The page explained the design of the
tokens path; a reader of it wants to stand up a target, satisfy the endpoint
contract, and know what they are getting.

Reordered to that: setup, what is stored per message, what your endpoint must
accept and return, prefix reuse, the fail-closed behaviour that can surface as
a 500, streaming, export, limits. `Per-token message attribution`,
`Serialization` and `Trace rehydration` were three sections about mechanism;
what a reader needs from them is three facts — templates must be append-only,
turns on one trajectory are serialized, and prefix reuse survives a restart —
so they are now one paragraph and one bullet.

Removed the design-review voice: "a deliberate divergence from the PRD's
reliability section" and "the PRD's Phase 4 describes token capture at a level
that omits several requirements" both address a reader who has read the PRD.
Nobody using the library has. The parity document is still linked, as further
reading rather than as justification.

Three corrections found by checking the page against the code:

- **"`--mask-abandoned` zeroes the mask on a branch that lost a race to a
  sibling"** — the phrasing R39 had just established as wrong. Replaced with a
  pointer to the definition.
- **"parity requirement 12"** — the same internal reference that came out of
  `graph.md`, still here.
- **A claim I wrote in this pass and then checked**: that the exported
  `loss_mask` "is exactly the concatenated `sampled_mask`". It is the
  concatenation *except* where the train-once rule or `--mask-abandoned`
  suppresses a node, which contributes zeros instead. Corrected before it
  shipped.

The upstream contract also gained the `model` field, which the proxy sends and
the page had omitted. 243 lines to 212.

## R41. What the reference implementation actually does about tokenization

A question about pointing a tokens target at vLLM turned into a design review,
because the premise was wrong and worth correcting in writing.

**In tokens mode the engine does not apply a chat template at all.** The proxy
POSTs `prompt_token_ids`; vLLM or SGLang consume tokens verbatim. So there is no
"the engine's templating" for a local renderer to diverge from at serving time.
The real risk is narrower: our render has to match the template the *model was
trained with*.

I prototyped an `EngineRenderer` that would render through the engine's
`/tokenize` endpoint, and then checked whether that is a good idea. It is not,
for three separate reasons:

| | |
| --- | --- |
| vLLM `/tokenize` | Does take `messages` and `add_generation_prompt`, so it would work — at one HTTP round trip per message per turn, on the synchronous fail-closed path |
| SGLang `/tokenize` | Takes a raw `prompt` string and `add_special_tokens`. No messages, no chat template. The approach cannot work there at all |
| SkyRL | Does not do it. Renders locally |

Reverted.

**What SkyRL's token-in/token-out proxy (PR #2143) actually does** is render
locally through [Prime Intellect's `renderers`](https://github.com/PrimeIntellect-ai/renderers),
pinned to a git revision. Its `TITORenderer` protocol is ours method for method
— `render`, `bridge`, `parse_response`, `get_stop_token_ids`, `decode_token`,
over a `RenderedPrompt(token_ids, message_indices, reused_prefix_length)` —
which is unsurprising, since the parity document was written from that PR. The
architecture matches. The renderer behind it does not.

Comparing `PrimeRendererAdapter` against our `HFChatTemplateRenderer` found
four real gaps, and the second and third were silent:

- Attribution: native `message_indices` from the library, against ours
  recovered by incremental diffing and requiring an append-only template.
- **Reasoning.** They configure `thinking_retention="all"` and return
  `reasoning_content`. Ours decodes with `skip_special_tokens=True` into a flat
  `content`. For a reasoning model, replaying that assistant turn re-renders
  differently from the tokens that were sampled, the token-space check refuses
  it, and the graph gets a branch where it should have had prefix reuse.
- **Tool calls.** They parse them with a status per attempt and emit structured
  `tool_calls`. Ours leaves a tool call as raw text in `content`, so a client
  replaying it as structured `tool_calls` renders differently — same silent
  branch.
- Bridging: theirs bridges and then verifies the reused prefix survived
  exactly. Our HF renderer returns `None` unconditionally, so prefix reuse
  never engages for any real model.

So the fix was never an engine renderer. `PrimeRenderer` now backs our protocol
with the same library, selected by a `prime:` prefix on a target's tokenizer
(`--tokenizer prime:Qwen/Qwen3-8B`) under a new `prime` extra. It uses
`create_renderer_pool` rather than a lock, which is the library's own answer to
concurrent trajectories, and it verifies the bridge prefix the same way SkyRL
does. Selection is explicit rather than "use it if installed", because a silent
downgrade to a thinner renderer is exactly the failure this exists to avoid;
asking for it without the extra installed is an error naming the command.

Eight tests cover it against an injected stand-in for the library, so the suite
still needs nothing installed: the `prime:` prefix reaching the pool, thinking
retention on by default, the generation prompt being requested, reasoning
content surviving onto the message, only cleanly-parsed tool calls becoming
structure, and a bridge that rewrites its prefix being refused.

Two things to know. The extra pulls eleven packages (`openai`, `tiktoken`,
`openai-harmony`, `httpx2` among them), and `uv sync --all-extras` installs
them. And the library reports a per-token `sampled_mask`, where our node stores
a single `sampled_start` boundary — which assumes an assistant message is
scaffold-then-sampled, contiguous. A template that injects scaffolding *inside*
an assistant turn breaks that assumption, and adopting the mask would fix it.
Not done here.

`HFChatTemplateRenderer` still has no test of its own.

## R42. architecture.md, for someone deploying it

The page was organised around the arguments for each decision -- "The
constraint that shapes everything", "Why the data plane has no framework",
"The spool is a byte contract, not an API", "Ordering in the write path".
Every heading was a position being defended.

Reordered around what a reader needs to do: what happens to one request, what
happens when something fails, what runs where and how to scale it, how to add
an upstream, where the code lives. The rationale survives where it changes a
decision -- why a credential is bound to one route, why one worker owns a
trajectory, why a certificate check cannot be skipped -- and goes where it was
only justification.

Three ASCII diagrams became mermaid: the component flow, a sequence diagram of
one request through the data plane and out to ingestion, and the single-process
against scaled-out deployment shapes. Checked the fences balance and that no
subgraph is left open; HTML in labels was dropped, since whether it renders
depends on the host's mermaid security level.

**Two things the page had wrong.** The module map still credited
`ingest/timing.py` with "signed gaps, overlap detection"; R9 moved the gap to
read time and R34 did the same for overlap, so it does durations and streaming
cadence and nothing else. And the fail-closed section still called tokens mode
"a deliberate divergence from the PRD", which addresses a reader who has read
the PRD -- the same thing that came out of `graph.md` and `tokens.md`.

## R43. Tokens mode scales horizontally now

Text mode always did: the data plane holds nothing per trajectory, and
ingestion decides the graph. Tokens mode did not, and the way it failed was
the bad kind.

Reproduced it first. Two replicas on one database, turn 1 to one and turn 2 to
the other, with the first replica's disk queue deliberately undrained:

```
depth=0 role=user      author=client
depth=1 role=assistant author=client   <- the model sampled this
depth=2 role=user      author=client
depth=3 role=assistant author=model
```

The second replica had no trace for the trajectory, rebuilt one from the graph,
and the graph did not have turn 1 yet. So the generation came back as something
the client replayed. It is not trainable, and it is gone. The trajectory did
report `complete: false` with `accepted_count` 2 against `exchange_count` 1, but
`calls_missing` stayed 0 — nothing was dropped, it was sitting in a queue
nobody was reading — so a dashboard watching that counter saw green.

**Affinity.** `proxy/ring.py` is a consistent hash over replicas keyed by
trajectory ID, and a tokens turn that arrives on a replica that does not own it
is relayed to the one that does. Measured over 20k IDs: adding a ninth replica
to eight moves 10.9% of trajectories where `hash % replicas` moves 89.0%, and
every moved trajectory is a trace to rebuild. Virtual nodes went from 128 to
1024 after measuring the split — the worst member sat 16% off even at 128 and
under 7% at 1024, for about 0.1 us per lookup. Skew matters more here than for
a stateless cache: the overloaded member holds the traces too.

Proved the relay rather than assuming it. Three turns sent to the replica the
ring says is wrong:

```
replica 8080: requests_served=3 turns_forwarded=3
replica 8081: requests_served=3 turns_forwarded=0
6 nodes, 1 leaves, 0 branch points   (all three assistants model-authored)
```

`turns_forwarded` is on `/healthz`, and a number that keeps climbing means the
load balancer is not hashing on the trajectory, so every turn pays a hop.

**The guard, because affinity cannot cover everything.** A turn can still
arrive somewhere that has genuinely never seen the trajectory — during a
rollout, or just after the ring changes. Rebuilding a trace now checks whether
the trajectory has captured turns that are not stored yet, waits briefly for a
worker that is merely behind, and returns `503 tokens_trace_unavailable`
instead of building on a graph it knows is short. The same experiment that
produced the silent `author: client` above now returns:

```
turn 2 -> proxy A: 503
  code:    tokens_trace_unavailable
  message: 1 earlier turn(s) of this trajectory are captured but not yet stored
```

Two things went wrong while building it, both caught by running it. The relay
first reused `forward.HOP_BY_HOP`, which strips `authorization` — correct when
forwarding upstream, where the capture credential is swapped for the real one,
and wrong for a peer that needs it to resolve the lease; the relay has its own
list now. And it read `response.status` and `response.body`, which the
transport protocol does not have.

---

# The write-path rework — September 2026

Eight items, in the order the design note put them, and the order matters: the
benchmark before anything touched the write path, the deletions before the
rename, the hierarchy before the file format that has to encode it.

The theme is subtraction. The four items that changed the system removed
roughly 1,500 lines of it — two tables, a credential system, a store module,
a cache — and the two that added anything added a reader and a checker.

## 0. Baseline the benchmark

A run, not a build, except that the token arm could never have run: the load
generator built its completion URL by joining the endpoint to the trajectory
id, which stopped being the route when the data plane moved under `/route/`.
Every proxy arm 404'd on turn 0. It uses the `base_url` the create response
hands back now — the only party that knows the prefix.

`docs/design/benchmarks-rework.md` holds the numbers, and item 6 holds the
comparison.

## 1. One process, one upstream

The largest deletion. A capture process serves exactly one inference server,
named on its own command line; changing it means relaunching.

Gone: the `targets` and `dataset_sinks` tables and their CRUD, `tenant` from
every table and every store signature, the control key and its keyring,
`CredentialCache`, `CaptureService.ensure_target`, and the AES machinery in
`secrets.py` — there is nothing at rest left to encrypt, so `cryptography`
moved to the dev extra.

What replaced them is smaller. `UpstreamConfig` is a frozen dataclass read from
flags and the environment. `LeaseRecord` lost six denormalized target fields,
because the request path reads the running configuration rather than a snapshot
that could be drained out from under it.

**Authentication was decided, not dropped.** The control plane is
unauthenticated: capture is an ephemeral in-cluster job, reachable only from
inside its own network, and there is nothing left to administer over it. The
credential that does real work is the per-trajectory lease on the data plane —
single-use, bound to one route, revoked by finishing. `control/app.py` and
`operations.md` both say so in full, because a reader finding no auth should
find the reasoning next to it.

The schema was rewritten in place rather than migrated: the tenant columns were
in every table and every index. A pre-rework database is detected in `migrate()`
and refused with a message, rather than skipping every migration and failing on
the first insert.

## 2. skyrl-capture

Package, distribution and command all become `skyrl-capture`; `ICAP_` gives way
to the `CAPTURE_` prefix the rest already used. Done straight after item 1 so
nothing about to disappear got renamed.

The install split folded in here, because it is the same pyproject edit.
`skyrl-capture` is text-mode capture with **no ML dependency at all**, and
`skyrl-capture[tokens]` is the only thing that brings a renderer — which
matters because capture installs beside a trainer that pins its own
`transformers`. That only holds while every renderer import stays inside the
function that builds one, so it is asserted twice: a test that imports the
runtime and fails if `transformers` appears in `sys.modules`, and a CI job that
installs the wheel with no extras. Every other job runs `--all-extras` and
would never notice.

## 3. project / run / trajectory

A run is one execution inside a project; a trajectory is one attempt inside it,
naming the task it attempted and the step that produced it. `run_id`, `task_id`
and `step` are indexed columns rather than annotations, because every question
an RL run is read with groups by them.

`GET /v1/runs/{id}/grid` answers the whole task × step view in one request. Two
decisions in it are worth keeping: a cell with several attempts reports their
mean and says how many, and a pair that was never attempted is **absent** rather
than zero — those are not the same thing, and a grid that drew them alike would
be lying in the one place it is read fastest.

## 4. The graph, in the writing process

Text mode read its graph back from SQL to extend it; it holds one `GraphIndex`
per live trajectory now, built from empty and never re-read. Storage did not
change, which is the point: verify the new builder against the old storage,
then move the storage separately. `tests/test_graph_in_memory.py` compares them
node for node across linear, branched, compacted, retried and restarted cases.

The comparison found two real bugs, which is the argument for writing it:

- **An identical retry corrupted the in-memory index.** `insert_node` returns
  the existing node id, and adding it again appended a duplicate child, doubled
  the parent's fork count, and re-attributed the node to the retrying exchange.
  Storage kept the original, so the two silently disagreed. `add` is idempotent
  on the node id now, and graph attribution stays a fact about the exchange that
  first introduced a node. The fork counter it corrupted had no readers, so it
  is gone rather than fixed.
- **Nothing released a graph for a trajectory that finished cleanly.** No later
  batch touches it, so a worker held every graph it had ever built and memory
  grew with the run rather than with the trajectories in flight.

## 5. The file-backed record

`--record-dir` writes each finished trajectory into a versioned, run-partitioned
directory that the exporters and the viewer read with no database and no server.

The design decision that makes it cheap: the record is written by serializing
the same `TrajectoryView` the exporters consume, and read by reconstructing one.
So "all four formats work off a directory" is true by construction, and a
record-backed export is byte-identical to a database-backed one — asserted, not
assumed.

This is deliberately **not** `db=file://`. PostgreSQL stays the live store;
leases, idempotency, the single-writer graph and the export queue all need
transactions, and the design note that asked for this also said not to port the
store layer. The record is the artifact, not a second backend.

A smoke test found two bugs, both about the reward, which is the thing a record
exists to carry. It was written *before* the reward landed, because draining can
finalize a trajectory and `finish` applied its annotations after the drain. And
a reward annotated later never reached the record at all. Both fixed; the index
became a log, where the last line for a trajectory wins, which is what lets a
late reward be a correction rather than a duplicate.

## 6. The benchmarks again

Capture-on throughput is flat to slightly up: 3405.9 rps before, 3432/3438/3486
across three runs after. Token per-turn overhead is unchanged at +9 ms p50,
prefix reuse is 80.0%, every integrity counter is zero.

The finding worth more than the numbers: **the harness's own ratio is noise at
this size.** It reports capture-on as a percentage of capture-off, and the
capture-off arm is the noisy one — 3227–3831 rps across four runs, ±9%, seven
times the variance of the arm being measured. The ratio swung from 89.6% to
108.0% with no code change between the last two runs.

## 7. The viewer

What "complete" means was written down first, in `docs/viewer.md`, because this
is the largest surface here and the one most likely to grow without a boundary.

The **task × step grid** is what the viewer is for, so it was built first. The
other half is **token inspection**: `GET /v1/trajectories/{id}/paths` returns
one entry per root-to-leaf path — a path, because that is what an export row is
— decoded into blocks of four kinds, not two. `replayed` is the one that
matters: assistant text the model did not produce, which in the counts is a node
with zero sampled tokens and indistinguishable from a tool result.

`skyrl-capture view ./traces` serves the same page off a record directory.
Building it found that the retirement sweep walked the graphs a worker holds,
and in tokens mode the proxy commits the graph — so no tokens trajectory was
ever retired or written to the record.

## 8. The comparator and the second verification layer

The two things `vs_miles.md` concluded were worth borrowing.

`TOKENS_AUDIT_PREFIX` returned a boolean: something diverged, and nothing to act
on. It classifies now — `length`, `scaffold`, `sampled`, `given` — and names the
node, the prompt offset and a decoded window either side. The classes are ours
rather than Miles's: theirs must tolerate `ASSISTANT_TEXT` because it compares
against a canonical re-render, and we never re-render a committed turn, so every
class here is a bug.

`skyrl-capture verify` is the layer the first one cannot be: it hands a
record's prompts back to the engine, greedily, and compares the completions.
That is the only check that can catch a record which is self-consistent and
still not what the engine saw, because the thing being checked is a boundary the
capture process cannot see across. Against the mock engine, whose completion is
a deterministic function of its prompt, it is the same claim at CI cost, and it
runs on every push.

**And it was run on a GPU.** Qwen3-4B-Instruct-2507 on an H100, greedy: 9
trajectories, 12 paths, 540 completion tokens reproduced exactly. Then the
negative control, which is the half that matters — one prompt token rewritten
in one record, `525` to `9999`, and the same run failed at completion token 5
of 39 with exit 1.

Getting there needed a piece the docs had assumed: `vllm serve` speaks the
OpenAI wire, not this one, so there was nothing to point `verify` at.
`scripts/vllm_token_engine.py` is vLLM behind the `tokens` wire in sixty lines
— a reference engine for verification, not a serving stack.

The run also exercised the viewer against real weights, and produced the
example `ui_requirements.md` was written about: a `replayed` block reading
`<|im_start|>assistant\nI decided to skip that step.<|im_end|>` sitting between
`given` context and a `sampled` completion, visibly a different thing from
both.

---

# Per-trajectory persistence — September 2026

The design is [design/durability.md](design/durability.md). What it replaced:
one global event log, one `LiveState` reducer over every trajectory in the
process, and one process-wide `Recorder` between them.

That shape had three consequences worth naming, because they are what the
change is for.

**A record was a replay.** Reading one meant applying every event of every
trajectory through the reducer to rebuild the state — so opening a finished
trajectory cost the whole run, and a viewer's first page cost it twice.

**Durability was a global sequence.** One queue, one fsync cursor, one number
that said how far behind the disk was. It made "is *this* trajectory safe?"
unanswerable, and it made two capture processes sharing a directory impossible:
their sequences would interleave.

**Persistence was optional.** `--record-dir` was a flag, and without it a run
left nothing. The default was the case where the run's whole output is lost.

## What it is now

One hot `ActiveTrajectory` per trajectory in local use. One append-only journal
per trajectory, named by a stable hash of its id. One compiled
`TrajectoryRecord` per finished trajectory, written once and read without a
reducer. One `RecordReader` that prefers the committed form and replays the
journal otherwise. Persistence is mandatory.

Nothing coordinates across trajectories, which is what lets several capture
processes share a directory — under the external invariant that routing gives
each trajectory exactly one writer. Storage does not check that and could not:
two writers interleaving individually-valid records is undetectable by
construction. `tests/test_persistence.py` asserts that as the documented
position rather than passing off a check that does not exist.

## The two contracts, made explicit

Text capture commits **behind** the response. Nothing about a forwarded request
waits for a disk, and a failure is a gap the trajectory carries — marked in
memory at once, and written as its own record on the next append that succeeds,
so a volume that comes back records what was lost while it was away.

Token capture commits **before** the response closes. The append is queued as
soon as the outcome exists and overlaps the send; only the final empty frame
waits on it. A failure raises instead of closing cleanly, which the client's
ordinary retry of the unchanged request handles — and is why TITO needs no
request id and no header.

Measured on the mock engine: 0.56 ms per turn spent waiting for durability at
the close, against an upstream phase of 0.45 ms and a render of 0.13 ms.

## The ambiguity that is left, and what is done about it

Persistence and network delivery cannot be atomic. A process that dies between
them leaves an exchange that is durable and may never have arrived, and no
request id would help: the client's retry is byte-identical to intentional
resampling, which is a thing an RL harness does on purpose.

So it is not guessed at. A durable exchange starts `delivery_confirmed=false`;
a second record confirms it after the ASGI send completes; a recovery flags an
unconfirmed one `delivery_uncertain`. It is resolved by evidence and only by
evidence — a later request whose own message history contains that assistant
output, which confirms it and reuses its exact token path. The converse is
deliberately not inferred.

Three readers, three different answers, all of them stated: the graph shows an
uncertain exchange flagged, training exports mark it untrainable with
`masked_reason: delivery_uncertain`, and replay leaves it out.

Text capture's equivalent doubt is trajectory-wide, because nothing identifies
which exchange a dead process might have been mid-commit on. A replacement
writes a zero-count `CaptureGap` — uncertainty rather than loss — so a viewer
reading the journal sees what the writer knows rather than only what the writer
remembers.

## What went, and what it cost

Gone: the global event log and its segments, `LiveState` and its reducer, the
`Recorder`, the sink protocol, the disk reader, the live reader, startup
replay, the expiry sweep and `TRAJECTORY_TTL_SECONDS`, `DELETE
/v1/trajectories/{id}`, `PATCH /v1/runs/{id}/metadata` and the run record
behind it, process-local idempotency keys, and memory-only capture.

Runs are derived now: the grouping their trajectories imply, computed on read.
There is no counter to keep in step with its members and no way for a run and
its trajectories to disagree.

Idempotency is by content rather than by key. Creation is idempotent by the
trajectory id and a hash of the body; finishing by a hash of the outcome. Both
hashes are on disk, so a retry after a restart reaches the same answer — which
a process-local key table could not, and which was the documented behaviour
being accepted rather than fixed.

`finish` is synchronous and returns `200`: by the time it answers the record is
committed, and the reply is the trajectory rendered from it in whichever of the
four formats was asked for. An RL harness scoring a rollout gets its training
rows in the call it was already making, with no job to poll.

## The viewer, split off

Reads and bulk exports are their own app now, mounted by default and removable
with `--disable-viewer`. A fleet runs one replica with it and the rest without,
because there is one indexer per record directory.

It reads files. A capture replica serving the viewer does not show what a proxy
is holding; it shows what survived, which is the point — a graph on screen that
a crash would erase is worse than one a second behind. Indexing is progressive:
the first page answers at once, `indexing` says the scan is still running, and
`total` is null until there is a stable one, because a pager prints whatever
count it is given as fact.

A record directory changes while it is read, so the viewer polls both
directories and the open trajectory. The assumption that a filesystem record
never changes was true of a finished export and is not true of this.

One concession to being co-hosted: when the capture process is also the viewer,
the writers tell the local index which trajectory they just changed. It is not
a change feed and not a subscription — the reader still reads the files — but
it means a replica serving reads for the directory it is writing is not a sweep
behind itself.

## What the goldens said

The behavioural goldens are what make a change this size reviewable. After the
rewrite the diff against them was **schema-only**: `schema_version` 3 to 4, the
new `integrity` fields, `revision`, the removed `accepted_count` and
`expires_at`, and the run listing losing metadata it no longer has. Two
behavioural diffs surfaced, and both were read rather than accepted:

- A poisoned trajectory was being marked `complete: false`. It is not: the turn
  that caused the poison was captured exactly, and so was everything before it.
  What ended is the trajectory, which `status` says, and why, which `errors`
  says. Reverted.
- A non-streaming TITO exchange reported `chunk_count: 0` where text mode
  reports `1` for the same shape. Now `1` in both. A fix, kept.

## Review: five failure boundaries

A review of the rework found five real defects, all of them in the space the
test suite did not reach -- not instability, but the cases where something else
has already gone wrong. Each is worth stating because in every one the wrong
behaviour is a quiet, plausible answer rather than a crash.

**A lost exchange took the graph with it.** An exchange's journal record
carries only the nodes that exchange introduced, so a refused or failed append
left every later record referencing ancestors the journal did not contain. The
live graph stayed whole and the response was served, so nothing surfaced until
somebody recovered or exported -- as a path that starts in the middle of a
conversation. The exchange row is expendable; the graph is not, so the nodes of
a lost record are now held and attached to the next record that can carry them.

**`finish` could report a completeness it did not have.** Past the grace it
logged a warning and carried on: compiled, committed, evicted, deleted the
journal. The turn still in flight then had its append silently dropped, and the
record said `complete: true` with the exchange nowhere. In TITO that meant a
response closing cleanly on an exchange that was never written, which is the
one thing the design says cannot happen. An append to a finished journal now
fails rather than being dropped, and the turns the grace did not wait for are
counted as missing before the compile.

**Lifecycle operations were not serialized.** Create, finish and metadata each
have awaits in the middle, so an annotation could be acknowledged and then
deleted with the journal `finish` was about to remove, and two rewrites of a
committed record were an unlocked read-modify-write through one PID-named
temporary file. Per-trajectory lock, unique temporary names.

The lock alone did not fix it, which is the part worth recording. Making the
race bite every run rather than sometimes -- eight trajectories finishing and
being annotated at once -- showed the annotation still failing, because the
journal's deletion happens outside the lock and `resolve` would happily adopt
a journal whose record was already committed. That is a worse bug than the
race: a process that died between the commit and the delete left a finished
trajectory writable again, and every append to it failed. Readers already
preferred the committed form; the write path does now too, and removes the
redundant journal when it finds one. The lock is reference-counted rather than
left in a map, because a long run names hundreds of thousands of trajectories.

**A standalone viewer never saw a record rewritten.** Once an id entered the
committed index it was skipped for ever, so a reward arriving after a
trajectory finished was invisible to any reader that did not share a process
with the writer -- which is the deployment the viewer split exists for. The
index compares size and modification time now.

**A bulk export could be created against a half-built index**, fixing a
selection that silently omitted whatever the background scan had not reached.
It waits for one complete pass; a listing may be a scan behind and a dataset
may not.

Each fix has a test that fails without it, checked by reverting the fix and
watching it fail -- including the two races, which were made to bite every run
rather than sometimes: eight trajectories finishing and being annotated at
once, and six concurrent rewrites of one record.

## Review, second pass: the barrier and the retry

Two findings, and the first one retracted a choice made in response to the
first review.

**`finish` past its grace could not account for what it was missing.** The
previous pass made it compile anyway and count the difference as
`calls_missing`. That cannot be made correct, and the second review showed
three ways over: a TITO turn awaiting its own commit is *both* an open turn
and a pending commit, so it is counted twice; `settle` shields the append, so
it keeps running past the timeout; and `flush_gaps` then waits behind that same
append on the store's per-trajectory lock, with no bound. If the append lands,
the journal read that follows picks the exchange up -- and the committed record
holds the exchange while declaring it lost.

There is no number to put there, because the thing being counted has not
stopped happening. So the barrier is a barrier: either wait ends in `503`,
retryable, trajectory untouched. Both reviews offered this as the first option
and it is the one that survives. A trajectory that cannot finish stays open --
which nothing expires -- and the retry after a restart is trivial, because the
hung turn went with the process that held it.

**A retried finish applied its metadata twice.** After `FinishRequested` was
written and the commit failed, the trajectory sat in `finalizing`, which is not
a terminal status, so the retry re-ran steps 1-2: the same annotations merged
again and a second revision counted for a correction nobody made, plus a second
`FinishRequested` for any replay to apply. The persisted hash now says "this
exact finish is already recorded" and those steps are skipped. The hash has to
*match*, not merely be present: a trajectory marked `finalizing` by a poison
has not had this finish's metadata applied, and skipping would drop it.

Both fixes have a test that fails without them -- `200 == 503` for the first,
`2 == 1` revisions for the second.

### And a segfault underneath both of them

Making the finish race bite every run -- twelve trajectories finishing at once,
each compressing its record in its own worker thread -- turned up
`ZstdError: cannot compress: Src size is incorrect`, and a direct reproduction
segfaulted the interpreter outright.

`compression.py` held one module-level `ZstdCompressor`. A compressor carries
internal state and cannot be used by two threads at once, so sharing one was
never a slow path; it was a crash waiting for enough concurrency to find it.
The bug predates this work -- the file is untouched by it -- but per-trajectory
persistence is what made it reachable: journal appends and committed records
both run in `asyncio.to_thread`, and per-trajectory locking exists precisely so
that two trajectories write at the same moment.

Thread-local compressors now, rather than per-call (a context allocated per
record on the write path is a cost paid for nothing) and rather than a lock
(serializing every compression across the process would put trajectories back
in each other's way). The test asserts the fix rather than reproducing the
crash: a test that segfaults takes the runner with it and reports nothing.

### The protocols were not the whole contract

A third review pass found `CommittedStore.exists` called by the registry and
never declared on the protocol -- so a PostgreSQL implementation could satisfy
the interface and fail at runtime. Checking the rest of the boundary found the
same thing again, worse: `finish` read a trajectory's journal by asking its
`ActiveStore` for a **filesystem path**, which nothing but the disk store could
ever have given it.

The second one is the interesting failure, because it is not a missing
declaration but a leaked implementation detail. Reading a journal back is a
product operation -- `records(trajectory_id)` -- and it is on the protocol now.
`CaptureCommands` takes `CommittedStore` rather than `DiskCommittedStore` while
we are here, since typing it concretely undid the point of having the interface.

A `Protocol` does not enforce this: nothing stops a caller reaching for a method
the implementation happens to have. So the lifecycle now runs end to end --
create, capture, finish, annotate -- against two dictionaries implementing
exactly the declared operations and nothing else. It catches both original gaps
when they are put back, one at a time, and it is what will catch the next.

The shared exchange fixture grew into a complete derived row along the way. It
had been minimal, so a renderer reading a field it lacked failed as a `KeyError`
from inside an exporter rather than as anything a reader of the test would
recognise.
