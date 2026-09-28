# Operations

## Deployment profiles

| Profile | Results | Shape |
| --- | --- | --- |
| Local dev / RL | A record directory | One process, nothing to set up |
| Self-hosted | One record directory, shared | Replicas behind a load balancer that hashes the trajectory id |

**There is no database.** Every trajectory is a file the capture process
writes as it runs. Nothing is installed, migrated or started first.

### Local, and embedded in an RL run

```bash
skyrl-capture serve --record-dir ./traces
```

One directory holds the result:

```
traces/                  the result -- keep this one
  manifest.json          versions, and the upstream, credential-free
  active/                a journal per trajectory in flight, CRC-framed
  committed/             a compiled record per finished trajectory
  exports/               jobs and artifacts, if you ran any
```

`--record-dir` says where it goes, and exports land beside it. One process runs
the proxy, the lifecycle API, the writer and — by default — the viewer.

Startup is a few seconds, nearly all of it loading the tokenizer in tokens
mode. There is nothing to download and no `initdb`.

**`--record-dir` is required.** Persistence is not a mode: a process with
nowhere to write is a process whose whole output is lost, so it refuses to
start rather than discovering that at the first request.

### Scaled

```bash
skyrl-capture serve --record-dir ./traces        # one pod
```

**Writing happens in the serving process, always.** There is no flag for it:
one pod parses and derives on one core, and what scales is pods.

- **Capturing several inference servers** is several processes, one each. That
  was always true.
- **Capturing more than one core's worth** means replicas of one pod, and the
  routing rule below stops being optional.

Replicas **may share one `CAPTURE_RECORD_DIR`** — a local directory or an RWX
volume. A trajectory's files are named by its id, so two replicas never write
the same file as long as routing gives each trajectory one writer. What the
filesystem has to provide is coherent append and read, a meaningful `fsync`,
and atomic same-directory rename; an NFS mount without close-to-open coherence
is not a place to put this.

**Exactly one replica serves the viewer.** There is one indexer per record
directory, so start one with the default `--viewer` and the rest with
`--disable-viewer` — or run a standalone `skyrl-capture view --record` against
the shared directory and disable it everywhere.

`PUBLIC_URL` must be the address workloads can reach, since it is what
`create_trajectory` hands back.

**With more than one replica, the load balancer must pin a trajectory to one of
them.** Without it, two replicas derive the same trajectory's graph from
divergent views and it grows branches the workload never took — with no error
anywhere. That is the next section.

## Routing: pinning a trajectory to a replica

**This matters the moment you run more than one capture replica**, in either
mode, and getting it wrong is silent in text mode. See
[architecture.md](architecture.md#a-trajectory-must-stay-on-one-replica) for why.

The rule: **hash on the trajectory id, which is the segment after `/route/`** in
`/route/{trajectory_id}/v1/...`.

> **Do not hash on the whole URI.** One trajectory can hit more than one
> endpoint — `/route/tr_abc/v1/chat/completions` and `/route/tr_abc/v1/embeddings` — and
> whole-URI hashing sends them to different replicas, which is the exact failure
> the rule exists to prevent. Hash that one segment only.

### Local: you usually need no load balancer at all

Affinity only matters with two or more replicas. For ordinary local work, run
one:

```bash
skyrl-capture serve                 # one process, everything
```

That is the profile the test suite and `docker compose` use, and nothing about
it needs routing.

### Local: two replicas, when you want to exercise multi-replica behaviour

Worth doing before a production rollout, because it is the only way to catch a
routing mistake on your laptop rather than in a graph a week later.

```bash
skyrl-capture serve --port 8000 --record-dir ./traces-0
skyrl-capture serve --port 8001 --record-dir ./traces-1
```

Each replica needs **its own record directory**, because a log has one writer.

There is nothing to share. Each replica holds its own state and owns its own
trajectories, which is what the routing rule makes true -- and what makes
getting the routing wrong so quiet, since a misrouted turn lands on a replica
with no memory of the trajectory rather than on a second writer. Then HAProxy
in front:

```
# haproxy.cfg
defaults
    mode http
    timeout connect 5s
    timeout client 5m
    timeout server 5m

frontend capture
    bind *:8080
    default_backend pods

backend pods
    # Hash the trajectory id: the third "/"-delimited field of the path.
    # ("/route/tr_abc/v1/..." splits to ["", "route", "tr_abc", ...], so field 3.)
    balance hash path,field(3,/)
    hash-type consistent
    server pod-0 127.0.0.1:8000
    server pod-1 127.0.0.1:8001
```

```bash
haproxy -f haproxy.cfg
# or: docker run --rm -p 8080:8080 \
#       -v "$PWD/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro" haproxy:2.9-alpine
```

`balance hash <expression>` needs HAProxy 2.4 or newer. `balance uri depth 2`
is the older equivalent -- depth 2, not 1, because `/route/` is now the first
segment.

Point workloads at `http://127.0.0.1:8080` and check both properties:

```bash
# One trajectory across several endpoints must land on one replica.
for ep in v1/chat/completions v1/embeddings v1/responses; do
  curl -s localhost:8080/route/tr_aaa/$ep
done
# Many trajectories must spread.
```

Measured with this config: one trajectory stayed on one replica across all
three endpoints, and 60 trajectory ids split 32/28 across two replicas.

nginx is equivalent if you prefer it — capture the segment, then hash the
capture rather than the URI:

```nginx
upstream capture {
    hash $trajectory consistent;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}
server {
    listen 8080;
    location ~ ^/route/(?<trajectory>[^/]+)/ {
        proxy_pass http://capture;
    }
}
```

### Kubernetes

The pod is one container running the proxy and the writer, one volume for the
record, and a grace period long enough to flush it.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: capture
spec:
  replicas: 3
  template:
    spec:
      # Long enough for shutdown to drain the queue into the record. That is
      # the queue depth over the writer's throughput -- seconds, normally.
      terminationGracePeriodSeconds: 60
      volumes:
        - name: records
          persistentVolumeClaim: { claimName: capture-records }   # the log, and exports beside it
      containers:
        - name: capture
          image: skyrl-capture:latest
          args: ["serve", "--record-dir", "/records"]
          ports: [{ containerPort: 8080 }]
          env:
            - { name: PUBLIC_URL, value: https://capture.example.com }
          volumeMounts: [{ name: records, mountPath: /records }]
          resources:
            # ~1 core proxy + ~1 core deriving and writing, measured.
            requests: { cpu: "2", memory: 2Gi }
          # Both probes answer one question: is the proxy serving? Neither is
          # wired to capture health -- that would turn a capture problem into a
          # serving problem. Capture health goes to metrics and alerts.
          livenessProbe:
            httpGet: { path: /healthz, port: 8080 }
          readinessProbe:
            httpGet: { path: /readyz, port: 8080 }
          lifecycle:
            preStop:
              exec: { command: ["sh", "-c", "sleep 5"] }   # leave rotation first
---
apiVersion: v1
kind: Service
metadata:
  name: capture
spec:
  selector: { app: capture }
  ports: [{ port: 80, targetPort: 8080 }]
```

For the ingress, the same rule applies and the same mistake is available.
With ingress-nginx, a regex path with a capture, hashing the capture:

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/upstream-hash-by: "$1"
spec:
  rules:
    - http:
        paths:
          - path: /route/([^/]+)/.*
            pathType: ImplementationSpecific
            backend: { service: { name: capture, port: { number: 80 } } }
```

**Verify this in your own cluster** rather than trusting it — annotation
behaviour varies by controller version, and the failure is silent. Envoy-based
gateways need care for the same reason: a hash policy on the `:path`
pseudo-header covers the *whole* path and query, which splits a trajectory
across replicas. Extract the segment into a header first, then hash the header.

The check that actually proves it, from any of these:

```bash
curl -s "$CAPTURE_ENDPOINT/v1/trajectories/$TRAJECTORY_ID/graph" \
  | jq '.branch_points | length'   # must be 0
```

A branch in a strictly linear conversation means two replicas built the same
trajectory's graph. Run it as a deploy-time synthetic.

## Configuration

Everything is environment-driven, so one image serves every role.

### Core

| Variable | Default | Notes |
| --- | --- | --- |
| `CAPTURE_RECORD_DIR` | — | **Required.** Where the record goes, and where exports are written beside it. A process with nowhere to write refuses to start. Replicas may share one, under the routing rule below |
| `CAPTURE_VIEWER` | `true` | Whether this replica serves the read API and bulk exports. One per record directory should; `--disable-viewer` is the flag |
| `CAPTURE_MODE` | `text` | What this process captures for its whole life: `text` or `tokens`. It decides which registry `UPSTREAM_TYPE` is resolved in |
| `UPSTREAM_TYPE` | `openai` (`tokens` in token mode) | The inference server this process captures: a text protocol (`openai`, `anthropic`) or, in token mode, an engine wire (`tokens`, `vllm`) -- or one a module contributes to either |
| `UPSTREAM_URL` | `http://127.0.0.1:8000/v1` | Its base URL |
| `UPSTREAM_API_KEY` | — | Presented to the upstream on every forwarded call. Never stored, never returned. The only credential capture holds |
| `UPSTREAM_MODEL` / `UPSTREAM_TOKENIZER` | — | Tokenizer is required for a `tokens` upstream |
| `UPSTREAM_MAX_MODEL_LEN` | — | `tokens`: context window used to clamp `max_tokens` |
| `PUBLIC_URL` | `http://127.0.0.1:8080` | Base URL handed to workloads |
| `FINISH_GRACE_SECONDS` | `5.0` | How long `finish` waits for turns already in flight, and then for their commits. Also how long a TITO response waits for durability before failing the connection |

**Nothing expires.** There is no trajectory TTL: a trial nobody finished stays
in the record as an unfinished journal until somebody does. A sweep that
finalized it would be guessing, and would do so exactly when a slow harness was
about to come back.

`FINISH_GRACE_SECONDS` bounds two waits inside `finish`, and reaching either
bound means the call does not finish: `503`, retryable, trajectory untouched.
A trajectory finished out from under its own workload stays open rather than
being compiled from a state something is still writing to, and a retry --
immediately, or after the restart that took the hung turn with it -- finishes
it from a journal that has everything.

### Proxy

| Variable | Default | Notes |
| --- | --- | --- |
| `PROXY_HOST` / `PROXY_PORT` | `127.0.0.1` / `8080` | |
| `PROXY_MAX_REQUEST_BYTES` | 64 MiB | Larger bodies get `413` |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | Connect timeout to the upstream |
| `UPSTREAM_READ_TIMEOUT` | `3600` | Long-running streams need a generous value |
| `UPSTREAM_IDLE_CONNECTIONS` | `2048` | Idle keep-alive connections held per upstream origin. Connection reuse is what keeps forwarding cheap. **Not** a limit on requests in flight — see below |
| `UPSTREAM_CA_BUNDLE` | — | Extra CA bundle for upstream TLS, for a self-hosted upstream behind a private CA |
| `CAPTURE_ENABLED` | `true` | `false` forwards without capturing. The benchmark baseline |
| `PAYLOAD_SAMPLE_RATE` | `1.0` | What fraction of bodies a `bodies="sampled"` trajectory keeps. Whether to sample is per-trajectory, chosen at creation |
| `CAPTURE_STREAM_CHUNKS` | `true` | Per-chunk timings. Summary timings are always recorded |
| `MAX_STREAM_CHUNK_RECORDS` | `4096` | Cap on stored per-chunk records for one response |
| `TOKEN_TRACE_BUDGET_TOKENS` | `20000000` | Tokens held across cached token traces before the least recently used are dropped. A dropped trace is rebuilt from the graph on its next turn |

### The record

| Variable | Default | Notes |
| --- | --- | --- |
| `RECORD_COMMIT_CAPACITY` | `1024` | Journal appends queued at once before capture is out of room. Text refuses the work and records a gap; TITO waits for capacity. Neither blocks a forwarded request |
| `RECORD_FSYNC` | `always` | `always`, `interval`, or `never` |
| `RECORD_FSYNC_INTERVAL` | `1.0` | Worst-case loss window with `interval` |
| `RECORD_COMPRESS` | `true` | zstd for records above a few kilobytes |

**`RECORD_FSYNC=always` is the default because the product contract rests on
it.** Token capture promises that a cleanly closed response has its exact
exchange on disk, and creation promises that a route handed out is a route a
replacement process could serve; both promises are an fsync. The other two
settings exist to measure what that costs and they weaken it.

Text capture is unaffected either way at the request path: its appends run
behind the response, so the fsync costs writer throughput rather than inference
latency. The format is specified in
[design/record-format.md](design/record-format.md).

## The upstream transport

There is one, and it is a minimal HTTP/1.1 client written for this path. There
used to be an httpx implementation beside it, chosen by a setting; the numbers
are why it is gone:

| | this transport | httpx |
| --- | --- | --- |
| Throughput, capture off | **2388 rps** | 697 rps |
| p50 / p99 latency | **2.7 / 3.9 ms** | 9.2 / 29.6 ms |
| Fraction of the no-proxy ceiling | **88%** | 27% |
| 128 concurrent token trajectories | **494 turns/s, p50 210 ms** | 49 turns/s, p50 1864 ms |

Measured on a 10-core laptop against the in-repo mock provider; see
[benchmarks.md](benchmarks.md). The difference is httpx's per-request work in
the forward path — redirect handling, cookie and auth flows, content
negotiation, event hooks — all of which a transparent proxy must not do anyway.
Roughly 3× on independent requests, and much worse on token traffic, where a
trajectory's turns are serialized so the added latency compounds instead of
overlapping and throughput *falls* as concurrency rises. Nobody would have
chosen it; keeping it meant a second production path through the one piece of
code that runs per inference request.

**What it supports**, exactly: HTTP/1.1 over TCP or TLS, keep-alive pooling per
origin, `content-length`, chunked and read-until-close bodies. No redirects, no
content decoding, no cookies, no HTTP/2. Anything else is an error rather than
a guess. It is tested over a real TLS handshake with a generated certificate,
including that an untrusted one is refused. Still **exercise your own upstream
through the proxy before a production rollout**, since no repository test can
cover every provider's TLS stack.

Three behaviours worth knowing, because they are what a proxy in front of
inference has to get right:

- **A request is retried only when it cannot have run.** A pooled connection
  the server had already closed is the one case: the request reached nothing,
  so it is sent again on a fresh connection. Every other failure is reported.
  An upstream that read the request and then died may already have generated,
  and a retry would generate twice.
- **`UPSTREAM_READ_TIMEOUT` bounds the gap between reads**, not the whole
  response. A long generation streams for minutes and is healthy throughout; an
  upstream that has stopped sending is not, and is failed rather than held open.
- **Nothing caps requests in flight.** `UPSTREAM_IDLE_CONNECTIONS` bounds idle
  connections held per origin. A cap on live ones would make capture queue
  inference behind itself, and the upstream decides its own concurrency.

The transport cannot be configured to skip certificate verification. The proxy
holds the upstream credential, so an unverified connection is exactly the case
where it could be handed to the wrong server; a private CA sets
`UPSTREAM_CA_BUNDLE` instead. A test asserts no such switch exists.

## Sizing the write path

There is nothing to size for deriving: parsing an exchange and planning its
graph change happen in the serving process, after the response has gone out, so
the only question is whether one core keeps up — and the answer is not a number
you choose but one you watch.

What *can* be sized is the bound on appends in flight. In text mode it is the
only place capture can lose data, and it loses data only when the disk cannot
keep up for long enough to fill it.

```
commit_capacity  >  peak_requests_per_second  x  worst_case_disk_stall_seconds
```

At 1000 rps with a one-second tolerance for a storage blip, that is 1000
appends — the default 1024 covers it. Memory cost is roughly the average
exchange size times capacity; a 4 KB average at 1024 is about 4 MB, and the
copy is released as soon as the append lands.

Watch `capture_commits_pending`. If it approaches capacity under normal load
the disk is the constraint, and `RECORD_COMMIT_CAPACITY` only buys time: what
fixes it is faster storage, or splitting the traffic across capture processes.

**In token mode the same bound is a wait rather than a loss.** A response waits
for capacity before it closes, so a slow disk shows up as latency on
`capture_tito_close_wait_ms` and, past `FINISH_GRACE_SECONDS`, as failed
connections that clients retry.

## Monitoring

`GET /healthz` reports `degraded` when exchanges have been refused or the sink
is failing — and **always returns 200**, so a liveness probe keyed on it never
restarts a proxy that is serving fine.
`GET /readyz` answers for the proxy, never for capture: a capture failure must
not move traffic.

`GET /metrics` exposes the counters for Prometheus:

| Metric | Alert on |
| --- | --- |
| `capture_commits_refused_total` | Any increase. In text mode this is data loss — nothing else surfaces it |
| `capture_commit_failures_total` | **Any increase.** The disk telling you it is in trouble |
| `capture_gaps_unwritten` | Anything above zero for more than a moment: capture knows something is missing and cannot write that down either |
| `commits.undelivered_nodes` | Same, for the graph nodes of a lost record. They ride on the next record that can carry them, so a sustained non-zero value means nothing is being written at all |
| `capture_commits_pending` | Sustained above ~50% of `RECORD_COMMIT_CAPACITY` |
| `capture_commit_oldest_pending_seconds` | Sustained growth. The oldest append still waiting is what a crash would lose |
| `capture_torn_tails_total` | Any increase outside a known restart: a writer is being killed mid-append |
| `capture_tito_close_wait_ms` | Token mode: growth here is durability on the response path |
| `capture_tito_poisoned_total` | Any increase. A trajectory whose graph could not be extended |
| `capture_tito_delivery_unconfirmed_total` | Any increase. Responses whose delivery was never recorded |

`/healthz` carries the same numbers under `commits`, plus:

| Field | What it means |
| --- | --- |
| `registry.hot_trajectories` | Trajectories this process holds in memory |
| `registry.recovered` | Trajectories adopted from a journal another process wrote. A restart's shape, and a routing mistake's |
| `store.append_ms_mean` / `store.fsync_ms_mean` | What the disk costs per record |
| `store.records_written` / `store.bytes_written` | What the record has taken |
| `commits.last_error` | The most recent append failure, if there was one |

And under `tokens`, when the [prefix audit](verification.md) is on:

| Field | What it means |
| --- | --- |
| `audit_failures` | Classified prefix divergences, by class (`length`, `scaffold`, `sampled`, `given`). **Every class is a bug** — both sides of that comparison are tokens this process produced |

Per trajectory, the `capture` block says what capture knows it does not know,
and the three doubts are separate:

| Field | Means |
| --- | --- |
| `complete` / `calls_missing` | Calls capture saw and could not write |
| `recovery_uncertain` | A replacement process adopted this text trajectory mid-run |
| `delivery_uncertain` | Exchanges that are exact and durable, and whose delivery to the client cannot be vouched for |

The same block travels with every trajectory in an export, so a dataset
consumer can tell per trajectory rather than only in aggregate — and training
exports mark an uncertain branch untrainable rather than leaving the judgement
to the consumer.

## Header capture

Only allowlisted headers are persisted, and **credentials are always removed
whatever the allowlist says.** Whatever a client puts in `Authorization` is its
own business -- capture authenticates nothing inbound -- and the upstream
header would carry the provider key. Neither is ever stored or forwarded, and
that is not configurable.

Check what a deployment will persist before sending it real traffic, by making
one trajectory and reading it back:

```bash
ID=$(curl -s localhost:8080/v1/trajectories -d '{"project":"preflight"}' \
  -H 'content-type: application/json' | jq -r .id)
curl -s "localhost:8080/v1/trajectories/$ID" | jq '.upstream, .mode, .bodies'
curl -s -X DELETE "localhost:8080/v1/trajectories/$ID" >/dev/null
```

The delete matters: a trajectory left open takes until the expiry sweep to
close and shows up in every listing until it does. Deleting it is itself an event, so a reader of the
record arrives at the same absence.

## Credentials

| Secret | Storage |
| --- | --- |
| Upstream credential | **None.** Read from `UPSTREAM_API_KEY` at startup and held in the process |

That is the only one. Capture issues no per-trajectory credential: the
trajectory id in a route is correlation, and a deployment that needs its
callers authenticated does that in front of capture — see below.

There is nothing at rest to encrypt, so there is no deployment key to manage
and no ciphertext that could be moved between records. Rotating the upstream
credential means restarting the capture process with a new
`UPSTREAM_API_KEY` — which is the same operation as changing anything else
about the upstream, because the upstream is startup configuration.

Each trajectory records an `upstream_snapshot`: the type, URL, model and
tokenizer it ran against, credential excluded. A later restart against a
different server therefore never reinterprets a trajectory that already ran.

## The control plane is unauthenticated

Deliberately. Capture is an ephemeral in-cluster job, brought up beside the
inference server it captures and torn down with it; its `/v1` API is reachable
only from inside that job's network, exactly like the inference server it sits
in front of. There is nothing to administer over it either — the upstream is
chosen when the process starts.

**The data plane is unauthenticated too**, and this is the part to be explicit
about: a client that can reach this process and knows an active trajectory id
can write to that trajectory. The id is not an unguessable bearer secret and
must not be treated as one. Capture is designed for a trainer-controlled
sidecar or a local service, where the network boundary is the boundary.

Anything else needs authentication in front — an authenticating reverse proxy,
service-mesh identity, a network policy — chosen by the deployment rather than
imposed here. If your SDK insists on a non-empty key variable, put a
placeholder or your own ingress credential there; capture strips it and never
records it.

Export ids and filenames become path segments, so they are sanitized:
separators and control characters become underscores, dot runs are collapsed so
no `..` survives (these names also reach `file://` URLs, where a client may
normalize a path), and the result is length-bounded. The store refuses a path
outside the exports directory whatever built it. Nothing else a caller names
reaches a path: a project is a value in the log.

## Backup and retention

**Back up the record directory. That is the whole answer.** It is the only
thing that outlives the process: a journal per trajectory in flight and a
compiled record per finished one, self-contained, readable with nothing
installed. Copy it, snapshot it, apply lifecycle rules to it. A journal is
append-only and a committed record is written once, so an incremental backup
copies whole files and never rewrites one.

Nothing else needs backing up. Hot memory is a working copy of the journals,
and a replacement process picks a trajectory up from its file.

The cost, stated plainly:

- **Text capture** commits behind the response, so the appends still queued
  when a process dies are gone — reported live as `capture_commits_pending`
  and `capture_commit_oldest_pending_seconds`. A replacement marks the
  trajectory `recovery_uncertain`, because nothing identifies which exchange it
  might have missed.
- **Token capture** commits before the response closes, so nothing a client
  received cleanly is lost. What a crash can leave is an exchange that is
  durable and whose delivery was never confirmed: kept, flagged
  `delivery_uncertain`, and resolved only if a later request's own history
  contains that assistant output.

**There is no delete route.** Removing a trajectory is removing its files:

```bash
rm record/committed/*/tr_01M22KM5XT26F1CMHEW9K64XJ6.json.zst
```

Leaving a delete in the API would have meant a training run could lose its own
data over HTTP, and it never deleted the file anyway.

## Failure playbook

| Symptom | Cause | Action |
| --- | --- | --- |
| `/healthz` `degraded`, refusals rising | The disk cannot keep up | Check `commits.last_error`. Writing cannot be spread across processes; split the load across capture processes instead |
| Pending commits near capacity, no refusals yet | The disk is falling behind | Same. Enlarging the bound only buys time |
| A token turn's connection fails | Its exchange could not be made durable in `FINISH_GRACE_SECONDS` | The client retries the unchanged request, which is the design. Fix the disk |
| `capture_lazy_recoveries_total` climbing in steady state | Trajectories are landing on the wrong replica | The load balancer is not hashing on the trajectory id. See the routing section |
| Workload gets `410` | The trajectory has finished, is finishing, or was poisoned | Create a new trajectory. A route serves one trial by design |
| Workload gets `404` | No such trajectory here or in the record | Check `base_url` came from `create_trajectory`, and that routing sends this trajectory to a replica sharing its record directory |
| A finish returns `503` | The record could not be written | Retryable, and the SDK retries it three times. The trajectory stays open, so finishing it again is safe |
| Workload gets `502` | Upstream unreachable | Check `UPSTREAM_URL` and `UPSTREAM_API_KEY`. The exchange records `transport_error` |
| Exchanges but no graph nodes | Endpoint carries no message history (embeddings, model listings) | Expected; those are captured but not graphed |
| token capture turn returns `tokens_capture_failed` | Token attribution failed | Check the tokenizer matches the model. See [tokens.md](tokens.md) |
| `finish` slow to answer | Turns still in flight, or their commits are behind | Both are bounded by `FINISH_GRACE_SECONDS`; the log says which it waited for |
| `finish` returns `503` saying turns are still in flight | The workload has not stopped | The SDK retries three times; past that, finish again once it has. Raise `FINISH_GRACE_SECONDS` if the tail is legitimately long. The trajectory is untouched and its journal has everything |
| A listing says `indexing` and no total | The viewer is still reading the directory | Expected on a large record. Pages arrive while it scans; the total appears when it finishes |

## Upgrades

The record is files, so there is no schema to migrate and no upgrade step to
run. A new build started against an existing record directory picks trajectories
up as requests for them arrive.

`RECORD_FORMAT_VERSION` and `JOURNAL_FORMAT_VERSION` cover the record; a record
this build cannot read is refused by name rather than misread, and a global
event log from before per-trajectory persistence is rejected outright — export
it with the build that wrote it, or re-capture. `SCHEMA_VERSION` is stamped
into every trajectory document and export record. `DERIVATION_VERSION` is
separate and covers prefix matching and graph inference, so derivation logic
can change without invalidating captured data — the journal stays the source of
truth.
