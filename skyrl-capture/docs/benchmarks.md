# Benchmarks

This benchmark write up, tracks the important performance and scaling aspects of the skyrl-capture system that requires curated evaluations.

The benchmarks done in this section are measured by replaying **391 real Claude Code sessions** (SemiAnalysis's AgentX
traces <link it>) through the proxy — median context ~100k tokens, ~110k+ requests. Full
results and method in [Results](#results-real-agentic-traffic).

Summary: 

| | **Text capture** | **Token capture** |
| --- | --- | --- |
| Cost at 96 concurrent sessions | **none measurable** | **4% of throughput** |
| Overhead with concurrency per process | **none measurable up to 384 concurrency** | goes from 4% at 96 concurrency to 15% at 192 concurrency |
| Ceiling, one process | **none found** (to 7.4M input tok/s) | ~75 turns/s, ~3.9M input tok/s |
| What degrades first | — | the tail: p99 7.2 s → 37 s past saturation |
| Scaling recipe | Horizontal scaling (stateless) | a 2nd process is worth **1.6x** once proxy-bound |
| Routing | — | **plain round-robin costs 3.4%**; affinity optional |

**Text capture is free** at every level this harness can reach.

**Token capture is free to ~100 concurrent sessions and cheap to ~190.**
Per-turn work is **O(delta), not O(context)** — measured flat from 8k to 80k of
context — so what costs is not a long conversation but a **cold start**: a
session's first turn, or any turn whose cached trace was evicted. Session length
is therefore a first-order variable, and these numbers use the 30-turn sessions
the traces actually contain.

## The question

A capture proxy sits on the critical path of every inference call. Two things
about it have to be true, and they are different questions:

1. **It must not add latency a caller can feel.**
2. **It must not become the bottleneck before the inference server does.**

The first is a per-request question and is answered by comparing capture on
against capture off through the same proxy. The second is a concurrency
question and is answered by sweeping load until something saturates, then
identifying what.

The two capture modes put different work on the critical path:

| | **Text mode** | **Tokens mode** |
| --- | --- | --- |
| On the critical path | forward; capture is asynchronous (~6.5 µs) | render, call `/generate`, verify, commit, **and wait for the journal append before the close** |
| Tokenization | the **engine** does it, every turn, over the whole prompt | the **proxy** does it, incrementally |
| The saving | — | the engine stops tokenizing |

The second row is what makes this interesting. Token-in/token-out does not *add*
tokenization, it **relocates** it — so the question is whether what the proxy
adds exceeds what it removed, and a benchmark whose engine model has no
tokenization cost cannot answer that in either direction.

**Serialization is a property of the workload, not of the mode.** A
multi-turn agent in text mode is serialized in exactly the same way:
turn N+1's request contains turn N's response, so the client cannot issue it
early. Both modes are therefore measured under the same multi-turn
workload. What serialization changes is how latency is *felt* — inside one
conversation there is no pipelining, so added milliseconds land directly on that
session's wall clock and no amount of concurrency hides them.

All numbers below are from a 10-core Apple Silicon laptop with PostgreSQL in
Docker, Python 3.13, uvloop, against the in-repo mock upstream. **They predate
both the store moving into memory (2026-09-18) and per-trajectory persistence
(2026-09-21)** and have not been re-run. The text proxy-path numbers should be
unaffected -- persistence was never on the text request path and still is not
-- and the ingestion numbers are a floor. The token numbers now carry one cost
they did not: the durability wait before a response closes, measured at 0.56 ms
per turn against the mock engine and reported live as
`capture_tito_close_wait_ms`. Raw reports are
in [`benchmark-runs/`](benchmark-runs/).
[Running a full-scale benchmark](#running-a-full-scale-benchmark) at the end
reproduces every table.

## Results: real agentic traffic

These are the numbers to quote. They come from replaying **SemiAnalysis's
AgentX traces** — 391 anonymised Claude Code sessions plus the 610 sub-agent
conversations they fan out into, 111,026 model requests in all — rather than
from synthetic traffic, which is uniform in every dimension real agentic traffic
is not.

The published traces carry token counts and prefix block hashes, not text: they
are built for KV-cache and scheduler studies, where what matters is how much of
each prompt the engine has already seen. That is enough to replay the shape.
`in` grows the way a conversation grows, `out` is what the model produced, and
the replay synthesises text to hit those counts. Input and output
distributions, turn counts and context growth are the trace's; the words are
not — which is the right way round, since the proxy's cost depends on token
counts and prefix structure rather than on what the tokens say.

| | Replayed here | Published (newsletter) |
| --- | --- | --- |
| ISL p50 / p90 / p99 | 99k / 413k / 616k | 88k / 272k / 675k |
| OSL p50 / p90 / p99 | 458 / 2,470 / 8,609 | 413 / 2.2k / 8.6k |
| Turns per conversation | p50 31, p90 196 | "tens or hundreds" |

**"Conversations" is concurrency, but a heavier kind than a user count.** A
session's turns are serialized, so N conversations is at most N requests in
flight. What this replay does *not* take from the traces is their arrival
schedule: each session issues its next turn the instant the previous returns.
Real sessions idle — the inter-turn gap in these traces is p50 4.2 s, p90
36.7 s, and 31% of gaps exceed 10 seconds, because a turn waits on tool
execution and a human. So 192 here means **192 continuously active sessions**,
and one process carries substantially more than 192 real users. AIPerf replays
the original schedule, so its "concurrency 384" counts clients rather than
saturated sessions; the two are not interchangeable.

All three arms run under identical closed-loop pacing, so the comparison between
them is unaffected. It is the absolute capacity figure that is conservative, by
a factor this harness discarded rather than measured.

### Session length is a variable, not a detail

Per-turn work in tokens mode is **O(delta), not O(context)** — that is what
prefix reuse buys, and it holds. Fixed 2k-token delta per turn, one session,
zero-latency engine so the number is the proxy's own work:

| Context | Per-turn proxy cost |
| --- | --- |
| 8,200 | 1.9 ms |
| 38,950 | 2.0 ms |
| 79,950 | 3.3 ms |

Flat across a 40x growth in context. The renderer's bridge is likewise flat at
8.3 ms from 5k to 50k tokens.

**The expensive event is a cold start**, not a long context: a session's first
turn renders its whole opening context, and so does any turn after the LRU
evicts its trace. AgentX sessions *open* at around 50k tokens, so a cold render
costs what a hundred steady-state turns cost.

That makes session length a first-order variable. These traces run a median of
**31 turns**, and measuring six of them charges every conversation a cold start
every sixth turn:

| | 6-turn sessions | 30-turn sessions |
| --- | --- | --- |
| Cold renders | 17% of turns | **3% of turns** |
| Bridge rate | 81.5% | **90.6%** |
| Mean per-turn proxy cost | 237.6 ms | **117.6 ms** |
| Throughput kept, 96 sessions | 88% | **96%** |

Everything below uses **30 turns**, which is what the traces actually do.

### The sweep

30 turns per session, traces capped at 120k input tokens so they fit the
the configured context window. One proxy process.

| Sessions | Arm | turns/s | p50 | p90 | p99 | Throughput kept |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | direct | 13.8 | 362 ms | 1,627 | 7,690 | — |
| | text | 13.7 | 362 | 1,620 | 7,695 | **99%** |
| | token | 13.7 | 360 | 1,693 | 8,023 | **99%** |
| 96 | direct | 37.6 | 378 | 1,888 | 7,686 | — |
| | text | 37.5 | 380 | 1,889 | 7,689 | **100%** |
| | token | 36.2 | 480 | 2,595 | 8,029 | **96%** |
| 192 | direct | 88.4 | 316 | 1,424 | 7,218 | — |
| | text | 88.1 | 316 | 1,426 | 7,221 | **100%** |
| | token | 75.4 | 714 | 2,495 | 7,169 | **85%** |
| 384 | direct | 131.9 | 445 | 2,217 | 7,687 | — |
| | text | 131.2 | 446 | 2,337 | 7,690 | **99%** |
| | token | 67.2 | 1,818 | 7,214 | 36,964 | **51%** |

In tokens per second:

| Sessions | direct in tok/s | token in tok/s | direct out tok/s | token out tok/s |
| --- | --- | --- | --- | --- |
| 32 | 720k | 717k | 9,049 | 9,014 |
| 96 | 2.04M | 1.96M | 30,123 | 28,964 |
| 192 | 4.59M | **3.91M** | 68,658 | 58,528 |
| 384 | 7.37M | **3.76M** | 96,008 | 48,922 |

### What the numbers say

**1. Text capture is free, and no ceiling was found.** It tracks the direct arm
within noise at every level.

**2. Token capture is free up to ~100 concurrent sessions**, and cheap to ~190.
At 32 sessions the p50 is *lower* than direct's, which is noise, not a gain —
but it means the cost is below what this harness resolves. At 96 it is 4% of
throughput and 100 ms of p50.

**3. One process saturates near 75 turns/s, ≈3.9M input tokens/s.** Flat from
192 sessions to 384. Below that it is close to free; above it the queue is
visible as latency, and p99 goes from 7.2 s to 37 s between those two rows.

**4. The tail degrades before the median.** At 192 sessions p50 is 2.3x direct
while p90 is only 1.8x; by 384 p99 is 4.8x. A tail SLO breaks first.

**5. Prefix reuse holds.** Bridge rate 90.6% at realistic session length, and
the residual full renders are almost exactly the one-per-session cold start.

**6. Memory was never the constraint.** Residency peaked at 8.83M tokens of the
20M budget with zero evictions. Worth watching anyway, because an eviction is
not a memory event — it is a **cold start**, and cold starts are the expensive
thing here.

### Where the token proxy's cost actually goes

The proxy reports per-phase wall time and event-loop lag on `/healthz`, which is
how the remaining cost was found. At 192 sessions on one process:

| Phase | ms/turn @ 8 sessions | @ 192 sessions | Split across 2 processes |
| --- | --- | --- | --- |
| render | 19.9 | 170.9 | 103 |
| parse | 2.7 | 101.3 | 22 |
| trace | 32.0 | 132.0 | 16 |
| commit | 0.9 | 1.5 | — |
| **loop lag, mean** | 8.9 | **20.2** | **8.0** |

These are wall times spanning `await` points, so they are *queueing plus work*.
Splitting the same 192 sessions across two processes -- halving the sessions per
event loop while leaving the per-turn work identical -- cuts render by 40%,
parse by 78% and trace by 88%. **That is the proof that the phases are dominated
by queueing rather than by work**, and it points at the real constraint.

**One process is CPU-bound, and rendering is the CPU.** The true per-turn cost is
roughly 20-30 ms at these context and output sizes; at 75 turns/s that is 1.5 to
2 cores of demand on a process that has about one, because the work is
GIL-bound. The queue is what everything else observes.

So the 15% at 192 sessions is not a mystery and not a leak: it is a process
running out of CPU, and the fix is the one the architecture already provides --
another process. Rendering itself has little headroom left; the bridge costs
8.3 ms standalone for a 4k-token delta, which is close to what the tokenizer
itself costs.

#### The startup cliff this found

Loop lag peaked near **4 seconds at every concurrency level, including 8
sessions** -- not contention, a single blocking call. It was renderer
construction: loading a tokenizer and filling a pool, inline on the event loop,
on the upstream's first turn. Every request in flight at that moment stalled behind
it, so a cold replica taking traffic saw a four-second cliff rather than one
slow turn.

Building it in a thread under a lock halves worst-case lag (4,047 ->
1,774 ms) at identical throughput, which is what moving a one-time cost off the
loop should do.

### Adding processes

The same sessions split across proxy processes, each session pinned to one
process, as a trajectory-affine load balancer pins it:

| Sessions | Setup | turns/s | p50 | p90 | vs direct |
| --- | --- | --- | --- | --- | --- |
| 192 | engine direct | 88.4 | 316 ms | 1,424 | 100% |
| 192 | 1 process | 75.4 | 714 | 2,495 | 85% |
| 192 | 2 processes | 71.3 | **465** | 2,544 | 81% |
| 384 | engine direct | 131.9 | 445 | 2,217 | 100% |
| 384 | 1 process | 67.2 | 1,818 | 7,214 | 51% |
| 384 | 2 processes | **105.9** | **1,105** | **3,707** | **80%** |

**Processes buy throughput only where the proxy is actually the bottleneck.** At
384 sessions, where one process is at 51%, a second is worth **1.6x** and takes
it to 80%. At 192, where one process already keeps 85%, a second buys no
throughput at all — 0.9x, within contention noise — because there was only 15%
to recover.

What a second process buys at *both* levels is latency: p50 714 -> 465 ms at
192, and 1,818 -> 1,105 ms at 384. So the rule is not "add processes as load
grows" but **"add processes when the proxy is the bottleneck, or when the tail
matters"**. Below saturation they can cost a little throughput to contention.

### Routing, and why there is no relay

An earlier design had a proxy relay a turn it did not own to the replica that
did, over a hash ring the proxies shared. Measured on four replicas, the relay
cost nothing: it passes the raw body along and streams the reply back, without
re-rendering or re-committing, so the extra hop spends I/O rather than the
rendering CPU a process is actually bound on.

**It was removed anyway**, and the measurement is why it is worth recording
what the removal does not mean. The relay was never load-bearing: with a load
balancer that hashes the trajectory id it never fires, and it cannot help
across a rebalance, because when the ring changes the replica receiving the
turn *is* the new owner. What it did was quietly correct a misconfigured
balancer, which is worth catching rather than hiding.

So routing is the load balancer's job and only its job — see
[architecture.md](architecture.md#a-trajectory-must-stay-on-one-replica). Do
not read the old relay numbers as permission to run round-robin: text mode has
no fallback at all, and a misrouted turn there is a graph with branches the
workload never took, with no error anywhere.

What a rebalance costs is unmeasured either way: adding a replica moves 1/N of
trajectories, and a moved trajectory is a **cold start**, which is the
expensive event in this system. Consistent hashing at the balancer is what
bounds it -- 8 to 9 replicas moves 10.9% of keys where modulo would move 89% --
but 10.9% of live sessions rebuilding a 100k-token trace at once is a
thundering herd nobody has watched.

### Sizing rule

For token mode on agentic traffic, from these numbers:

* **Up to ~100 concurrent sessions per process: free.** 96-99% of engine
  throughput, p50 within 100 ms.
* **To ~190: cheap.** 85% of throughput on one process. A second process here
  is for the tail, not for throughput.
* **Past ~190: proxy-bound.** One process holds ~75 turns/s and ~3.9M input
  tokens a second. Partition and add processes; the second is worth 1.6x.
* **Budget on p90.** The tail degrades before the median, and p99 goes from
  7.2 s to 37 s between 192 and 384 sessions on one process.
* **Sessions here are saturated** — no think time. Real sessions idle p50 4.2 s
  between turns, so one process carries several times these numbers in real
  users.
* **Watch the bridge rate and the cold-start rate**, not the context length.
  Per-turn work is O(delta); what costs is a session's first turn and any turn
  whose trace was evicted.

Text mode needs none of this: nothing in this sweep found its limit.

## Methodology

Getting a trustworthy answer on one machine took several decisions that are
easy to get wrong. Each of them was, at first.

### The harness ceiling must be stated, not assumed

Every run begins with a calibration arm that drives the upstream directly, with
no proxy in the path. That number bounds every other number in the report, and
without it a reader cannot distinguish a proxy limit from a harness limit. It
is what makes the "28% of ceiling" finding below possible to state at all — and
that finding is what led to the lean transport.

**Read every table by comparing across a row, not down a column.** Where the
ceiling arm itself degrades, that row is measuring the harness.

### The load generator must not be the bottleneck

A single Python event loop driving many concurrent requests saturates well
before the proxy does. One generator process peaks near 1.7k rps and *loses*
throughput above about 8 concurrent requests:

| Concurrency, one process | Throughput | p50 |
| --- | --- | --- |
| 1 | 1471 rps | 0.66 ms |
| 8 | 1521 rps | 4.80 ms |
| 32 | 239 rps | 96.97 ms |
| 64 | 242 rps | 191.04 ms |

Four processes at concurrency 8 aggregate to about 4200 rps against the same
upstream, so the collapse is the generator, not the server. The first version
of this benchmark ran one generator at concurrency 64 and reported **190 ms of
"added p99 capture latency"** — the generator's own queueing, attributed to the
proxy.

Load is generated by several processes at modest per-process concurrency, and
exact latency samples are aggregated rather than averaged percentiles. The
mock upstream is multi-process for the same reason (`--workers N`).

### Added latency is only meaningful below saturation

At the ceiling, latency differences are queueing. The `--target-rps` flag paces
arrivals open-loop below the ceiling, which is the only mode in which "added
latency" measures per-request work. The harness records which mode it ran in
and suppresses the latency verdict in the wrong one.

### Ordering and warmup

Arms run **sequentially**, never concurrently, so they do not compete. The
baseline runs first, so residual warm-up cost is charged to it rather than to
capture. Each arm's own warmup — establishing connections, loading a tokenizer,
paying first-call costs — is excluded from the samples. A tokenizer loaded
inside the measured window once produced a reported **+4,115 ms p95**; warming
it first took that to 27 ms.

### Zero-drop verification

A throughput win paid for by dropping capture events is not a win. After each
capture-on arm the harness drains ingestion and asserts that every successful
request produced a durable exchange row. Every run below recorded every
exchange with zero drops.

## Supporting measurements

The sections below are synthetic: uniform turns, uniform padding, a
microsecond-to-millisecond upstream. They are how the pieces were isolated —
what one request costs, where the forward path's limit is, what the capture path
itself burns — and none of them should be read as a capacity figure. The numbers
to quote are the ones above.

### Text mode, synthetic

#### Throughput and latency by concurrency

Closed-loop, 3,000 requests per arm, 4 load processes, lean transport. Latency
is end to end as the client sees it.

| Total concurrency | Ceiling (no proxy) | Capture off | Capture on | Capture kept |
| --- | --- | --- | --- | --- |
| 8 | 4550 rps — p50 1.1 / p99 2.0 ms | 3783 — 1.4 / 2.5 | 3023 — 1.9 / 3.6 | 80% |
| 32 | 4328 — 4.2 / 17.2 | 3375 — 5.4 / 32.1 | 2820 — 7.0 / 29.0 | 84% |
| 64 | 1592 — 14.6 / 197.1 | 1447 — 15.0 / 674.1 | 1192 — 19.2 / 609.5 | 82% |
| 128 | 770 — 102.9 / 698.7 | 683 — 83.7 / 940.3 | 560 — 103.2 / 1088.9 | 82% |
| 256 | 519 — 245.7 / 2322.8 | 528 — 157.9 / 1864.4 | 459 — 169.3 / 2497.8 | 87% |

Zero dropped capture events at every level.

**Capture costs a steady 13 to 20% of throughput, flat across concurrency.**
That ratio is the honest headline: it does not degrade as load rises.

**The last three rows measure this laptop, not the proxy.** The ceiling arm —
no proxy in the path at all — falls from 4328 to 519 rps over those rows. Four
load processes, eight mock workers, the proxy, the ingestion worker and
PostgreSQL are contending for ten cores. The clean range of this harness is
**up to 32 concurrent requests**; past that all three arms are queueing
together, which is why "capture kept" *improves* to 87% at the top.

#### Added overhead below saturation

Open-loop — the mode in which added latency is meaningful — at 950 requests per
second sustained, lean transport:

| Metric | Value | Threshold | Result |
| --- | --- | --- | --- |
| Added p50 latency | **+0.07 ms** | — | |
| Added p95 latency | −0.63 ms | — | |
| Added p99 latency | **−0.35 ms** | ≤ 1 ms | pass |
| Throughput | **101.1%** | ≥ 99% of baseline | pass |
| Added error rate | 0.0 points | ≤ 0.01 points | pass |
| Dropped capture events | **0** | 0 | pass |
| Exchanges recorded | **4017 / 4017 accepted** | all | pass |

At 250 requests per second with the `httpx` transport, for comparison:

| Metric | Non-streaming | Streaming |
| --- | --- | --- |
| Added p50 latency | +0.09 ms | −0.39 ms |
| Added p99 latency | −0.53 ms | −0.67 ms |
| Throughput | 100.3% | 101.3% |
| Dropped capture events | 0 | 0 |

The negative numbers are noise: the difference is below what this harness can
resolve at these rates. The honest reading is that capture adds no measurable
latency below saturation.

#### Capture-path CPU cost, measured directly

Microbenchmarked in isolation (`docs/LOG.md` has the harness), per request:

| Step | Cost |
| --- | --- |
| Decode request + response headers | 1.54 µs |
| Construct the observed exchange | 3.34 µs |
| Build the metadata dict | 1.09 µs |
| Serialize metadata and frame the bodies | 2.37 µs |
| CRC-frame the queue record | 0.29 µs |
| Append to the bounded ring, routed to its worker | 0.41 µs |
| **Total capture path** | **6.5 µs** |

That measurement predates the write path this now has: there is no queue and
no worker to route to, and what the request path does after the response has
gone out is derive the exchange and submit one event.

This is why the "added latency" measurements have to be read carefully. 6.5 µs
of CPU cannot produce milliseconds of latency. Any larger figure is measuring
something else.

#### Throughput ceiling at saturation

Closed-loop, driving each arm to its limit, with the lean transport:

| Arm | Throughput | p50 | p99 |
| --- | --- | --- | --- |
| Upstream direct, no proxy (harness ceiling) | 2715 rps | 2.28 ms | 3.34 ms |
| Capture off | **2388 rps** (88% of ceiling) | 2.73 ms | 3.87 ms |
| Capture on, ingestion in a separate process | 2036 rps | 3.33 ms | 4.74 ms |
| Capture on, ingestion in-process | 2023 rps | 3.23 ms | 4.97 ms |

And with `httpx`, which was the default when these were first measured:

| Arm | Throughput | p50 | p99 |
| --- | --- | --- | --- |
| Upstream direct, no proxy | 2590 rps | 2.19 ms | 4.07 ms |
| Capture off | 725 rps (28% of ceiling) | 9.08 ms | 24.0 ms |
| Capture on, ingestion in a separate process | 555 rps | 11.9 ms | 42.1 ms |
| Capture on, ingestion in-process | 621 rps | 11.2 ms | 33.0 ms |

**The forward path, not capture, was the throughput limit.** With `httpx` the
capture-off baseline reached only 28% of the ceiling, so forwarding cost 3.6×
before capture was switched on at all — the client library's per-request work:
redirect handling, cookie and auth flows, content negotiation, and event hooks,
none of which a transparent proxy may do. The lean HTTP/1.1 client takes the
baseline to 88%.

### Token mode

These need two pieces running: the mock engine, and a capture process pointed
at it. There is no target to register -- the upstream is on the command line --
and no key to present, so the load generator needs neither.

```bash
# 1. The mock engine. TOKENIZE_MS_PER_1K is what the text path pays and the
#    token path does not, so leaving it out settles the comparison in advance.
#    CHARS_PER_TOKEN must match the workload's real ratio: the text arm is
#    charged on this estimate and the token arm on real token IDs.
MOCK_ENGINE_TPOT_MS=1 \
MOCK_ENGINE_TTFT_MS_PER_100=5 \
MOCK_ENGINE_MAX_COMPLETION=200 \
MOCK_ENGINE_TOKENIZE_MS_PER_1K=1 \
MOCK_CHARS_PER_TOKEN=5 \
  python -m tools.mock_server --port 9188 --workers 8

# 2. Capture, in whichever mode is being measured. Use a real fast tokenizer --
#    `builtin` is a pure-Python byte tokenizer for the test suite, two orders of
#    magnitude slower than production, so measuring it says nothing about the
#    design. One process serves one upstream, so the two modes are two
#    processes on two ports, each with its own state and queue directory.
skyrl-capture serve --port 8080 \
  --mode tokens --upstream-type tokens --upstream-url http://127.0.0.1:9188/generate \
  --model bench --tokenizer Qwen/Qwen3-0.6B --max-model-len 131072

skyrl-capture serve --port 8081 \
  --upstream-type openai --upstream-url http://127.0.0.1:9188/v1 --model bench

# 3. The three arms. --prompt-tokens is what puts the context near a real
#    agentic session; without it the workload is orders of magnitude short and
#    the tokenization saving is invisible.
for n in 32 96 192 384; do
  for arm in "--arm text-direct" \
             "--endpoint http://127.0.0.1:8081" \
             "--endpoint http://127.0.0.1:8080"; do
    python -m tools.bench.tokens_load $arm --agents $n --turns 6 \
      --max-tokens 200 --prompt-tokens 2000 --processes 12 --json
  done
done
```

To replay the AgentX traces, fetch them once and point `--traces` at the
result. Keep only what a replay needs -- the block hashes are ~99% of the 1.5 GB
and describe a prefix structure the `in` growth already implies:

```bash
python - <<'EOF'
import httpx, json
from huggingface_hub import hf_hub_url
url = hf_hub_url("semianalysisai/cc-traces-weka-with-subagents-060826",
                 "traces.jsonl", repo_type="dataset")
out, buf = open("plans.jsonl", "w"), b""
with httpx.stream("GET", url, follow_redirects=True, timeout=1800) as r:
    for chunk in r.iter_bytes(1 << 20):
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            t = json.loads(line)
            main = [{"t": q["t"], "in": q["in"], "out": q["out"]}
                    for q in t["requests"] if "in" in q]
            if len(main) >= 2:
                out.write(json.dumps({"id": t["id"], "parent": None,
                                      "requests": main}) + "\n")
            # A sub-agent group is its own burst of conversation.
            for q in t["requests"]:
                if q.get("type") != "subagent":
                    continue
                inner = [{"t": i["t"], "in": i["in"], "out": i["out"]}
                         for i in q.get("requests", []) if "in" in i]
                if len(inner) >= 2:
                    out.write(json.dumps({"id": f'{t["id"]}:{q["agent_id"]}',
                                          "parent": t["id"],
                                          "requests": inner}) + "\n")
EOF

# The engine must allow the real output distribution (OSL p99 is 8.6k), and
# the capture process the real input one.
MOCK_ENGINE_MAX_COMPLETION=16384 ...   # as above, plus the other knobs
skyrl-capture serve --port 8080 --mode tokens --upstream-type tokens \
  --upstream-url http://127.0.0.1:9188/generate \
  --model bench --tokenizer Qwen/Qwen3-0.6B --max-model-len 400000

for n in 32 96 192; do
  for arm in "--arm text-direct" \
             "--endpoint http://127.0.0.1:8081" \
             "--endpoint http://127.0.0.1:8080"; do
    python -m tools.bench.tokens_load $arm --agents $n --turns 6 \
      --traces plans.jsonl --max-input-tokens 120000 --processes 8 --json
  done
done
```

Poll `/healthz` *while* a run is in flight to see residency; traces are
forgotten at finish, so a reading taken afterwards is always zero.

To reproduce the multi-process result, start N proxies and split the load. Each
conversation stays on one proxy, which is what a trajectory-affine load
balancer does in a deployment:

```bash
# Each replica owns its own state and its own queue directory; they share only
# the record directory, which each writes its own trajectories into.
for port in 8080 8081 8082 8083; do
  CAPTURE_DATA_DIR=./capture-data-$port \
    skyrl-capture serve --port $port --record-dir ./traces &
done
for port in 8080 8081 8082 8083; do
  python -m tools.bench.tokens_load --endpoint http://127.0.0.1:$port --agents 96 \
    --turns 6 --max-tokens 200 --prompt-tokens 2000 --processes 3 \
    --endpoint http://127.0.0.1:$port --json &
done
wait
```

Watch `CPU usage` while that runs. On one laptop the load generators, the mock
workers and the proxies contend, so the result is a floor on what partitioning
buys rather than a measurement of it.

## Running a full-scale benchmark

The single-machine harness is for development. For a real 10k-rps run, separate
every component onto its own machine:

```bash
# 1. Upstream: the real inference service, or several mock instances behind a
#    load balancer so the upstream is not the ceiling.

# 2. Capture: several replicas behind a load balancer that hashes the
#    trajectory id. Each replica owns its queue volume, and ingests in-process.
PROXY_HOST=0.0.0.0 SPOOL_DIR=/mnt/spool-0 skyrl-capture serve

# 3. Load: many generator hosts, each modest per process.
python -m tools.bench.loadgen \
  --url http://capture-lb/route/tr_.../v1/chat/completions \
  --api-key trk_... --concurrency 8 --duration 300 --target-rps 400
```

Then check, for the capture-on run:

- `capture_events_dropped_total` stayed at zero,
- `capture_ring_depth` stayed well below capacity,
- the exchange count equals the successful request count,

and compare against the same topology with `CAPTURE_ENABLED=false`.

Sizing guidance for the ring and for ingestion shards is in
[operations.md](operations.md).
