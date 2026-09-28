# Token capture: correctness and performance log

Working notes for the token-in/token-out path. Kept separate from
[LOG.md](../LOG.md) because this is one subsystem being taken apart in detail:
what was tested, what broke, what the numbers were.

The invariant everything here serves:

> For every committed turn, the tokens stored along the path from the root to
> the assistant node must equal exactly `prompt_token_ids + completion_ids` —
> the bytes the engine received and returned. If that ever fails, an export's
> `input_ids` are not what the model saw, and training on them is training on
> a reconstruction.

## Setup

Real renderer (`renderers` over `Qwen/Qwen3-0.6B`, a reasoning model with a
dedicated renderer config), driving the real `TokenTrace`. The engine is
simulated so the tokens are controlled, but nothing about rendering,
attribution, bridging or commit is faked.

---

## Findings

### F1. Pool construction raced on `transformers`' lazy module

`create_renderer_pool` builds its renderers in a `ThreadPoolExecutor`, and
every worker runs `from transformers import AutoTokenizer`. `transformers` is a
lazy module, so concurrent resolution of the same attribute can leave one
thread looking at a partially populated module. The symptom is an `ImportError`
for a name that plainly exists:

```
ImportError: cannot import name 'AutoTokenizer' from 'transformers'
```

It is nondeterministic, which is worse than a hard failure: the first run of
the day worked, and a later identical run did not. Resolving the attributes
once before constructing the pool makes it deterministic — 5 of 5 after, having
failed 2 of 3 before.

This is ours to fix, not the library's: any embedder building a pool hits it.

### F2. `renderers` needs `transformers` 4.x, and does not say so

The pinned revision imports
`transformers.tokenization_utils.PreTrainedTokenizer` in seven renderer
modules, a path 5.x removed, while its own metadata asks only for
`transformers>=4.50.0`. Resolution therefore picks 5.x and the failure arrives
later as an unrelated import error, because the half-loaded module is already
cached in `sys.modules`.

Pinned `transformers>=4.50,<5` in the `tokens` extra, with the reason written
next to it, and `PrimeRenderer` now names the real cause if it sees a
non-4.x version. SkyRL hit the same thing and monkeypatches the missing path
back in; pinning says the same thing without writing into another package's
namespace.

---

## Correctness

Nine scenario families, each driving the real `TokenTrace` through the real
renderer and checking the invariant after every commit. All hold.

| Scenario | What it exercises | Result |
| --- | --- | --- |
| Multi-turn | An agent feeding its own replies back | bridges every turn |
| Reasoning replayed | `reasoning_content` returned in history | bridges |
| Reasoning dropped | the client strips it, as most SDKs do | full render, forks |
| Tool call → result → continue | tool syntax round trip | bridges through the tool turn |
| Client repair | an edited assistant message | forks, both branches exact |
| Tools widened mid-conversation | same messages, new tool set | full render, new branch |
| Mid-history edit | an earlier message rewritten | forks at the edit |
| Compaction | transcript replaced by a summary | forks at the system message |
| Late system prompt | system added on turn two | new root |
| Concurrency | 24 threads × 6 renders through one pool | byte-identical to serial |
| Corrupted bridge prefix | one token rewritten past the probe | refused at commit |

### F3. Prefix reuse is all-or-nothing, and the client decides

Bridging engages only when the client replays the model's *actual* sampled
output. Anything else — a reworded assistant message, a stripped reasoning
block, a fabricated history — drops to a full render and a new branch.

Measured on Qwen3, same conversation, one difference:

```
reasoning replayed:  bridge=y  reused=58  prompt=71  2 new nodes
reasoning dropped:   bridge=n  reused=0   prompt=34  3 new nodes, trace forks
```

Both are token-exact, so nothing is wrong with the capture. What is lost is
prefix reuse, and that is not a small loss:

```
 messages  tokens   full render    bridge   speedup
        1      16       0.235ms   0.091ms      2.6x
       11     201       1.027ms   0.101ms     10.2x
       41     767       3.330ms   0.110ms     30.4x
      101    1907       7.794ms   0.136ms     57.1x
      201    3808      15.025ms   0.196ms     76.7x
```

A full render is linear in the whole context; a bridge is flat. So a client
that drops reasoning content pays 77x the rendering cost per turn on a long
conversation, and gets a forest of shallow branches instead of one trajectory.

**This is the single most important thing for a user of tokens mode to get
right**, and it is entirely on the client side: replay what the proxy returned,
including `reasoning_content`.

### F4. The first benchmark measured the wrong path

Worth recording because it nearly produced a confident wrong answer. The first
end-to-end benchmark built its conversation by appending *invented* assistant
messages, then reported that proxy overhead grew from 1.4 ms to 2.9 ms with
depth. The profile showed 48 renders and **zero bridges**: a fabricated
assistant message is a client-authored node and anchors no inference boundary,
so the benchmark had measured the no-reuse path exclusively.

Rebuilt as a real agent loop that feeds back the model's own replies, the same
measurement is flat:

```
  turn  msgs  direct p50  proxy p50  added p50
     0     1       0.64ms     2.17ms     1.54ms
     5    11       0.84ms     2.28ms     1.44ms
    20    41       0.82ms     2.43ms     1.62ms
    40    81       0.93ms     2.59ms     1.66ms
```

A benchmark that does not reproduce the client behaviour the system is
optimized for measures the fallback.

---

## Scaling to a million tokens

700 turns, ~1,450 tokens added per turn, ending at **1,015,477 tokens** in one
trajectory. Per-turn CPU of the proxy's own work, with the inference call
removed.

Anything the proxy does that is linear in the *whole* context is quadratic over
the trajectory, so the question is how many such passes there are and how tight
each one is.

### Before

```
  turn  ctx tokens   prepare     fetch    bridge    commit     total
     9      13,687     0.09m     2.15m       ---     0.60m     2.89m
    99     144,277     1.07m     8.12m       ---     5.30m    14.79m
   199     289,477     2.29m    17.35m       ---    11.03m    31.22m
```

Six passes over the context per turn: hashing the history, rebuilding the
previous prompt twice, building the new prompt, verifying it against the
transition, and verifying it again against the nodes. Plus three walks of the
whole path just to compute a *length*.

### After

```
  turn  ctx tokens   prepare     fetch    bridge    commit     total
   199     289,477     2.27m     0.00m     7.00m     1.35m    10.94m
   399     579,877     4.55m     0.00m    16.01m     3.01m    24.42m
   499     725,077     7.57m     0.01m    20.14m     3.36m    32.18m
   699   1,015,477     9.87m     0.01m    26.97m     5.22m    43.19m
```

At the same 289k tokens: **31.2 ms → 10.9 ms per turn**, and 3.07 s → 1.43 s of
cumulative CPU over 200 turns. A full million-token trajectory costs 15.2 s of
proxy CPU across 700 turns, against inference calls that are themselves seconds
each at that context length.

### What changed

| Change | Why it mattered |
| --- | --- |
| `TokenNode.cumulative_tokens` | Three places walked the entire path and copied every token to ask how long a prefix was. Now an integer already on the node |
| Bounded cache of recent transitions' tokens | Bridging needs the previous turn's exact prompt, which was rebuilt from the path every turn. It is already in hand at commit, so it is kept — `fetch` went from 17 ms to 0.00 ms |
| Pass sequences to the renderer unconverted | `list(...)` on the previous prompt copied the whole context before the library did the same |
| One prefix verification, not two | `_validate` and `_plan_commit` were both comparing the reused prefix. They are inductively the same claim, so the weaker one is now a length check and the stronger one — against committed node deltas — is kept |
| Renderer probes the boundary, trace checks the tokens | The renderer-level full comparison cost three more passes to re-answer what the trace answers authoritatively |
| `orjson` round trip instead of a recursive Python walk | `normalize_json` was 82% of message canonicalization, re-sorting keys that `orjson.OPT_SORT_KEYS` sorts in C and that dict equality does not need |

The last one is on the text path too: every ingested message went through it.

### What is left, and what the floor is

At 289k tokens the irreducible work for one turn — concatenating the previous
context with the new tail, and serializing the result for the wire — is
**1.72 ms**. The proxy is at 10.9 ms, so the gap is 6.3x, down from 18x.

```
  bridge   7.00 ms   the library building the new prompt and its attribution
  prepare  2.27 ms   hashing the whole history to match it against the graph
  commit   1.35 ms   comparing the prompt against the committed node deltas
```

`prepare` is at its own floor: the protocol hands us the entire history every
turn, so matching it against the graph means hashing it, and that is
`orjson` plus `sha256` over about a megabyte. A cheap equality fast path was
considered and rejected — Python equality treats `1` and `1.0` as the same
value, so reusing a cached hash on that basis could send the engine a prompt
that differs from what the client asked for. Wrong tokens are not a trade worth
making for 2 ms.

`bridge` is the remaining lever and the one worth taking next. The library is
asked for the entire new prompt, but the tokens a turn *appends* depend only on
the new messages and the template, not on how long the conversation already is.
Rendering the delta against a short stand-in prefix and splicing it onto the
cached context would make it O(tail) instead of O(context) — the same
"dummy-prefix incremental tokenize" the Miles TITO implementation uses. It is
not done here because it moves a correctness guarantee: the spliced tail is not
verified by anything local, where today's full bridge is checked against the
stored nodes. The safe shape is to verify the splice against a full bridge
while the context is still short, which is exactly when verification is cheap,
and trust it afterwards.

### Memory

Retained cost is about **76 MB per million tokens** — 36 MB of token IDs,
32 MB of logprobs, 8 MB of masks — with token IDs stored as tuples of Python
ints at 36 bytes each. An `array("i")` would hold them in 4.

Peak RSS for the million-token run is 1.07 GB, and almost all of that is
transient rather than retained: every turn materializes a fresh full-context
list of token IDs and another of message indices. The splice above would remove
both. Cache size barely moves it — holding four recent contexts instead of one
costs under 100 MB — so it is set to 2, enough to cover a branch and a
continuation.

### F5. Nothing evicted a token trace

`TokenService.forget` existed and was called from one test and nowhere in
`src/`. `_traces` and `_locks` therefore grew for the life of the process: every
trajectory a replica had ever served stayed resident, holding its token arrays
at about 76 MB per million tokens.

That, not the hash ring, was the ceiling on how many users a tokens deployment
could take. It is also the kind of failure that looks like nothing until a
replica dies.

The fix rests on what a trace actually is. It is a **cache** -- the graph and
the object store can rebuild it, which is exactly what a cold replica already
does -- so it can be dropped, and eviction costs a slow turn rather than
anything lost. Three changes:

- `forget` is called on finish, on delete, and from the expiry sweep, so a
  trajectory that can take no more turns stops occupying memory.
- The cache is an LRU bounded by **tokens resident**, not by trajectory count:
  trajectories differ by four orders of magnitude in length, so a count bound
  would be meaningless. `TOKEN_TRACE_BUDGET_TOKENS` defaults to 20M, roughly
  1.5 GB of token arrays.
- `/healthz` reports `cached_traces`, `resident_tokens`, `trace_budget_tokens`
  and `traces_evicted`. Resident climbing toward the budget with evictions
  rising means a replica is holding more live trajectories than it can cache,
  and every eviction is a rebuild someone waits for.

A test pins the part that matters: after an eviction the next turn still
continues the trajectory rather than forking it, and the graph still has both
generations attributed to the model.

---

## Concurrent trajectories

`bench/tokens_load.py` runs N trajectories at once, each feeding its replies
back verbatim so prefix reuse engages, and times only the turns:

```bash
python -m tools.bench.tokens_load --target tokens-demo --agents 16 --turns 12
```

One replica, a real tokenizer, the bundled mock as the engine:

| agents | upstream alone | through the proxy | p50 | p95 | p99 | resident tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 458 calls/s | 294 turns/s | 2.3 ms | 6.8 | 9.3 | 512 |
| 2 | 1733 | 439 | 3.3 | 12.8 | 14.4 | 1,024 |
| 4 | 2095 | 573 | 6.5 | 11.0 | 12.5 | 2,048 |
| 8 | 1548 | **590** | 12.1 | 20.4 | 27.7 | 4,096 |
| 16 | 714 | 517 | 25.7 | 52.7 | 68.2 | 8,192 |
| 32 | 295 | 212 | 56.4 | 460.8 | 727.2 | 16,384 |
| 64 | 143 | 113 | 265.6 | 1879.2 | 3207.0 | 32,768 |

The knee is at **8 to 16 concurrent trajectories**, peaking near 590 turns/s
with p50 under 26 ms, and no failures anywhere in the sweep. Resident tokens
track agents exactly and nothing was evicted, so memory behaved linearly.

**The rows past 16 agents are not a measurement of the proxy.** The mock
upstream collapses from 2095 calls/s to 143 over the same range, so both ends
are degrading and the harness cannot separate them. Everything below 32 agents
is a clean proxy number; above it, this setup has nothing to say.

### F6. The first version of this benchmark measured setup

It timed trajectory creation and finish alongside the turns. At one agent only
**21% of the wall clock was inside a turn**, so the reported figure was mostly
control-plane round trips: 93 turns/s where the real answer was 573. Creating
before the clock starts, releasing every agent from one barrier, and finishing
after it stops gives the numbers above.

Second time this trap has been hit in this file -- F4 was a benchmark that
measured the no-reuse path. A load generator has to be read as carefully as the
thing it measures.

### What this does and does not tell you about users per replica

It says CPU is not the constraint, but only by argument: the engine here
answers in about a millisecond, where a real one takes hundreds. The next
section measures that instead of arguing it.

What binds instead is memory, and that is now a number you can set.
`TOKEN_TRACE_BUDGET_TOKENS` defaults to 20M tokens of resident trace, so a
replica holds roughly 400 live 50k-token trajectories, or 20 at a million
tokens each, before it starts evicting -- and an eviction costs a rebuild on
that trajectory's next turn, not an error. `/healthz` reports
`resident_tokens` against `trace_budget_tokens` with `traces_evicted`, which is
the signal to add a replica.

---

## Against an engine that behaves like one

A millisecond engine is the worst case for a proxy's overhead ratio and tells
you nothing about a real deployment. So the mock grew a latency model:

* **1 ms per output token**, so 200 tokens out is 200 ms.
* **5 ms per 100 *uncached* input tokens** — 2,000 new input tokens is 100 ms,
  and a prompt the session already sent is free.

Uncached is the part that matters. The engine keeps a prefix per `session_id`
and charges only for what extends it, which is the same thing prefix reuse is
worth on our side. Off unless `MOCK_ENGINE_TPOT_MS` is set, so the suite is
unaffected.

The comparison drives the same token shapes both ways: through the proxy as
Chat Completions, and straight at the engine as `prompt_token_ids`. Shapes are
read back from what the proxy actually sent, so both arms give the engine
identical work.

```bash
python -m tools.bench.tokens_load --target rl --agents 32 \
  --turns 16 --max-tokens 200 --compare
```

| agents | direct turns/s | direct p50 | proxy turns/s | proxy p50 | proxy adds | throughput kept |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.9 | 204.4 ms | 4.8 | 205.9 ms | +1.6 ms | 98% |
| 4 | 19.4 | 204.5 | 19.1 | 209.3 | +4.8 ms | 98% |
| 8 | 38.7 | 206.1 | 38.2 | 208.0 | +2.0 ms | 99% |
| 16 | 76.7 | 203.7 | 75.4 | 208.1 | +4.4 ms | 98% |
| 32 | 152.2 | 203.2 | 144.2 | 207.1 | +3.9 ms | 95% |
| 64 | 123.0 | 236.1 | 82.3 | 366.5 | +130 ms | 67% |
| 128 | 54.5 | 1376.0 | 45.8 | 2095.6 | +720 ms | 84% |

**Up to 32 concurrent trajectories the proxy costs about 2% of a turn** —
2 to 5 ms on a 205 ms call — and keeps 95 to 99% of the engine's throughput.

Past that the mock engine is the thing being measured, not the proxy. At 64
agents the direct arm is already off its own curve (123 turns/s where 64
agents at 205 ms should give 312) and by 128 both arms are in the seconds,
with the proxy returning `502 tokens_upstream` as the engine fails under it.
A single-process asyncio mock has no batching, which is most of what a real
serving engine does at that concurrency. **The clean range of this harness is
1 to 32 agents.**

### F7. A tokenizer loaded inside the measured window

The first run of this comparison reported the proxy adding **4,115 ms at p95**.
Turn 0 alone was 4,174 ms: the target's first turn loads its tokenizer, and it
landed in the tail as though the proxy had stalled. A warmup trajectory before
the clock starts brought p95 from 4,175 ms to 27 ms.

Third measurement trap in this file, after a benchmark that measured the
no-reuse path and one that measured trajectory setup. The pattern each time is
the same: a number far enough from the others to be implausible, which is worth
more attention than a number that merely looks bad.
## Where one proxy process actually stops

The table above stopped at 128 agents and blamed the mock, which was half
right: a single-process asyncio upstream was the ceiling, and so was a
single-process load generator. Both grew processes — `mock_server --workers N`
forks N uvicorn processes over one bound socket sharing the session prefix map
through a manager, and `tokens_load --processes N` splits the agents across N
load processes that meet at a `multiprocessing.Barrier` before the clock
starts. With the upstream and the client out of the way, the sweep goes to 512.

| agents | engine only | proxy | proxy p50 | throughput kept |
| --- | --- | --- | --- | --- |
| 32 | 156.6 turns/s | 143.2 | 211 ms | 91% |
| 64 | 307.7 | 281.9 | 212 ms | 92% |
| 128 | 610.6 | 519.3 | 215 ms | 85% |
| 256 | 1166.3 | 815.6 | 244 ms | 70% |
| 512 | 894.5 | 820.7 | 333 ms | 92% |

Zero failures everywhere, all 4,096 turns landing at 512 agents. **One proxy
process saturates at about 820 turns/s**, which it reaches at 256 agents and
holds flat to 512; past saturation the queue shows up in latency (p50 244 ms →
333 ms, p99 722 ms → 2,854 ms) rather than in errors. The engine-only arm
degrades at 512 too, which is why the last row's 92% looks better than the
256-agent row's 70% — at 512 both arms are queueing.

### F8. The ceiling is the process, not the machine

At its ceiling the proxy used **40% of one core of ten**, so it was not
CPU-bound. Two cheaper explanations both failed: `uvloop` was missing, and
installing it moved nothing (519 → 541, 816 → 841, 821 → 793 turns/s — noise).

Running the same 512 agents split across **two** proxy processes, 256 each:

| | turns/s | p50 |
| --- | --- | --- |
| one process, 512 agents | 820.7 | 333 ms |
| two processes, 256 agents each | 814.0 + 747.4 = **1561.4** | 241 / 257 ms |

**1.9x, and latency back down to its unsaturated value.** So the limit is one
asyncio event loop's per-request overhead — a turn crosses it several times, to
read the request, render, call upstream, verify and commit — and the way past
it is more processes, which is what the hash ring is for. Nothing in the
tokens path has to become faster for the system to serve more.

### F9. The transport is worth 10x at concurrency

Before any of that, the same sweep on `httpx` — the default at the time —
showed a collapse, not a plateau:

| agents | proxy, httpx | proxy, lean |
| --- | --- | --- |
| 32 | 143.1 turns/s, p50 209 ms | 147.9, 208 ms |
| 64 | 85.3, 466 ms | 274.1, 216 ms |
| 128 | 49.1, 1864 ms | 494.2, 210 ms |

Throughput *falling* as agents rise is congestion collapse, and it was entirely
an httpx artifact: `UPSTREAM_TRANSPORT=lean`, the minimal HTTP/1.1 client in
`transport/http.py`, holds p50 at the engine's own 203 ms floor all the way to
128 agents. The text benchmarks had already found httpx's per-request work to
be the limit there; this is the same finding, louder, because tokens mode's
turns are serialized within a trajectory and cannot absorb the added latency.
Together the two results are what made `lean` the default for both modes.

### Three ways the harness lied before it worked

Each of these produced a plausible number that was wrong, and each is now
fixed in `bench/tokens_load.py`:

* A load process that crashed left its siblings waiting forever on the
  cross-process barrier and the parent waiting forever on the queue. The run
  hung instead of saying what went wrong. Workers now abort the barrier and
  report a crash, and the parent waits with a deadline.
* The 512-agent run reported **zero turns and 512 failures** three times. The
  measured window had succeeded every time; the *teardown* — 512 `finish` calls
  fired at once — timed out, and the exception discarded the report. Teardown
  is now chunked and its failures are recorded, not raised.
* `kern.ipc.somaxconn` is 128 on macOS, so at 512 agents the kernel dropped
  connections during setup that never reached the server. Creation is chunked
  and retried. This is the load generator hitting an OS limit, not the proxy
  refusing work, and it happens outside the clock.

The shape is the same as F4, F6 and F7: the implausible number was the useful
one. A run that reports zero is easy to believe as saturation and was not.
