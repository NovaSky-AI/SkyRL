# The write-path rework, measured

Two halves of one page: the baseline taken before items 1-5, and the same
measurements after them. The order is the point -- five changes to the write
path landed between them, and without a number from before, a worse number
after would name no cause.

**The answer: nothing regressed.** Capture-on throughput is flat to 1.3% up,
per-turn token overhead is unchanged at +9 ms p50, prefix reuse is 80.0%, and
every integrity counter is zero. The detail is below, including why the
harness's own pass/fail ratio turns out not to be measurable at this size.

---

# Baseline, before the rework

Taken before the write-path rework (drop multi-tenancy, rename, hierarchy,
in-process graph, file-backed record). Everything after that is measured
against this page. Without it, a worse number at the end names no cause.

Machine: 192 cores, 2 TB, Linux 6.12, Python 3.12.13, uvloop on. PostgreSQL is
the embedded server, on a unix socket under `/tmp`. Both runs use the mock
upstream, so upstream cost is a constant and cancels out of every delta.

## Text: capture on versus capture off

`skyrl-capture bench --requests 3000 --concurrency 8 --processes 6`, closed-loop.

| arm | rps | p50 ms | p95 ms | p99 ms | dropped |
| --- | ---: | ---: | ---: | ---: | ---: |
| upstream direct (no proxy) | 4543.5 | 4.99 | 11.44 | 32.23 | - |
| capture off | 3583.0 | 7.23 | 8.85 | 16.78 | - |
| capture on, split ingestion | 3405.9 | 9.52 | 10.85 | 14.01 | 0 |
| capture on, in-process ingestion | 3311.4 | 10.23 | 11.56 | 12.79 | 0 |

Capture on holds **95.06%** of capture-off throughput with ingestion in its own
process, **92.42%** with it on the proxy's loop. Every accepted exchange was
ingested (3048 of 3048); the split worker was 2815 behind when load stopped and
caught up in 8.07 s, which is queue depth rather than loss.

This is closed-loop, so the latency columns are queueing at the ceiling, not
per-request cost. The 99%-of-baseline throughput threshold fails at this size:
3000 requests per arm over six load processes is a short run, and the number to
beat later is 95.06%, not the threshold.

## Tokens: per-turn overhead

`python -m tools.bench.tokens_load --agents 32 --turns 6
--max-tokens 200 --prompt-tokens 2000 --processes 8`, against a mock engine
with `MOCK_ENGINE_TPOT_MS=1 MOCK_ENGINE_TTFT_MS_PER_100=5
MOCK_ENGINE_TOKENIZE_MS_PER_1K=1`, tokenizer `Qwen/Qwen3-0.6B`.

| arm | turns/s | p50 ms | p95 ms | p99 ms | turn 0 | turn 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| engine directly, no proxy | 142.0 | 207.7 | 309.7 | 310.3 | 308.6 | 207.5 |
| text target through the proxy | 140.1 | 209.1 | 317.6 | 319.1 | 315.2 | 208.9 |
| tokens target through the proxy | 125.2 | 218.0 | 403.7 | 419.8 | 393.7 | 209.1 |

Per-turn overhead against the engine: **+1.3 ms p50** for a text target,
**+10.3 ms p50** for a tokens target. Turn 0 carries the tokenizer load and the
first render; by turn 5 the tokens arm is within 1.6 ms of driving the engine
directly, which is what prefix reuse buys.

## Prefix reuse

From `/healthz` over the whole session (382 turns):

```
turns_served    382
render_full      80     first turn of each trajectory, plus the warm-up
render_bridged  305     extended a cached prefix
```

**Prefix-reuse rate 79.8%** (305/382), with `traces_evicted 0` and
`commit_failures 0`. 80 full renders for 76 trajectories is the floor: every
trajectory pays one.

Phase breakdown, milliseconds per turn:

```
trace     96.084     load or rebuild the trajectory's token trace
prepare    0.074
render     7.917
upstream 213.436     the mock engine
parse      2.608
commit     0.247
respond    0.050
```

`trace` is the number worth remembering. It is the largest cost the proxy
itself pays, and it is what item 4 — holding the graph in the writing process
instead of rebuilding it from SQL — is meant to remove.

## Reproducing

The embedded PostgreSQL needs a usable locale; `initdb` fails with "invalid
locale settings" under an unset `LANG`. Run with `LANG=C LC_ALL=C` or point
`DATABASE_URL` at a server.

---

# After the rework, 2026-09-17

Items 1–5 in place: single upstream, renamed package, the hierarchy, the
in-memory graph builder, the file-backed record. Same machine, same commands,
same mock upstream.

## Text: the capture path did not move

The harness was run four times in total, once before and three times after.

| run | ceiling | capture off | capture on, split | capture on, local | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 4543.5 | 3583.0 | **3405.9** | 3311.4 | 95.1% |
| after 1 | 4605.4 | 3830.9 | **3432.3** | 3304.8 | 89.6% |
| after 2 | 4488.8 | 3410.6 | **3437.5** | 3268.2 | 100.8% |
| after 3 | 4680.5 | 3227.3 | **3486.1** | 3378.6 | 108.0% |

Read the bold column, not the last one. **Capture-on throughput is flat to
slightly up: 3405.9 → 3432, 3438, 3486 rps**, a spread of ±1.2% around a mean
that is 1.3% above the baseline. The in-process arm is the same story within
2%. Nothing in items 1–5 costs the capture path anything measurable, which is
what would be expected: the data plane lost a credential cache lookup and a
lease-record indirection, and gained nothing.

**The ratio is noise, and that is the finding worth keeping.** The capture-off
arm ranges 3227–3831 rps across four runs — ±9%, seven times the variance of
the capture-on arm — so the harness's "≥99% of baseline" threshold is not
measurable at 3000 requests per arm on this machine. It swung from 89.6% to
108% with no code change between the last two runs. Either raise the request
count until the baseline arm stabilizes, or compare absolute capture-on
throughput across runs, which is what the table above does.

Capture integrity was perfect in every run: 3048 of 3048 accepted exchanges
ingested, **zero dropped**, zero ingest errors, zero misrouted records.

## Tokens: per-turn overhead and prefix reuse

| | baseline | after |
| --- | ---: | ---: |
| engine directly, turns/s | 142.0 | 142.2 |
| through the proxy, turns/s | 125.2 | 126.3 – 127.2 |
| turn latency p50 | 218.0 ms | 215.3 – 219.3 ms |
| turn latency p95 | 403.7 ms | 400.1 – 410.4 ms |
| turn 0 (cold) | 393.7 ms | 365.2 – 388.9 ms |
| turn 5 (warm) | 209.1 ms | 209.9 – 212.5 ms |
| **prefix-reuse rate** | 79.8% | **80.0%** (320/400) |
| `commit_failures` | 0 | 0 |

Per-turn overhead against the engine is unchanged: **+9 ms p50**, and within
3 ms of driving the engine directly by turn 5. Reuse is 80.0%, which is the
floor — every trajectory pays one full render, and 80 full renders for 76
trajectories across two runs is exactly that.

Phase breakdown, milliseconds per turn, over 400 turns:

```
trace     91.361   (baseline 96.084)
render     7.094   (baseline  7.917)
upstream 214.712
```

`trace` is what item 4 was pointed at, and it is flat. That is expected rather
than disappointing: item 4 moved the **text** graph builder into the writing
process, and `trace` is the *token* trace, which has always been built in
memory. A single-run reading of 178 ms/turn earlier in the session was a cold
start — tokenizer load and pool fill land in the first turns — and washes out
by 400 turns.

## New counters, all at their expected values

| | |
| --- | --- |
| `ingest.graphs_recovered` | **0** — no trajectory read its graph back from storage |
| `ingest.live_graphs` / `live_graph_nodes` | 0 after the run — every finished graph was released |
| `ingest.misrouted` | 0 |
| `ingest.record_errors` | 0 |

## Reproducing

```bash
skyrl-capture bench --requests 3000 --concurrency 8 --processes 6 \
  --database-url "$DSN" --output report.json

python -m tools.bench.tokens_load --agents 32 --turns 6 \
  --max-tokens 200 --prompt-tokens 2000 --processes 8 --json
```

The embedded PostgreSQL needs a usable locale; `initdb` fails with "invalid
locale settings" under an unset `LANG`. Run with `LANG=C.utf8 LC_ALL=C.utf8`.
