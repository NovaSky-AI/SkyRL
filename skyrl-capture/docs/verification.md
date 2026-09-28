# Verification

Two layers, and they check different things. The first proves the record is
consistent with itself; the second proves it is consistent with the engine.
Only the second can catch a record that is perfectly self-consistent and still
not what the model saw, because the thing being checked is a boundary the
capture process cannot see across.

| | Layer 1: the prefix audit | Layer 2: re-feed the engine |
| --- | --- | --- |
| Where | Inside the proxy, per turn | Afterwards, against a finished record |
| Cost | O(context) per turn; off by default | One engine call per path |
| Proves | The graph and the prompt agree | The prompt is the one the engine saw |
| Misses | A prompt the engine never saw | A divergence that never reached a sampled span |
| Run it | `TOKENS_AUDIT_PREFIX=1` | `tools/verify/` |

## Layer 1: the prefix audit

On the normal path the reused prefix is not re-read. It is the tokens those
nodes were committed from, handed to the renderer's bridge and returned under a
contract the library refuses rather than breaks, so agreeing on *where* the
prefix ends is agreeing on the prefix — a length check, not a walk.

`TOKENS_AUDIT_PREFIX=1` re-proves it token for token on every bridged turn, and
**refuses** the turn if it disagrees. `TOKENS_AUDIT_PREFIX=report` classifies
and counts the disagreement and lets the turn commit, which is for an
investigation rather than a deployment.

A disagreement is classified, because "it diverged" is detectable and not
diagnosable:

| Class | Where it lands | What to look at |
| --- | --- | --- |
| `length` | the totals disagree | a commit or bridge accounting bug |
| `scaffold` | before a node's `sampled_start` | the renderer, or its turn-close trimming |
| `sampled` | at or after `sampled_start` | the tokens the engine returned are not the tokens stored — the most serious class |
| `given` | inside a client-authored node | a bridge that let a client message retokenize |

Every class here is a bug. Miles's comparator tolerates its `ASSISTANT_TEXT`
class because it compares against a canonical re-render, and the canonical
retokenization of sampled text is the thing that is wrong. We never re-render a
committed turn, so both sides of this comparison are tokens this process
produced, and none of these differences has an innocent reading.

A report names the class, the node, the absolute prompt offset, and a decoded
window either side:

```
sampled tokens diverge at prompt offset 1412, 27 into node nd_01M2… (model/assistant):
stored '…the migration and\n' vs sent '…the migration then\n'
```

Counts appear on `/healthz` under `tokens.audit_failures`, by class.

## Layer 2: re-feed the engine

```bash
uv run python -m tools.verify.cli --record ./traces --run-id run-a \
  --engine-url http://127.0.0.1:8000/generate --model policy
```

It takes each path's recorded `input_ids` and `loss_mask`, splits at the first
sampled token, hands the prompt back to the engine greedily, and compares what
comes back with the completion the record says was produced:

```
input_ids  = [ ......... prompt ......... | .... completion .... ]
loss_mask  = [ 0 0 0 0 0 0 0 0 0 0 0 0 0  | 1 1 1 1 1 1 1 1 1 1  ]
                                          ^ the split
```

Two things are checked at once. **The completion matches**, so the stored
prompt is the prompt that produced the stored completion — a record holding a
prompt the engine never saw diverges within a few tokens, and the report says
where. And **the engine accepts the prompt at all**, which catches a malformed
scaffold, a doubled turn marker or a truncated tail: a prompt a served model
mangles and nothing on the CPU side would notice.

The split is the *first* sampled span, not the whole path. A multi-turn path
has several, and every one after the first depends on the turn before it, so
re-feeding the lot would verify the harness rather than the record.

Exit code 1 on any failure, so it drops into CI without a wrapper.
`--output report.json` writes the per-path verdicts.

A trajectory with nothing to re-feed is marked `-` and counted separately, not
as a pass: "3 of 3 verified" for three text-mode trajectories would be exactly
the overclaim this command exists to catch elsewhere.

### The GPU run

**Executed against `vllm serve` itself, 2026-09-18.** Qwen3-4B-Instruct-2507
on one H100 80GB, greedy: **4 trajectories, 4 paths, 57 prompt tokens re-fed,
192 completion tokens reproduced exactly.** Every path `accepted`, no
`diverged_at` set on any of them.

A sampled run was checked the same day at larger scale, with
`--tolerate-divergence` so the prompt is verified while the completion is
allowed to differ: **25 trajectories, 32 paths, 1027 prompt tokens accepted.**
That is the check worth running against an ordinary RL run, where nothing is
greedy -- it still proves the stored prompt is the one the engine saw, which is
the half capture cannot know on its own.

`vllm serve` serves this wire itself. `/inference/v1/generate` comes from
vLLM's own `scale_out` entrypoint, mounted by `api_server.py` for any
generate-capable model, and capture's `vllm` upstream kind speaks it — singular
`token_ids`, `choices[0]` back, `X-Session-ID` for affinity. Nothing sits in
between.

```bash
# 1. the engine under test -- no wrapper, just vLLM
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 9500

# 2. capture a greedy run through it
skyrl-capture serve --mode tokens --upstream-type vllm \
  --upstream-url http://127.0.0.1:9500 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --record-dir ./traces
# ... run your harness with temperature 0 ...

# 3. hand the prompts back
uv run python -m tools.verify.cli --record ./traces --run-id <run> \
  --engine-type vllm --engine-url http://127.0.0.1:9500 \
  --model Qwen/Qwen3-4B-Instruct-2507
```

The URL is the server's root: the `vllm` kind appends `/inference/v1/generate`.

> An earlier run of this check went through a wrapper script
> (`scripts/vllm_token_engine.py`), which put vLLM behind capture's own batched
> `tokens` wire because `/inference/v1/generate` had not been noticed. That
> script is deleted, and the numbers above come from the invocation shown here
> -- vLLM's own endpoint, nothing in between.

```
ok     tr_01M2S4QX44T1D3M49ZWA8V7MSC  accepted=1
ok     tr_01M2S4QXK5KJV2BZY4GZ9Q515E  accepted=1
ok     tr_01M2S4QY1ZSKKXAMDQE3SDJEYQ  accepted=1
ok     tr_01M2S4QYGT287YBQG119WS6SJX  accepted=1
4 of 4 trajectories verified against the engine.
```

**Greedy is not optional.** The check is that the engine is a function of its
input; with temperature, a mismatch means nothing. For a record captured from a
sampled run, `--tolerate-divergence` keeps the acceptance half and drops the
equality half, which is still worth running and is a weaker claim.

**The negative control matters more than the pass.** Rewriting a single prompt
token in one trajectory's record — `525` to `9999`, in a message the model was
given — made the same run fail, on the same weights:

```
tr_01M2PV8K9FFEMR07MCVSTX8PJ0-p0000: the engine produced a different completion
from token 5 of 39. The stored prompt is not the prompt that produced the
stored completion
8 of 9 trajectories verified against the engine. 1 FAILED.     (exit 1)
```

One token in, five tokens to divergence. That is the class of bug no amount of
internal consistency checking can see, caught.

The same command runs against the mock engine on every push, in the
`verify-against-an-engine` job, because the mock's completion is a
deterministic function of its prompt — the same claim at CI cost. And
`tests/test_verify.py` corrupts a record on purpose, so the refusal is tested
rather than assumed.

### What it cannot check

- **Text mode.** No token ids, so there is no prompt to hand back. Paths come
  back `skipped`, not `passed`.
- **A fully replayed path.** Nothing was sampled, so nothing reproduces.
- **Anything about the reward.** That is the trainer's business, and the
  viewer's job stops at *is this record right*.
