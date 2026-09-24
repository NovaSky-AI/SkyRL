# skycap

Trajectory capture for RL rollouts. A harness points its unchanged OpenAI client
at a per-trajectory URL. skycap records every model call into a context graph —
one node per message, where resamples, subagents, compaction and harness edits
are forks — and, when the trajectory finishes, returns one training sample per
root-to-leaf path.

skycap is its own package inside this repository and does not depend on
`skyrl`.

## Run a server

Text mode forwards to any OpenAI-compatible server and records what it sees:

```bash
cd skycap && uv sync
uv run skycap serve --upstream-url http://engine:8000/v1 --record-dir ./record
```

Token mode renders the prompt itself through
[`renderers`](https://github.com/PrimeIntellect-ai/renderers) and calls a
token-in/token-out engine, so the stored tokens are the ones inference saw,
with logprobs, routed experts and sampling masks:

```bash
uv sync --extra tokens
uv run skycap serve --mode tokens --engine skyrl --upstream-url http://router:8000 \
  --tokenizer Qwen/Qwen3-8B --max-model-len 32768 \
  --sampling-overrides '{"top_k": 50}' --sampling-mask --record-dir ./record
```

`--engine vllm` speaks vLLM's own `/inference/v1/generate`. `--engine skyrl`
speaks SkyRL's `/skyrl/v1/generate` and releases the router session when a
trajectory ends.

## Capture a rollout

```python
from skycap import CapturePool

pool = CapturePool(["http://capture-0:8080", "http://capture-1:8080"])
async with pool.trajectory({"task": "t1", "step": 3}) as trajectory:
    run_harness(base_url=trajectory.base_url)        # any OpenAI client
    result = await trajectory.finish({"reward": 1.0})

result.status          # "finished", or "failed" if a turn couldn't be attributed exactly
for sample in result.samples:
    sample.input_ids, sample.loss_mask, sample.logprobs
    sample.routed_experts, sample.sampling_mask
```

Creates go round-robin over the servers, and each trajectory's URL names its
server, so no router or load balancer is involved. An SDK retry
(`x-stainless-retry-count`) gets the original call's reply rather than a second
sample.

## The record

Each trajectory is written once, when it ends (finish, idle TTL or graceful
shutdown): a document, plus sidecars for tokens (with the text they decode to
and each token's offset in it), routed experts and sampling masks. The format
is specified in [`docs/format.md`](docs/format.md), which is what any reader,
such as the viewer, implements.

## Develop

```bash
cd skycap
uv sync --extra tokens
uv run pytest
```

Formatting and lint are the repository's (`bash format.sh` from the root).
