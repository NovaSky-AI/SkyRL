# Token capture

In text mode the proxy watches an exchange between your workload and a
provider and stores the text. In tokens mode it **owns the renderer**: it turns
messages into token IDs itself, calls a token-in/token-out endpoint, and stores
the exact prompt and completion token IDs, a per-token sampled mask, logprobs,
and MoE routed experts.

Use it when the tokens matter — RL on a policy you serve yourself, where
re-tokenizing text at training time would not reproduce what the model saw.

Your workload does not change. It keeps speaking OpenAI Chat Completions and
cannot tell the difference.

## Setup

```bash
uv sync --extra tokens          # brings the tokenizer and the renderer

skyrl-capture serve --mode tokens --upstream-type tokens \
  --upstream-url http://your-router/generate \
  --model glm-5.2 \
  --tokenizer zai-org/GLM-5.2 \
  --max-model-len 32768

skyrl-capture run --project rl-run -- python agent.py
```

`--tokenizer` is a Hugging Face name. The proxy renders with it through
[Prime Intellect's `renderers`](https://github.com/PrimeIntellect-ai/renderers),
which is also what SkyRL's token-in/token-out proxy renders with — so a capture
and a SkyRL rollout build the same prompt. It knows the per-model template
differences that matter here: where reasoning content goes, how tool calls are
framed, which tokens are scaffold.

`--tokenizer builtin`, or leaving it unset, selects an offline ChatML renderer
over a byte tokenizer. It downloads nothing and is what the test suite and the
benchmark use; it is not for real models.

Every trajectory this process creates has `mode: tokens`, because the mode
follows the upstream and the upstream is fixed when the process starts.
Everything else is the same as text mode: routes, the graph, labels and
annotations, exports, the UI.

### Which wire the engine speaks

`type: "tokens"` sends a batch of one -- `prompt_token_ids`, `session_id` in the
body -- and expects parallel `response_ids` / `response_logprobs` /
`stop_reasons` back. That is capture's own shape, and an engine exposing it
needs nothing further.

**`type: "vllm"` is vLLM's own wire**, and is what `vllm serve` mounts:
`/inference/v1/generate`, a singular `token_ids` request, a `choices[0]`
response, session affinity in the `X-Session-ID` header, and `SamplingParams`
rules -- five differences from the shape above, any one of which fails the turn.
Point it at the server's root and the kind appends the path:

```bash
skyrl-capture serve --mode tokens --upstream-type vllm \
  --upstream-url http://engine:8000 \
  --model Qwen/Qwen3-4B-Instruct-2507 --tokenizer Qwen/Qwen3-4B-Instruct-2507
```

**A wire that is not a protocol is not shipped here.** capture speaks protocols,
not engines: `vllm` is here because the payloads are vLLM's, served by vLLM. A
trainer that *forks* that endpoint onto its own path -- SkyRL serves the same
shape at `/skyrl/v1/generate` -- subclasses it in its own repo and overrides one
attribute:

```python
class SkyRLTitoProtocol(VLLMTitoProtocol):
    name = "skyrl"
    generate_path = "/skyrl/v1/generate"
```

Carrying every trainer's variant here would be the coupling the registry exists
to avoid. So the registry is open.

Contributing one is a subclass and a `register()` call:

```python
# my_project/capture_upstream.py
from skyrl_capture.tito.upstream import TitoProtocol, register

class MyEngine(TitoProtocol):
    name = "my-engine"

    def url(self, url):                  # where a generate call goes
        return f"{url.rstrip('/')}/v2/infer"

    def request(self, *, prompt_token_ids, sampling_params, model, session_id):
        return {...}, {"X-Session-ID": session_id}     # (body, extra headers)

    def response(self, body):            # -> completion ids, logprobs, stop reason
        return {...}

register(MyEngine())
```

This is the token registry, not the text one: an engine wire is what capture
*calls*, a text protocol is what clients *speak to* capture, and neither
plugin can see the other's types.

Importing the module is what registers it, which is enough when the proxy runs
in your own process (`CaptureService`). A separate `skyrl-capture serve` imports only
this package, so point it at yours:

```bash
skyrl-capture serve --port 8080 --upstream-module my_project.capture_upstream
```

Repeatable, `-u` for short. A module that will not import fails at startup
rather than surfacing later as a request the proxy cannot forward.

A worked example is SkyRL's, in that repo:
`examples/train_integrations/harbor/skyrl-capture/upstream.py`.

## What gets stored per message

The graph has the same shape as text mode — one node per message — and each
node additionally carries the tokens that message introduced.

A user or tool message introduces tokens the model did not produce:

```json
{
  "message": {"role": "user", "content": "solve this"},
  "tokens": {
    "token_ids": [151644, 872, 198, "..."],
    "sampled_mask": [false, false, false, "..."],
    "logprobs": [0.0, 0.0, 0.0, "..."],
    "sampled_start": null,
    "routed_experts": null
  }
}
```

An assistant node is the interesting one, because it is **not purely
sampled**:

```json
{
  "tokens": {
    "token_ids":    [151644, 77091, 198,  15496, 11, 1917],
    "sampled_mask": [false,  false, false, true, true, true],
    "logprobs":     [0.0,    0.0,   0.0,  -0.31, -0.02, -1.44],
    "sampled_start": 3
  }
}
```

The first three tokens are the chat template's **generation scaffold** —
`<|im_start|>assistant\n`. The template emitted them, not the model. They are
part of the prompt the model saw, not part of what it produced.
`sampled_start` is the boundary:

- the call's prompt is every prior node's tokens, plus this node's tokens
  before `sampled_start`
- the completion is this node's tokens from `sampled_start` onward

Train past that boundary and you are training on tokens no model generated.
Exports apply it for you: the `loss_mask` in `token-samples` is these masks
concatenated down the path, with any node that the train-once rule or
`--mask-abandoned` suppressed zeroed out.

## Prefix reuse, and the one thing your client must do

On each turn the proxy tries to **extend the previous turn's exact tokens**
rather than re-render the conversation. It matches the longest message prefix
in the graph, resumes from a real inference boundary — an assistant node that
ended a completed call, under the same tool set — and asks the renderer to
continue from that exact prompt and completion.

How much it saves grows with the conversation:

| messages | context tokens | re-render | extend | |
| --- | --- | --- | --- | --- |
| 11 | 201 | 1.03 ms | 0.10 ms | 10x |
| 41 | 767 | 3.33 ms | 0.11 ms | 30x |
| 101 | 1,907 | 7.79 ms | 0.14 ms | 57x |
| 201 | 3,808 | 15.03 ms | 0.20 ms | 77x |

Re-rendering is linear in the whole context; extending is flat.

**Reuse engages only if your client replays the model's output verbatim.** Send
back anything else — a reworded assistant message, a summary, and in particular
an assistant turn with `reasoning_content` stripped, which several SDKs do by
default — and the proxy falls back to a full render and the trajectory forks.

The same turn, with and without the reasoning block the proxy returned:

```
reasoning replayed:  extends, 58 tokens reused, 2 new nodes
reasoning dropped:   full render,           3 new nodes, and the trace forks
```

Both store exact tokens. The second pays for a full render every turn and
splits one conversation into two branches. Return the assistant message whole,
including `reasoning_content`.

Whatever happens, the tokens are checked before anything is stored. The prompt
must reproduce the committed node deltas token for token; if it does not, the
turn fails rather than storing a prefix that disagrees with what inference
received.

Turns within one trajectory are **serialized**, because turn N+1's prefix is
turn N's output. Different trajectories run in parallel, over a pool of
renderers. The in-memory trace is a cache of a cache: if a turn lands on a
process that has never seen the trajectory, it is rebuilt from the graph that
process holds. A process that never saw the trajectory at all cannot rebuild
it, which is why routing must pin a trajectory to one replica.

## What your endpoint must accept and return

The proxy POSTs to `--upstream-url`:

```json
{
  "prompt_token_ids": [[151644, 8948, "..."]],
  "sampling_params": {
    "max_tokens": 512,
    "logprobs": true,
    "temperature": 0.7,
    "stop_token_ids": [151645]
  },
  "model": "glm-5.2",
  "session_ids": ["tr_..."],
  "session_id": "tr_...",
  "cache_salt": "run-7"
}
```

and expects:

```json
{
  "response_ids": [[15496, 11, 1917]],
  "response_logprobs": [[-0.31, -0.02, -1.44]],
  "stop_reasons": ["stop"],
  "rollout_expert_indices": [[[[0, 3], [5, 7]], "..."]]
}
```

**Selected-token logprobs are required.** A response without them is rejected
and the turn fails; it is not stored without them.

`rollout_expert_indices` is optional, but if present it must cover the prompt
*and* the completion, or the turn is rejected.

`session_id` is there so a router can keep this trajectory's prefix cache warm
on one worker.

This is the contract a SkyRL-style router speaks. A vanilla OpenAI-compatible
server does not — it takes messages and returns text. Point a tokens upstream at
an endpoint that accepts token IDs.

## What happens when a turn cannot be attributed

Text capture is fail-open: if the upstream answered, your workload gets the
answer even when capture cannot keep up.

Tokens mode verifies and commits the tokens before it responds, because the
proxy produced them and a response it could not attribute would otherwise reach
training looking exact.

If that check fails, **you still get the completion** — the engine generated it
and you paid for it — with an `x-capture-status: poisoned` response header. What
stops is the trajectory: it accepts no further turns, and a later one is
refused with `410`.

That is the part worth understanding. The failed turn is not in the graph, so a
next turn would render its assistant message out of the history you replay and
record model-generated tokens as `author: client` — exactly the silent
corruption the check exists to catch. A stopped trajectory with a visible gap is
recoverable; a trajectory that quietly relabels a sampled turn is not.

**Durability is fail-closed too, and this is the part that changed.** A token
response does not close until its exchange is on disk. The append is queued as
soon as the full outcome exists and runs while the response is being sent, so
TTFT and chunk cadence are unchanged; only the final empty frame waits on it —
about half a millisecond per turn against the mock engine.

If that append cannot be made durable, the connection fails rather than closing
cleanly. The client's ordinary retry of the unchanged request is the recovery,
which is why a token route needs no request id and no capture header: a retry
is just another request.

What a crash can leave is the one thing persistence and network delivery cannot
make atomic — an exchange that is durable and whose delivery was never
confirmed. It is kept, flagged `delivery_uncertain`, shown in the graph, marked
untrainable in a training export, left out of a replay, and resolved only by
evidence: a later request whose own history contains that assistant output,
which confirms it and reuses its exact token path. A request that does *not*
continue from it says nothing about it, because repeated identical prompts are
valid resampling.

## Streaming

The token endpoint returns a whole completion, so a client that sets
`stream: true` gets an SSE stream reconstructed from the sampled tokens after
generation finished. The content is identical to the non-streaming path; only
the arrival timing differs. The exchange records `stream_synthesized: true`, so
nothing mistakes that cadence for a measurement of the model's real timing.

## Exporting for training

```bash
skyrl-capture export --project rl-run --format token-samples \
  --mask-abandoned --output dataset.jsonl
```

One row per root-to-leaf path: `input_ids` for the whole conversation, a
`loss_mask` of the same length marking the positions the model produced, and
`rollout_logprobs` and `rollout_expert_indices` on the same index. The
trajectory's `annotations` come along, so every branch of a branched trace
carries the same outcome.

Rows are never dropped, and each sampled node is a training target in exactly
one row unless you pass `--allow-repeated-targets`. That matters on a branched
trajectory: the reply the branches share appears in every row, and trains in
one of them.

See [exports.md](exports.md) for the full record, and for what
`--mask-abandoned` does and does not do — it is not a way to pick a winner out
of parallel samples.

## Limits

- **`n > 1` is rejected.** One turn commits exactly one sampled assistant node.
  `n > 1` would need n sibling nodes and a defined order between them.
- **`max_tokens` is clamped** to `max_model_len - prompt_tokens`. A prompt that
  leaves no room is rejected with `context_length_exceeded` rather than sent.
- **`transformers` must be 4.x.** The `tokens` extra pins it. Installing
  `transformers>=5` alongside tokens mode fails at the first render, with an
  `ImportError` naming `AutoTokenizer`.
- **Long contexts cost work proportional to the context, every turn.** A
  1,015,477-token trajectory over 700 turns costs 15.2 s of proxy CPU in total,
  or 43 ms on its last turn. Budget for it alongside the inference call, which
  at that context length is itself seconds.
  [design/tokens_log.md](design/tokens_log.md) has the breakdown.

## Further reading

[design/tokens_log.md](design/tokens_log.md) records what was tested and measured.
[design/token-capture-parity.md](design/token-capture-parity.md) lists the
requirements this path has to meet that a summary of "token-in/token-out"
would miss, and where each is implemented.
