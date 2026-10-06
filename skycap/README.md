# skycap

Trajectory capture for RL rollouts. A harness points its unchanged OpenAI client
at a per-trajectory URL. skycap records every model call into a context graph —
one node per message, where resamples, subagents, compaction and harness edits
are forks — and, when the trajectory finishes, returns training samples: by
default one per root-to-leaf path.

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
uv run skycap serve --mode tokens --upstream-url http://engine:8000 \
  --tokenizer Qwen/Qwen3-8B --max-model-len 32768 \
  --sampling-overrides '{"top_k": 50}' --sampling-mask --record-dir ./record
```

The engine is vLLM, over its own `/inference/v1/generate`. Another engine's wire
is a subclass of `skycap.tokens.engine.VLLMEngine`.

By default a reply is parsed: a thinking model's reasoning comes back as
`reasoning_content`, and tool calls as `tool_calls`. Add `--use-raw-content`
when the harness was written against a vLLM server with no reasoning or tool
parser. Replies then match that server's: the completion's own text as
`content`, with thinking inline and tool calls unparsed, and
`reasoning_content: null`. A harness that replays `content` and drops
`reasoning_content` (Terminus-2 through LiteLLM, for example) then sends each
turn back unchanged, and a thinking model's history stays one path. With parsed
replies, every replayed turn would lose its thinking and fork the graph.

## Embed a server

A trainer can run a server in its own process instead, from the same options
`skycap serve` takes. It gets a thread and event loop of its own:

```python
from skycap import CaptureService

service = CaptureService(
    "http://engine:8000", mode="tokens", tokenizer="Qwen/Qwen3-8B",
    max_model_len=32768, record_dir="./record",
)
url = service.start()        # hand this to a CapturePool
...
service.stop()               # writes the trajectories still in memory
```

How a call reaches the model is built inside from those options. An engine
with another wire passes `engine=` (a `skycap.tokens.engine.VLLMEngine`
subclass), which is the one piece an embedder supplies.

## Expose a server to remote harnesses

A harness that runs outside the server's network, such as an agent inside a
remote sandbox (Daytona, Modal), can't reach the server's own URL. With an
exposure, the server listens a second time with the harness routes alone
(`/t/{id}/v1/chat/completions` and `/models`; no control plane is routed
there), and the exposure makes that listener reachable:

```bash
# A Cloudflare quick tunnel: outbound internet only, no account, development only.
uv run skycap serve --upstream-url http://engine:8000/v1 --expose cloudflare
# An address the sandboxes route to: this node's, or a relay's (frp on a public VM) forwarding the port here.
uv run skycap serve --upstream-url http://engine:8000/v1 \
  --expose external_host --expose-kwargs '{"host": "203.0.113.7", "port": 11500}'
```

```python
from skycap.exposure import load_exposure

service = CaptureService(..., exposure=load_exposure("cloudflare"))
service.start()              # returns once the tunnel is up
service.exposed_url          # https://<random>.trycloudflare.com
```

`create` then also returns each trajectory's route on the exposed URL,
`trajectory.exposed_base_url`, to hand to the remote harness. The trajectory id
in the path is what a caller must know.

| `--expose` | Reached at | Limits |
| --- | --- | --- |
| `cloudflare` | a random `https://*.trycloudflare.com` URL | development only: at most 200 requests in flight per tunnel (more get 429), a response that hasn't started within ~125 s gets 524, no SLA. Adds ~20 ms per call. |
| `external_host` | `http://{host}:{port}` | plain HTTP; the address has to route to this node |
| `pkg.module:Class` | whatever the class returns | an `skycap.exposure.Exposure` subclass, built with `--expose-kwargs` |

A custom way in implements three methods; the server binds the listener at
`bind()`, calls `start` once it serves, and `stop` before it stops:

```python
class Exposure:
    def bind(self) -> tuple[str, int]: ...      # default: a free loopback port, for a tunnel
    def start(self, harness_url: str) -> str: ... # make harness_url reachable; return the URL callers use
    def stop(self) -> None: ...
```

cloudflared is taken from `PATH` or downloaded once (Linux), and is tied to the
server's process: it stops when that process exits, however it exits.

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

### Which paths train

`finish` returns a sample per root-to-leaf path of the graph, each sampled
message a target in exactly one of them. `finish(..., paths="final")` returns
only the path to the reply of the trajectory's last model call, with every
sampled message on it a target: branches off it, such as a reply the harness
discarded and asked again for, don't train. It takes the last call to be the
main loop's; a harness whose side calls (a subagent, a summarizer) can return
after that needs a custom rule.

Those are the two built-in path rules (`skycap.paths`). A custom rule is a
function of the graph that returns rows, each a path from a root and the model
nodes on it to train. skycap builds the samples (tokens, loss masks, routes)
from the rows, after checking that each path follows parent links, every target
is a model node on its path, and no node is a target twice:

```python
from skycap.graph import MessageGraph
from skycap.paths import Row, final_path

def last_reply(graph: MessageGraph) -> list[Row]:
    """Only the final reply trains, with the conversation before it as context."""
    return [Row(row.path, row.targets[-1:]) for row in final_path(graph)]

service = CaptureService(..., path_rules={"last_reply": last_reply})  # or "my_rules:last_reply"
await trajectory.finish({"reward": 1.0}, paths="last_reply")
```

A server accepts only the rules it was given, so a request never imports code.
`skycap serve --path-rule last_reply=my_rules:last_reply` does the same;
without `NAME=`, a rule is named by its import path. The record's `samples`
field says which rule trained a trajectory, and a repeated `finish` can't ask
for another one. A rule that raises fails the `finish` with `PathRuleError`
(HTTP 500, `"code": "path_rule_failed"`); the trajectory is still ended and
written, without samples, and a later `finish` runs the rule again.
`pool.trajectory(meta, paths=...)` sets the rule a trajectory is finished
with, including when its block raises before finishing.

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
