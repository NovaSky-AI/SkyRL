# Harbor through skycap

Harbor runs **unmodified, in text space**. Each trial's agent is pointed at its
own [skycap](../../../skycap) trajectory URL. skycap renders every prompt,
calls SkyRL's router with token ids, and records a context graph. Training gets
the exact tokens and logprobs the engine sampled, without Harbor knowing about
tokens.

The difference from the sibling [`harbor`](../harbor) integration: there Harbor
collects per-turn token ids itself (`collect_rollout_details`), which is why
summarization is banned. Here a rewritten history is a new branch of the
graph, so **summarization is allowed**. Each root-to-leaf path becomes one
training row, unless `skycap.train_paths` says otherwise (below).

## Running

```bash
uv run --isolated --extra fsdp --extra harbor --extra skycap \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=policy \
  generator.step_wise_trajectories=true generator.merge_stepwise_output=false \
  trainer.algorithm.max_seq_len=32768 \
  data.train_data="['/path/to/harbor/tasks']"
```

The rest of the configuration is the sibling's: `harbor_trial_config` holds
Harbor's `TrialConfig`, with defaults from `../harbor/harbor_trial_config/default.yaml`.
`skycap.*` sets the record directory (default `{trainer.export_path}/skycap`),
the idle TTL, the renderer pool size, which paths train, and how agents inside
sandboxes reach skycap (`skycap.exposure`, [below](#agents-inside-sandboxes-exposure)).

## Which paths train

`skycap.train_paths` picks the rows of each rollout:

- `all` (default): every root-to-leaf path, each sampled message trained once.
  A reply the harness discarded and asked again for (mini-swe-agent on a format
  error) is a dead-end path, and trains with the rollout's advantage too.
- `final`: only the path to the rollout's last node, the conversation the
  harness ended with. One row per rollout; nothing off it trains.
- `pkg.module:function`: a custom rule, a function of skycap's context graph to
  rows, each a path and the model nodes on it to train. skycap builds the
  tokens, loss masks and routes, and checks the rows. The module must be
  importable on every node, since each skycap server imports it at start.

For example, the final path plus every discarded reply under 64 sampled tokens,
each as a row of its own (`skycap.train_paths=my_rules:final_and_short_discards`):

```python
from skycap.graph import MessageGraph
from skycap.paths import Row, final_path

def final_and_short_discards(graph: MessageGraph) -> list[Row]:
    rows = final_path(graph)
    final = set(rows[0].path) if rows else set()
    for leaf in graph.leaves():
        tokens = graph.nodes[leaf].tokens
        if leaf not in final and graph.nodes[leaf].author == "model" and tokens is not None:
            if len(tokens.token_ids) - tokens.sampled_start < 64:
                rows.append(Row(graph.path_to(leaf), [leaf]))
    return rows
```

Every row of a trial carries its reward. See [skycap's README](../../../skycap/README.md#which-paths-train)
for what a rule may return.

## mini-swe-agent and other installed agents

Terminus-2 calls the model from the trainer's process. An *installed* agent such as
mini-swe-agent runs inside its sandbox and calls the model from there, so the sandbox has to reach
skycap's harness routes over the network, through the exposure skycap is configured with
(`skycap.exposure`, [below](#agents-inside-sandboxes-exposure)); the generator refuses to start such an
agent without one. `agents.py` says, per Harbor agent, how it is pointed at its
trajectory's URL: Terminus-2 through its kwargs, mini-swe-agent through `OPENAI_BASE_URL` and the
other variables its LiteLLM client reads. An agent missing from the table is taken to be an
installed one speaking OpenAI's API.

```bash
export DAYTONA_API_KEY=...   # and WANDB_API_KEY, optionally
TASKS=code_contests-0000,code_contests-0002 \
  bash examples/train_integrations/harbor_skycap/run_codecontests_mini_swe_agent.sh
```

The recipe defaults to Qwen3.5-2B on one GPU with `mini_swe_agent.yaml`, mini-swe-agent's
tool-calling config with the Harbor task as its instance template. Things that differ from
Terminus-2:

- **Parsed replies.** mini-swe-agent acts through tool calls, so skycap answers with parsed
  `reasoning_content` and `tool_calls` (`skycap.use_raw_content` defaults per agent: on for
  Terminus-2, off otherwise). LiteLLM's `openai/` provider sends both back verbatim, so each turn
  extends the last one token for token.
- **Discarded replies.** A reply with no tool call is dropped from mini-swe-agent's history, and it asks
  again. In skycap's graph that reply is a dead-end branch: `skycap.train_paths=all` trains it with the
  rollout's advantage, and `final` doesn't.
- **Context overflow.** When a prompt leaves no room in the model's context, skycap refuses it with
  OpenAI's wording, so LiteLLM raises `ContextWindowExceededError` and the agent exits instead of
  retrying. skycap records the refusal and `finish` reports it (`context_length_exceeded`); the
  generator then treats the trial like Terminus-2's `ContextLengthExceededError`: reward 0,
  `context_length`, and overlong filtering applies.
- **Thinking.** `skycap.chat_template_kwargs.enable_thinking=true` samples a model whose template
  leaves thinking off (Qwen3.5-2B) with thinking on. `skycap.thinking_retention` (default `all`) keeps
  earlier turns' thinking in the prompt, so turns keep extending; `tool_cycle` strips it as Qwen's
  template does, re-rendering the prompt.

## How it fits

| Piece | What it does |
| --- | --- |
| `entrypoints/main_harbor_skycap.py` | Starts the skycap servers in token mode, in front of the router, from the run's config. Stops them at the end, which writes every trajectory still in memory. |
| `servers.py` | The server pool: one Ray actor per server, each running a `skycap.CaptureService` on a port of its own. skycap builds how calls reach the model from the options; the integration supplies only its engine wire. |
| `exposure.py`, `tunnel.py` | How agents inside sandboxes reach each server's harness routes: the harness-only gateway, the `Exposure` interface and its built-ins. `tunnel.py` runs a Cloudflare quick tunnel. |
| `engine.py` | `SkyRLEngine`: skycap's vLLM wire on `/skyrl/v1/generate`, with packed routed experts and sampler support decoded by SkyRL's own `generate_wire`, and sessions released at `/finish_session`. |
| `harbor_generator.py` | Per trial: create a trajectory, point the agent's `api_base` at it, run Harbor, and `finish` with the reward to get the samples. A retry gets a fresh trajectory. |
| `agents.py` | Per Harbor agent: how it is pointed at its trajectory's URL, the model name it is given, and whether it runs inside its sandbox. |
| `compose.py` | Samples to a step-wise `GeneratorOutput`: a trial's paths are contiguous under its `TrajectoryID`, the last one marked `is_last_step` and carrying the reward. |

What's imposed on every call:
- **Sampling:** `generator.sampling_params` (`temperature`, `top_p`, `top_k`, `min_p`), since the trainer computes logprobs with them.
- **`cache_salt`:** derived from the policy's weight version. It rides in the request body and skycap forwards it.
- **Session id:** the trajectory id, sent to the router as `X-Session-ID`.

Masking is the sibling's:
- **Timeout or failed rollout:** the whole instance is masked.
- **Context-length stop:** trains with reward 0, unless overlong filtering is on.
- **Failed inside skycap** (e.g. an unattributable prompt): the trial isn't trained on.

R3 (rollout routing replay) needs
`generator.inference_engine.enable_return_routed_experts=true` and
`trainer.policy.megatron_config.moe_enable_routing_replay=true`, with Megatron
and vLLM's `mp` backend, as for SkyRL's own generator. Each row carries routes
for its whole prompt and response, each from the forward pass that ran that
token. A trial whose trained path lacks routes is retried, then masked, and
counted in `generate/skycap/num_missing_route_trajectories`.

## Agents inside sandboxes: exposure

Terminus-2 calls the model from the trainer's process, so it reaches skycap at
each server's own URL. An agent installed in its sandbox (Daytona, Modal, ...)
calls the model from there, and needs a URL the sandbox network can reach.
`skycap.exposure.type` picks how each server is made reachable:

| `skycap.exposure.type` | Agents get | Needs, and limits |
| --- | --- | --- |
| `none` (default) | each server's own URL | Nothing: for agents that call from the trainer's process (Terminus-2). |
| `external_host` | `http://{host}:{port + i}` for server `i` | `skycap.exposure.host` (and `port`, default 11500): an address the sandboxes route to. Either the node's own (a public IP or a network peered with the provider's; all servers on that node, e.g. `skycap.placement_strategy=STRICT_PACK`) or a relay's, such as [frp](https://github.com/fatedier/frp) on a small public VM with one TCP forward per server (`VM:port+i` to that server's node). Firewall it: the traffic is plain HTTP. |
| `cloudflare` | a random `https://*.trycloudflare.com` URL per server | Outbound internet only, no account. Development only: a quick tunnel takes at most 200 requests in flight (run several servers past that), cuts a response that hasn't started within ~125 s, and has no SLA. `kwargs.timeout` and `kwargs.attempts` tune its startup. |
| `module:Class` | whatever the class returns | Your own `Exposure` subclass, below. |

For hundreds of concurrent agents, use `external_host` (directly, or through
a relay): a quick tunnel's in-flight cap and response timeout are hit first.

Only the harness routes (`/t/{id}/v1/chat/completions` and `/models`) are
exposed, through a gateway in front of each server; the control plane (create,
finish, read) stays private. The random trajectory id in the path is what a
caller must know. When anything is exposed, every agent, Terminus-2 included,
is given the exposed URL.

```bash
# Four servers behind an frp VM at 203.0.113.7 that forwards ports 11500-11503:
  skycap.num_servers=4 skycap.exposure.type=external_host skycap.exposure.host=203.0.113.7
# A quick tunnel per server:
  skycap.exposure.type=cloudflare
```

A new way in (a named Cloudflare tunnel, a hosted relay, Tailscale, ...) is a
subclass of `exposure.Exposure`, importable on every node. For example, named
Cloudflare tunnels created beforehand, one per server:

```python
import subprocess

from examples.train_integrations.harbor_skycap.exposure import Exposure

class NamedTunnels(Exposure):
    def __init__(self, tunnels: list[str], hostnames: list[str]) -> None:
        # Only store options: each server's actor builds its own instance.
        self.tunnels, self.hostnames = tunnels, hostnames
        self.process = None

    # bind(index) -> (host, port) of the server's gateway; by default a free loopback port.

    def start(self, gateway_url: str, index: int) -> str:
        # gateway_url serves this server's harness routes on this node, e.g. http://127.0.0.1:41234.
        self.process = subprocess.Popen(["cloudflared", "tunnel", "run", "--url", gateway_url, self.tunnels[index]])
        return f"https://{self.hostnames[index]}"  # agents get {this}/t/{trajectory id}/v1

    def stop(self) -> None:  # also runs when opening failed
        if self.process is not None:
            self.process.terminate()
```

```bash
  skycap.exposure.type=my_pkg.tunnels:NamedTunnels \
  skycap.exposure.kwargs.tunnels="[skycap-0,skycap-1]" \
  skycap.exposure.kwargs.hostnames="[skycap-0.example.com,skycap-1.example.com]"
```

The type, the import and the kwargs (against the constructor's signature) are
checked before Ray starts. Each server's Ray actor builds its own instance,
starts it after the server and stops it before the server.

## Limits

- **Sampler support** (`enable_return_sample_support_set`) is passed through,
  padded to `top_k`.

## Tests

```bash
uv run --isolated --extra skyrl-train --extra harbor --extra skycap --extra dev pytest tests/integrations/harbor_skycap
```

A fake Harbor trial talks HTTP to a real skycap server, which calls a mock of
SkyRL's router. No GPU, sandbox or tokenizer download is needed.
