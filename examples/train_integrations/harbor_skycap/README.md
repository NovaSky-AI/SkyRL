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
the idle TTL, the renderer pool size, which paths train and how agents inside
remote sandboxes reach skycap ([below](#agents-inside-remote-sandboxes)).

## Which paths train

`skycap.train_paths` picks the rows of each rollout:

- `all` (default): every root-to-leaf path, each sampled message trained once.
  A reply the harness discarded and asked again for (mini-swe-agent on a format
  error) is a dead-end path, and trains with the rollout's advantage too.
- `final`: only the path to the reply of the rollout's last model call, the
  conversation the harness ended with. One row per rollout; nothing off it
  trains.
- `pkg.module:function`: a custom rule, a function of skycap's context graph to
  rows, each a path and the model nodes on it to train. skycap builds the
  tokens, loss masks and routes, and checks the rows. The module must be
  importable on every node, since each skycap server imports it at start. A
  rule that raises masks that rollout without retrying the trial.

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

## Agents inside remote sandboxes

Terminus-2 calls the model from the trainer's process, so it reaches skycap at
each server's own URL. Harbor's installed agents (mini-swe-agent, Claude Code,
...) call it from inside their Daytona or Modal sandbox, which can't reach the
cluster. `skycap.exposure` serves each server's harness routes to them, through
skycap's exposure ([skycap's README](../../../skycap/README.md#expose-a-server-to-remote-harnesses)):

```bash
# A Cloudflare quick tunnel per server: outbound internet only, for development.
  harbor_trial_config.agent.name=mini-swe-agent skycap.exposure.type=cloudflare
# An address the sandboxes route to; server i listens on port + i (frp on a public VM, or the node's own).
  skycap.exposure.type=external_host skycap.exposure.kwargs.host=203.0.113.7 skycap.exposure.kwargs.port=11500
```

Only the harness routes are reachable that way, never the control plane. An
installed agent gets its trajectory's route on the exposed URL in its sandbox's
environment (`OPENAI_API_BASE`, `HOSTED_VLLM_API_BASE`, and placeholder keys);
Terminus-2 keeps the server's own URL. A quick tunnel takes at most 200 calls in
flight, and a call whose reply hasn't started after about 125 s fails, so use
`external_host` for many agents or long replies.

## Records and W&B

Each skycap server writes its trajectories to `skycap.record_dir` on its own
node. `skycap.record_mirror=s3://bucket/prefix` (any fsspec URL; install `s3fs`
or `gcsfs`) has every server also copy them there, in the background.
`skycap.record_mirror_config` takes the mirror's options, e.g.
`'+skycap.record_mirror_config={exclude: [experts, sampling_mask]}'` to keep
routed experts and sampling masks out of the remote copy.

With `trainer.logger=wandb`, each step is indexed as a version of the artifact
`skycap-records-train-<run id>`, aliased `train-step-N` and `latest`. It holds
a `step.json` with a row per trajectory opened in the step (every attempt,
with `trained`, `superseded` and the record's `path` and `mirror`), and, for
mirrored records, a W&B reference to each document (no bytes are copied). A
local-only record is in the index only. `skycap.wandb.phases=[train,eval]`
indexes eval into `skycap-records-eval-<run id>`; `skycap.wandb.enabled=false`
turns the index off. Logging runs on a background thread and never fails a
step: W&B errors and timeouts are logged and counted, and at the end of
training it waits up to two minutes for what is queued.

## How it fits

| Piece | What it does |
| --- | --- |
| `entrypoints/main_harbor_skycap.py` | Starts the skycap servers in token mode, in front of the router, from the run's config. Stops them at the end, which writes every trajectory still in memory. |
| `servers.py` | The server pool: one Ray actor per server, each running a `skycap.CaptureService` on a port of its own. skycap builds how calls reach the model from the options; the integration supplies only its engine wire. |
| `engine.py` | `SkyRLEngine`: skycap's vLLM wire on `/skyrl/v1/generate`, with packed routed experts and sampler support decoded by SkyRL's own `generate_wire`, and sessions released at `/finish_session`. |
| `harbor_generator.py` | Per trial: create a trajectory, point the agent's `api_base` at it, run Harbor, and `finish` with the reward to get the samples. A retry gets a fresh trajectory. |
| `record_index.py` | `SkycapRecordIndex`, a trainer callback: the per-step W&B index of the trajectories the generator logged in a `RecordLog`. Nothing in it is Harbor's. |
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

## Limits

- **Sampler support** (`enable_return_sample_support_set`) is passed through,
  padded to `top_k`.
- **No W&B record index under the fully-async trainer.** It fires no callbacks.

## Tests

```bash
uv run --isolated --extra skyrl-train --extra harbor --extra skycap --extra dev pytest tests/integrations/harbor_skycap
```

A fake Harbor trial talks HTTP to a real skycap server, which calls a mock of
SkyRL's router. No GPU, sandbox or tokenizer download is needed.
