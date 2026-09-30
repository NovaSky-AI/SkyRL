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
training row.

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
the idle TTL, the port and the renderer pool size.

## Records and W&B

Each skycap server writes its trajectories to `skycap.record_dir` on its own
node. `skycap.record_mirror=s3://bucket/prefix` (any fsspec URL; install `s3fs`
or `gcsfs`) has every server also copy them there, in the background.

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
