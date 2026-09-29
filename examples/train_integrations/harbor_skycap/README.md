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
the idle TTL, the port and the renderer pool size. `skycap.wandb_artifact=true` uploads each
step's documents, without sidecars, as a version of the `skycap-records-<run id>` W&B artifact,
aliased `step-N` and `latest`.

## How it fits

| Piece | What it does |
| --- | --- |
| `entrypoints/main_harbor_skycap.py` | Starts the skycap servers in token mode, in front of the router, from the run's config. Stops them at the end, which writes every trajectory still in memory. |
| `servers.py` | The server pool: one Ray actor per server, each running a `skycap.CaptureService` on a port of its own. skycap builds how calls reach the model from the options; the integration supplies only its engine wire. |
| `engine.py` | `SkyRLEngine`: skycap's vLLM wire on `/skyrl/v1/generate`, with packed routed experts and sampler support decoded by SkyRL's own `generate_wire`, and sessions released at `/finish_session`. |
| `harbor_generator.py` | Per trial: create a trajectory, point the agent's `api_base` at it, run Harbor, and `finish` with the reward to get the samples. A retry gets a fresh trajectory. |
| `artifacts.py` | With `skycap.wandb_artifact`, uploads the step's `{id}.json.zst` documents, retried attempts included, after the step's trials end. An upload failure is logged and does not fail the step. |
| `compose.py` | Samples to a step-wise `GeneratorOutput`: a trial's paths are contiguous under its `TrajectoryID`, the last one marked `is_last_step` and carrying the reward. |

What's imposed on every call:
- **Sampling:** `generator.sampling_params` (`temperature`, `top_p`, `top_k`, `min_p`), since the trainer computes logprobs with them.
- **`cache_salt`:** derived from the policy's weight version. It rides in the request body and skycap forwards it.
- **Session id:** the trajectory id, sent to the router as `X-Session-ID`.

Masking is the sibling's:
- **Timeout or failed rollout:** the whole instance is masked.
- **Context-length stop:** trains with reward 0, unless overlong filtering is on.
- **Failed inside skycap** (e.g. an unattributable prompt): the trial isn't trained on.

## Limits

- **No R3 yet.** skycap records routed experts, but SkyRL's trainer refuses them
  with step-wise output. So the generator refuses
  `enable_return_routed_experts=true`, and the trainer change is a follow-up.
- **Sampler support** (`enable_return_sample_support_set`) is passed through,
  padded to `top_k`.
- **One skycap server per run.** The generator takes a list of URLs and spreads
  trajectories over them, for when servers are launched separately.

## Tests

```bash
uv run --isolated --extra skyrl-train --extra harbor --extra skycap --extra dev pytest tests/integrations/harbor_skycap
```

A fake Harbor trial talks HTTP to a real skycap server, which calls a mock of
SkyRL's router. No GPU, sandbox or tokenizer download is needed.
