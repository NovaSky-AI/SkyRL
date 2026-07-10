# Harbor recipes on the Arctic RL backend — design

Companion doc to PR **NovaSky-AI/SkyRL#1879**.

## Goal

Let any Harbor recipe opt into the Arctic RL backend the same way stock
SkyRL recipes do — one CLI flag, no changes to Arctic Platform or
Harbor:

```
trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint
```

Prior state:

- `integrations.arctic_rl.entrypoint` (already in `main`) dispatches
  from `main_base.py` and gets `ArcticGenerator`, which calls
  `arctic_client.generate()` directly (no HTTP).
- Harbor recipes use `main_harbor.py` and `HarborGenerator`, which
  drives `terminus-2` inside a sandbox VM that talks to the model over
  `POST /v1/chat/completions` via LiteLLM.

The two paths never met: Harbor's agent lives inside a Daytona/Modal
sandbox and can only reach the model over HTTP; Arctic's sampling side
is a `ReplicaPool` reached only via async
`arctic_client.generate(prompts, sampling_params)`. This PR is the
adapter that closes that gap without touching either side.

Upstream SkyRL recently removed its OpenAI HTTP shim
(`inference_engine_client_http_endpoint.serve`) — vLLM's `VLLMServerActor`
now serves OpenAI natively. This PR replaces the shim we used to lean on
with a self-contained FastAPI app in `openai_bridge.py`, so nothing here
depends on an internal SkyRL module that can move again.

## Change surface

```
+ integrations/arctic_rl/harbor_entrypoint.py    NEW   entrypoint + config
+ integrations/arctic_rl/openai_bridge.py        NEW   InferenceEngineInterface
                                                       + self-contained FastAPI shim
+ integrations/arctic_rl/examples/
      run_codecontest_arctic_harbor.sh           NEW   env-driven launcher
                                                       (0.6B smoke -> 8B)
+ integrations/arctic_rl/docs/HARBOR_DESIGN.md   NEW   THIS FILE
+ examples/train_integrations/harbor/
      run_codecontest_arctic.sh                  NEW   Qwen3-8B / 8-GPU parity
                                                       with run_codecontest.sh
~ examples/train_integrations/harbor/entrypoints/
      main_harbor.py                             +13   override_entrypoint peek
~ examples/train_integrations/harbor/README.md   +18   Arctic RL backend +
                                                       companion launcher
~ integrations/arctic_rl/README.md               +20   Harbor recipes section
```

No changes to `pyproject.toml`. No changes to any core `skyrl/` module.
No changes to `arctic-platform`, `arctic-inference`, or `harbor`. All
integration code lives under `integrations/arctic_rl/`.

## Architecture

```
                Single Ray cluster (Arctic in-process; HTTP only inside sandbox)
+---------------------------------------------------------------------------------------+
|                                                                                       |
|  SkyRL driver (num_gpus=0)                                                            |
|                                                                                       |
|  ArcticHarborExp  --> HarborGenerator --> Harbor Trial                                |
|                              |            (Daytona / Modal sandbox VM)                |
|                              |                     |                                  |
|                              |                     |  POST /v1/chat/completions       |
|                              |                     v  (via LiteLLM hosted_vllm/...)   |
|                              |          +---------------------------------+           |
|                              +--------->|  ArcticInferenceEngineAdapter   |           |
|                        get_endpoint_url |  - FastAPI on a daemon thread   |           |
|                        finish_session   |  - Implements                   |           |
|                                         |    InferenceEngineInterface     |           |
|                                         +---------------------------------+           |
|                                                     |                                 |
|                                                     |  await arctic_client.generate   |
|                                                     v                                 |
|  Arctic RL Ray actors:                                                                |
|    - DeepSpeedWorker (x N training GPUs)  --NCCL/CUDA-IPC-->  InferenceWorker         |
|    - InferenceWorker (x M sampling GPUs, ArcticInference vLLM)                        |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

## `openai_bridge.py` — `ArcticInferenceEngineAdapter`

Implements
`skyrl.backends.skyrl_train.inference_servers.base.InferenceEngineInterface`
so `HarborGenerator` can call the two methods it actually needs
(`get_endpoint_url()`, `finish_session()`) on this object directly, with
no wrapper.

Responsibilities:

1. Build a FastAPI app (`/v1/chat/completions`, `/v1/completions`,
   `/v1/models`, `/health`).
2. Serve it via `uvicorn.Server` in a daemon thread on a port reserved
   at startup (`ARCTIC_HARBOR_SHIM_PORT`, default 8000, auto-bumps if
   busy).
3. Translate OpenAI requests into `arctic_client.generate(...)` and
   the results back into OpenAI-shaped JSON.
4. Forward or no-op the rest of the interface. Weight sync, sleep/wake,
   and pause/resume all go through `arctic_client` directly, not this
   adapter.

### Response shape

Harbor's `terminus-2` uses `collect_rollout_details=True`, which makes
LiteLLM request `logprobs=True` + `extra_body.return_token_ids=True`
and then read three vLLM-specific response fields — matching vLLM's
`ChatCompletionResponseChoice` (`vllm.entrypoints.openai.protocol`):

- `choices[i].token_ids` (direct choice field, NOT nested inside
  `provider_specific_fields`) — LiteLLM auto-folds any non-`Choices`
  field into `provider_specific_fields`, so terminus-2 gets it at
  `choice.provider_specific_fields["token_ids"]`.
- `response.prompt_token_ids` (top-level) — read via `getattr` on
  LiteLLM's `ModelResponse`; that model inherits OpenAI's
  `BaseModel(extra="allow")` so unknown top-level keys survive.
- `choices[i].logprobs.content[j].logprob` — standard OpenAI shape,
  filled in from Arctic's per-token logprobs when the request enables
  them.

Without any of those, `terminus-2._collect_subagent_rollout_details`
short-circuits, `chat.rollout_details` stays empty, and
`HarborGenerator._harbor_agent_loop` fails every trajectory with
"empty/missing rollout_details". The bridge emits all three; a
standalone LiteLLM round-trip test in the run log verifies the shape.

### What the shim deliberately does *not* touch

`HarborGenerator` reads `completion_token_ids` / `prompt_token_ids` /
`logprobs` straight out of `rollout_details` — it never re-tokenizes
the response text. That means anything the shim does to `content`
(strip whitespace, inject `reasoning_content: None`, etc.) is invisible
to the trainer: gradients depend only on the token IDs vLLM's sampler
emitted, which the shim forwards verbatim from
`arctic_client.generate()`. The FSDP baseline runs the same recipe
with vLLM's native OpenAI endpoint and the same LiteLLM
`<think>`-stripping behavior; it trains fine without any of those
mutations, and so does the Arctic path. Keeping the shim thin means
every future SkyRL integration on top of Arctic inherits a bridge that
just shape-shifts, not one that carries workarounds for one caller's
quirks.

## `harbor_entrypoint.py` — `ArcticHarborExp` + `main()`

- **`ArcticHarborSkyRLConfig(SkyRLTrainConfig)`** — adds Harbor's
  `harbor_trial_config: dict` and points `generator` at
  `HarborGeneratorConfig` (defined by Harbor). `trainer` is bumped to
  `ArcticTrainerConfig` so `trainer.arctic_rl.*` is available. **No
  `from __future__ import annotations`** — SkyRL's
  `build_nested_dataclass` introspects `dataclasses.fields(cls)[i].type`
  and expects a concrete class, not a string.
- **`ArcticHarborExp(ArcticRLExp)`** — reuses `ArcticRLExp.__init__`
  (which pre-inits the Arctic client) verbatim. Only overrides:
  - `get_train_dataset` / `get_eval_dataset` → `HarborTaskDataset`.
  - `get_generator` → `HarborGenerator`, passing the adapter as
    `inference_engine_client` (so `HarborGenerator` picks up the shim
    URL via `.get_endpoint_url()` and bakes it into LiteLLM's
    `api_base`).
  - `_setup_trainer` → spins up the OpenAI HTTP shim **before**
    constructing `HarborGenerator` (order matters — `HarborGenerator`
    reads the URL at `__init__`), then hands the same
    `_ArcticInferenceEngineStub` upstream uses to `ArcticPPOTrainer` so
    colocated sleep / wake still route to the Arctic server.
- **`main()`** — mirrors `integrations.arctic_rl.entrypoint.main()`:
  pre-init the Arctic client on the driver, forward sandbox-provider
  credentials + `PYTHONPATH` onto Ray workers, dispatch the remote
  entrypoint task. Environment-variable prefixes forwarded:
  `ARCTIC_`, `WANDB_`, `DAYTONA_`, `MODAL_`, `E2B_`, `RUNLOOP_`,
  `GKE_`, `OPENAI_`, `HF_`.

## `main_harbor.py` edit

+13 lines: the same `trainer.override_entrypoint=` peek
`skyrl.train.entrypoints.main_base` already uses. Any existing Harbor
launcher becomes an Arctic launcher by appending the flag to its CLI.

```python
def main() -> None:
    for arg in sys.argv[1:]:
        if arg.startswith("trainer.override_entrypoint="):
            override = arg.split("=", 1)[1]
            from importlib import import_module
            return import_module(override).main()
    ...
```

## Reference launchers

Both invoke Harbor's own `main_harbor` with the override flag:

- `examples/train_integrations/harbor/run_codecontest_arctic.sh` —
  mirrors the FSDP baseline `run_codecontest.sh` (Qwen3-8B / 8 GPUs),
  discoverable next to it.
- `integrations/arctic_rl/examples/run_codecontest_arctic_harbor.sh` —
  env-driven, defaults to a Qwen3-0.6B / 4-GPU smoke, scales up via
  `NUM_GPUS=... MODEL=...`.

Both use `uv run --isolated --with arctic-platform --with 'arctic-inference[vllm]' --with liger-kernel --with 'transformers==4.57.6' --with '<flash-attn wheel>'`.
Harbor's step-wise training requirements
(`generator.step_wise_trajectories=true`,
`generator.merge_stepwise_output=true`) are baked into both so
prefix-merge invariance holds.

Shim host/port + `trainer.arctic_rl.*` knob tables live in the arctic
[`README.md`](../README.md) — no duplication here.

## Smoke result — Qwen3-8B, 8×H100, CodeContests

The end-to-end smoke on `upstream/main + this PR` (`arctic_smoke_*.log`):

```
Started: 'step'
Started: 'generate'
Finished: 'generate', time cost: 131.71s
Finished: 'sync_weights', time cost: 5.87s   (CUDA IPC, 8 replicas, 399 params)
Finished: 'step', time cost: 142.27s
{'actor/pg_loss': -0.0955..., ...}
```

Zero "empty/missing rollout_details" warnings, non-zero policy gradient
loss, weight sync using colocated CUDA IPC. Convergence run (~40 steps)
is queued after this smoke completes.

## Risks and known limits

- **`use_zorro=false` default.** ZoRRo's prompt-group dedup path
  assumes the "same problem × N samples" prefix pattern; Harbor's
  agentic prompts don't share prefixes. Off by default; flip when the
  prompt distribution allows.
- **Sandbox concurrency.** Daytona's free tier caps concurrent
  sandboxes at 10 CPUs. On that tier keep
  `train_batch_size × n_samples_per_prompt ≤ 16` and
  `MAX_CONCURRENCY ≤ 8`; the launcher runs unmodified but back-pressures
  on sandbox creation.
- **Chat template path.** The adapter re-applies the chat template
  server-side so training-time re-tokenization sees the same prompt
  bytes as sampling. Both Harbor and Arctic read the template from
  `generator.inference_engine.engine_init_kwargs.chat_template` — must
  point at the same file.

## Alternatives considered

- **In-process transport, no HTTP.** Harbor's agent runs inside a
  sandbox VM, so it needs a real network endpoint.
- **Reuse `inference_engine_client_http_endpoint`.** Upstream removed
  it along with the `enable_http_endpoint` config surface; reimplementing
  the FastAPI app in-integration insulates this PR from further churn.
- **Fork `HarborGenerator` to call `arctic_client` directly.** Would
  fork Harbor's `terminus-2` LiteLLM path — hard to keep in sync with
  upstream Harbor. The adapter preserves the LiteLLM path verbatim.
- **New extra in `pyproject.toml`.** `uv` still evaluates the full
  `[project.optional-dependencies]` graph under `--isolated`, so an
  `arctic-inference[vllm]==0.18` pin would block launchers that also
  need harbor / megatron / fsdp (vLLM 0.19+). `uv run --isolated
  --with ...` (used by every arctic launcher) sidesteps this.
