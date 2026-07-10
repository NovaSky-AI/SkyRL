# Harbor recipes on the Arctic RL backend — design

Companion doc to PR **NovaSky-AI/SkyRL#1879**. Explains what changed and
why so the diff (four new files + three tiny edits) can be reviewed on
intent rather than mechanics.

## Goal

Let any Harbor recipe opt into the Arctic RL backend the same way stock
SkyRL recipes do — one CLI flag, no changes to Arctic Platform, no
changes to Harbor:

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

Additionally, upstream SkyRL recently removed its OpenAI HTTP shim
(`inference_engine_client_http_endpoint.serve`) because vLLM's own
`VLLMServerActor` now exposes an OpenAI endpoint natively. That shim is
what our earlier design leaned on; the current PR replaces it with a
self-contained FastAPI app in `openai_bridge.py` so nothing here depends
on an internal SkyRL module that can move again.

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

Two new components: the entrypoint that wires `HarborGenerator` +
`ArcticRLExp` together, and the adapter that bridges HTTP <-> Arctic
in-process generate.

## `openai_bridge.py` — `ArcticInferenceEngineAdapter`

Concretely implements
`skyrl.backends.skyrl_train.inference_servers.base.InferenceEngineInterface`
so `HarborGenerator` can call the two methods it actually needs
(`get_endpoint_url()`, `finish_session()`) on this object directly, with
no wrapper.

Responsibilities:

1. Build a FastAPI app (`/v1/chat/completions`, `/v1/completions`,
   `/v1/models`, `/health`).
2. Serve it via `uvicorn.Server` in a daemon thread on a port we
   reserve (`ARCTIC_HARBOR_SHIM_PORT`, default 8000, auto-bumps if
   busy).
3. Translate OpenAI requests into `arctic_client.generate(prompts,
   sampling_params)` calls and translate results back into
   OpenAI-shaped JSON.
4. No-op or best-effort forward the rest of the interface — weight
   sync, sleep/wake, pause/resume, etc. all flow through the Arctic RL
   server via `arctic_client`, not through this adapter.

### Response shape (why the earlier smoke was returning empty rollouts)

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

Without ANY of those, `terminus-2._collect_subagent_rollout_details`
short-circuits, `chat.rollout_details` stays empty, and
`HarborGenerator._harbor_agent_loop` fails every trajectory with
"empty/missing rollout_details". The bridge emits all three; a
standalone parser test (`/tmp/test_shim_response.py` in the run log)
verifies the round-trip through LiteLLM.

### Two Qwen3-specific subtleties (documented in-place)

1. **`_sanitize_assistant_text`** — Qwen3's BPE fuses three consecutive
   `\n` into token `1406`. If the raw completion starts with `\n\n`,
   then concatenated with the chat template's trailing `\n` after
   `<|im_start|>assistant` the assistant message re-tokenizes with the
   fused token instead of `[198, ...]`. That trips SkyRL's
   `get_response_ids_and_loss_mask_from_messages` assertion at training
   time. `.lstrip()` suppresses that path.
2. **`reasoning_content: None` on every choice.message dict.** LiteLLM's
   `_parse_content_for_reasoning` will otherwise strip a
   `<think>...</think>` prefix out of `content` and leave a bare
   `\n\n{JSON}`, which re-triggers the same three-newline fusion.

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

## Small edits to existing files

### `examples/train_integrations/harbor/entrypoints/main_harbor.py`

+13 lines: same `trainer.override_entrypoint=` peek that
`skyrl.train.entrypoints.main_base` already uses. Direct copy of the
pattern; nothing Harbor-specific.

```python
def main() -> None:
    for arg in sys.argv[1:]:
        if arg.startswith("trainer.override_entrypoint="):
            override = arg.split("=", 1)[1]
            from importlib import import_module
            return import_module(override).main()
    ...
```

This is the entire user-facing entry point: any existing Harbor
launcher becomes an Arctic launcher by appending
`trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint`
to its CLI.

### `pyproject.toml`

**No change.** The stale `arctic-rl` extra that existed on this
branch's fork was already dropped in upstream's own arctic_rl refactor
(PR #1859). Nothing in this PR touches `pyproject.toml`.

### README updates

- `integrations/arctic_rl/README.md`: new "Harbor recipes" section
  pointing at both reference launchers + this design doc; updated
  file-layout block.
- `examples/train_integrations/harbor/README.md`: new "Arctic RL
  backend" subsection cross-linking the arctic_rl README and the
  Harbor-side companion launcher.

## Reference launchers

Two launchers are provided; both invoke Harbor's own `main_harbor`
with the override flag, they only differ in packaging:

1. **Harbor-side companion** —
   `examples/train_integrations/harbor/run_codecontest_arctic.sh`.
   Line-for-line parity with the FSDP baseline
   `examples/train_integrations/harbor/run_codecontest.sh`: same
   Qwen3-8B / 8-GPU recipe, same knob layout, drop-in comparison
   for reviewers. Discoverable from the Harbor folder next to the
   FSDP script.

2. **Env-driven variant** —
   `integrations/arctic_rl/examples/run_codecontest_arctic_harbor.sh`.
   Structure mirrors `run_gsm8k_grpo_4gpu.sh` /
   `run_bird_grpo_*.sh`; defaults smoke on Qwen3-0.6B / 4 GPUs so
   first-launch feedback lands in ~5 min from a cold uv cache, scale
   up via `NUM_GPUS=... MODEL=...`.

Both use the same uv incantation:

- `uv run --isolated --with arctic-platform --with 'arctic-inference[vllm]' --with liger-kernel --with 'transformers==4.57.6' --with '<flash-attn wheel>'` — no `uv sync` required.
- Every knob is a shell variable with a safe default, so a user's own
  launcher can `source` this one and override.
- `MAX_CONCURRENCY` default is a workable value for the Daytona free
  tier (10 concurrent CPUs → ≤ 8 concurrent trials). Bump when on a
  paid tier.
- Harbor's step-wise training requirements
  (`generator.step_wise_trajectories=true`,
  `generator.merge_stepwise_output=true`) are baked in — matches
  upstream's `examples/train_integrations/harbor/run_codecontest.sh`.

### Configuration

The bridge takes host/port from environment variables so we add no new
SkyRL config fields:

| env var                              | default    | notes                                                      |
|--------------------------------------|------------|------------------------------------------------------------|
| `ARCTIC_HARBOR_SHIM_HOST`            | `0.0.0.0`  | bind address (needed on 0.0.0.0 for sandboxes to reach us) |
| `ARCTIC_HARBOR_SHIM_PORT`            | `8000`     | starting port; auto-bumps if busy                          |
| `ARCTIC_HARBOR_SHIM_ADVERTISED_HOST` | `127.0.0.1`| host clients (sandbox) should call; set to the driver node IP if the sandbox runs off-node |

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

- **`use_zorro=false` in the reference launcher.** ZoRRo's shape
  assertions on the prompt-group dedup path aren't verified through the
  OpenAI shim yet (Harbor produces heterogeneous, agentic prompts that
  don't share the "same problem × N samples" prefix pattern ZoRRo
  optimizes for). Safe default is off; users can flip when the prompt
  distribution is amenable.
- **Sandbox-provider concurrency.** The Daytona free tier caps
  concurrent sandboxes at 10 CPUs. The launcher defaults keep
  `train_batch_size × n_samples_per_prompt ≤ 32` and
  `MAX_CONCURRENCY ≤ 64` so paid-tier users can dial up. On the free
  tier, `run_codecontest_arctic_harbor.sh` runs but back-pressures on
  sandbox creation; see the sandbox tier note in the README.
- **Chat template lookup.** The adapter re-applies the chat template
  server-side so training-time re-tokenization sees the same prompt
  bytes as sampling. The template path is read from
  `generator.inference_engine.engine_init_kwargs.chat_template`; both
  Harbor and Arctic must be pointed at the same file.

## Alternatives considered

- **In-process transport (no HTTP shim).** Rejected because Harbor's
  agent runs inside a sandbox VM, not in the driver's process — it
  needs a real network endpoint.
- **Reuse SkyRL's `inference_engine_client_http_endpoint`.** Was the
  earlier design; upstream removed that module along with the
  `enable_http_endpoint` config surface. Reimplementing the FastAPI app
  in-integration keeps this PR insulated from any further shim churn.
- **Fork `HarborGenerator` to call `arctic_client` directly.** Rejected
  because it would fork Harbor's `terminus-2` LiteLLM path — non-trivial
  to keep in sync with upstream Harbor changes. The adapter is a small,
  isolated module and preserves Harbor's LiteLLM/tool-call path
  verbatim.
- **New extra in `pyproject.toml`.** Rejected: `uv` still evaluates the
  full `[project.optional-dependencies]` graph under `--isolated`, so
  an `arctic-inference[vllm]==0.18` pin would block any launcher that
  also needs harbor / megatron / fsdp (vLLM 0.19+). The launcher's
  `uv run --isolated --with ...` pattern (already used by every other
  Arctic launcher) sidesteps this without any pyproject churn.
