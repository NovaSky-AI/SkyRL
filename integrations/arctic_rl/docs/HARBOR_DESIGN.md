# Harbor recipes on the Arctic RL backend — design

Companion doc to PR **NovaSky-AI/SkyRL#1879**. Explains what changed and why,
so the diff (five files, ~600 lines net) can be reviewed on intent rather
than mechanics.

## Goal

Let any Harbor recipe opt into the Arctic RL backend the same way stock
SkyRL recipes do — one CLI flag, no changes to Arctic Platform, no changes
to Harbor:

```
trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint
```

Prior state:

- `integrations.arctic_rl.entrypoint` (already in `main`) dispatches from
  `main_base.py` and gets `ArcticGenerator` (direct `arctic_client.generate()`
  calls, no HTTP shim).
- Harbor recipes use `main_harbor.py` and `HarborGenerator`, which drives
  `terminus-2` inside a sandbox that talks to the model over
  `POST /v1/chat/completions` via LiteLLM.

The two paths never met: Harbor's agent lives inside a Daytona/Modal
sandbox and can only reach the model over HTTP; Arctic's sampling side is
a `ReplicaPool` reachable only via async `arctic_client.generate(...)`.
This PR is the adapter that closes that gap without touching either side.

## Change surface (5 files)

```
+ integrations/arctic_rl/harbor_entrypoint.py     202 lines   NEW
+ integrations/arctic_rl/openai_bridge.py         358 lines   NEW
+ integrations/arctic_rl/examples/
      run_codecontest_arctic_harbor.sh            233 lines   NEW   reference launcher
+ integrations/arctic_rl/docs/HARBOR_DESIGN.md     THIS FILE  NEW
~ examples/train_integrations/harbor/entrypoints/
      main_harbor.py                              +13 lines   override_entrypoint peek
~ examples/train_integrations/harbor/README.md    +12 lines   cross-link + one-liner
~ integrations/arctic_rl/README.md                +20 lines   Harbor recipes section
```

No changes to `pyproject.toml`, no changes to any core `skyrl/` module, no
changes to `arctic-platform` or `harbor`. All arctic-side code lives under
`integrations/arctic_rl/`.

## Architecture

```
                  Single Ray cluster (no HTTP for Arctic; HTTP only inside sandbox)
+---------------------------------------------------------------------------------------+
|                                                                                       |
|  SkyRL driver (num_gpus=0)                                                            |
|  ArcticHarborExp  ------------> HarborGenerator ---------> Harbor Trial               |
|                                                          (Daytona / Modal sandbox)    |
|                                                              |                        |
|                                                              | POST /v1/chat/comp.    |
|                                                              v                        |
|                                            +----------------------------------+       |
|                                            |  Arctic OpenAI shim              |       |
|                                            |  (SkyRL's FastAPI endpoint,      |       |
|                                            |   fed by ArcticInferenceEngine   |       |
|                                            |   Adapter -> arctic_client)      |       |
|                                            +----------------------------------+       |
|                                                              |                        |
|                                                              | ArcticRLClient.        |
|                                                              | generate(prompts, sp)  |
|                                                              v                        |
|  Arctic RL Ray actors  --------------------------->  InferenceWorker  <=== NCCL/IPC   |
|    - DeepSpeedWorker (x N training GPUs)             (ArcticInference vLLM)   =====   |
|    - InferenceWorker (x M sampling GPUs)                                              |
|    - weight sync (NCCL or CUDA-IPC in colocated mode)                                 |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

The two new modules are the only new components; everything else is the
existing Arctic RL entrypoint (unchanged) and the existing Harbor
generator (unchanged).

## New modules

### `openai_bridge.py` — `ArcticInferenceEngineAdapter`

Duck-types the surface that SkyRL's existing FastAPI OpenAI shim
(`inference_engine_client_http_endpoint.serve`) reads from an
`InferenceEngineClient`. Exposes:

- `.model_name`, `.backend = "vllm"`
- `.enable_http_endpoint`, `.http_endpoint_host`, `.http_endpoint_port`
  (forwarded from `generator.inference_engine.*`)
- `async chat_completion(payload)` — translates the OpenAI request into
  `arctic_client.generate(prompts, sampling_params)`, applies the chat
  template (matches `HarborGenerator`'s lookup), maps vLLM finish reasons
  to OpenAI ones, wraps the response in OpenAI schema.
- `async completion(payload)` — same, for the (rarely used) `/v1/completions`
  path; decodes token-id prompts before handing to Arctic since Arctic's
  wire only speaks text.
- `spin_up_http_endpoint()` — runs SkyRL's shim in a daemon thread and
  waits for readiness.
- `__getstate__` drops the uvicorn thread so Ray can serialize the object
  onto workers.

Two subtle points that show up as inline comments in the file and are
worth understanding for review:

1. **`_sanitize_assistant_text`** — Qwen3's BPE tokenizer fuses three
   consecutive `\n` into a single token (`1406`). If the raw completion
   starts with `\n\n`, then concatenated with the chat template's
   trailing `\n` after `<|im_start|>assistant` the assistant message
   tokenizes with the fused token instead of `[198, ...]`. That trips
   SkyRL's `get_response_ids_and_loss_mask_from_messages` assertion at
   training time. `.lstrip()` on the completion suppresses that path.

2. **`reasoning_content: None`** on every choice.message dict. LiteLLM's
   `_parse_content_for_reasoning` will otherwise strip a
   `<think>...</think>` prefix out of `content` and leave a bare
   `\n\n{JSON}`, which re-triggers the same three-newline fusion above.
   Setting `reasoning_content` short-circuits LiteLLM's parser.

Both are documented in-place in the file.

### `harbor_entrypoint.py` — `ArcticHarborExp` + `main()`

- **`ArcticHarborSkyRLConfig(SkyRLTrainConfig)`** — adds Harbor's
  `harbor_trial_config: dict` + `generator: HarborGeneratorConfig`
  (already defined by Harbor), and overrides `trainer` to
  `ArcticTrainerConfig` so `trainer.arctic_rl.*` is available. **No
  `from __future__ import annotations`** — SkyRL's
  `build_nested_dataclass` introspects `dataclasses.fields(cls)[i].type`
  and expects a concrete class, not a string.
- **`ArcticHarborExp(ArcticRLExp)`** — reuses `ArcticRLExp.__init__`
  (which pre-inits the Arctic client) verbatim. Only overrides:
  - `get_train_dataset` / `get_eval_dataset` → `HarborTaskDataset`.
  - `get_generator` → `HarborGenerator`, passing the OpenAI adapter as
    the `inference_engine_client` (so `HarborGenerator` can hit the shim
    URL via LiteLLM without knowing Arctic exists).
  - `_setup_trainer` → spins up the OpenAI HTTP shim **before**
    constructing `HarborGenerator` (Harbor bakes the shim URL into
    LiteLLM's `api_base` at construction time), then hands the same
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
`skyrl.train.entrypoints.main_base` already uses (added there in commit
`2587f0e`). Direct copy of the pattern; nothing Harbor-specific.

```python
def main() -> None:
    for arg in sys.argv[1:]:
        if arg.startswith("trainer.override_entrypoint="):
            override = arg.split("=", 1)[1]
            from importlib import import_module
            return import_module(override).main()
    ...
```

This is the entire user-facing entry point: any existing Harbor launcher
becomes an Arctic launcher by appending
`trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint` to
its CLI.

### `pyproject.toml`

**No change.** The stale `arctic-rl` extra that existed on this branch's
fork was already dropped in upstream's own arctic_rl refactor (PR #1859).
Nothing in this PR touches `pyproject.toml`.

### README updates

- `integrations/arctic_rl/README.md`: new "Harbor recipes" section pointing
  at the reference launcher + this design doc; updated file-layout block.
- `examples/train_integrations/harbor/README.md`: new "Arctic RL backend"
  subsection cross-linking the arctic_rl README.

## Reference launcher

`integrations/arctic_rl/examples/run_codecontest_arctic_harbor.sh`
reproduces the FSDP2 Harbor CodeContests recipe with the Arctic backend.
Structure mirrors `run_gsm8k_grpo_4gpu.sh` and `run_bird_grpo_*.sh`:

- `uv run --isolated --extra skyrl-train --extra harbor --with arctic-platform --with 'arctic-inference[vllm]' --with liger-kernel --with 'transformers==4.57.6' --with "flash-attn@<torch-2.10 wheel>"` — no `uv sync` required (see `integrations/arctic_rl/README.md` for why).
- Every knob is a shell variable with a safe default, so a user's own
  launcher can `source` this one and override.
- `MAX_CONCURRENCY` default is a workable value for the Daytona free tier
  (10 concurrent CPUs → ≤ 8 concurrent trials). Bump when on a paid tier.

## Results — Qwen3-8B, 8×H100, CodeContests

Apples-to-apples baseline vs Arctic (same prompts via `trainer.seed=42`,
same batch shape, same rate limit). 8 steps each; single node.

| metric                              | FSDP2 baseline | Arctic RL     | speedup    |
|-------------------------------------|----------------|---------------|------------|
| step time — mean                    | 212.3 s        | **168.7 s**   | 1.26×      |
| step time — median                  | 220.0 s        | **170.7 s**   | 1.29×      |
| generate — mean                     | 198.4 s        | **156.9 s**   | 1.26×      |
| generate — median                   | 205.0 s        | **159.7 s**   | 1.28×      |
| sync_weights — mean                 | 6.3 s          | 7.2 s         | ≈ tie      |
| pass@4 — mean over 8 steps          | 0.375          | 0.500         | overlap    |
| avg_raw_reward — mean over 8 steps  | 0.156          | 0.250         | overlap    |
| Daytona sandbox failures            | 0              | 0             |            |

Workload characteristics (over ~400 CodeContests trials, Qwen3-8B tokenizer):

- Multi-turn: `terminus-2` agent, `max_turns=32`; observed 1–14 assistant
  turns per trial, median 2, mean 2.4.
- Prompt tokens per vLLM call: median 7,230; p90 14,325; tail saturates
  the 16 K context cap.
- Completion tokens per vLLM call: median 1,095; p90 13,120 (long
  `<think>` blocks).

Speedup is dominated by the `generate` phase — consistent with the
Arctic-Inference (FCA + fused kernels) + colocated ZeRO-3 DeepSpeed with
CUDA-IPC weight sync stack.

## Risks and known limits

- **`use_zorro=false` in the reference launcher.** ZoRRo's shape
  assertions on the prompt-group dedup path aren't verified through the
  OpenAI shim yet (Harbor produces heterogeneous, agentic prompts that
  don't share the "same problem × N samples" prefix pattern ZoRRo
  optimizes for). Safe default is off; users can flip to `true` when the
  prompt distribution is amenable.
- **Sandbox-provider concurrency.** The Daytona free tier caps concurrent
  sandboxes at 10 CPUs. The launcher defaults keep
  `train_batch_size × n_samples_per_prompt ≤ 32` and
  `MAX_CONCURRENCY ≤ 64` so users on paid tiers can dial up. On the free
  tier, `run_codecontest_arctic_harbor.sh` runs but back-pressures on
  sandbox creation; see the sandbox tier note in the README.
- **Chat template lookup.** The adapter re-applies the chat template
  server-side so training-time re-tokenization sees the same prompt bytes
  as sampling. The template path is read from
  `generator.inference_engine.engine_init_kwargs.chat_template`; both
  Harbor and Arctic must be pointed at the same file.

## Alternatives considered

- **In-process transport (no HTTP shim).** Rejected because Harbor's
  agent runs inside a sandbox VM, not in the driver's process — it needs
  a real network endpoint. Reusing SkyRL's existing OpenAI shim
  (`inference_engine_client_http_endpoint`) plus a duck-typed adapter
  costs zero new HTTP code.
- **Fork `HarborGenerator` to call `arctic_client` directly.** Rejected
  because it would fork Harbor's `terminus-2` LiteLLM path — non-trivial
  to keep in sync with upstream Harbor changes. The adapter is an
  isolated ~350-line module and preserves Harbor's LiteLLM/tool-call
  code path verbatim.
- **New extra in `pyproject.toml`.** Rejected: `uv` still evaluates the
  full `[project.optional-dependencies]` graph under `--isolated`, so an
  `arctic-inference[vllm]==0.18` pin would block any launcher that also
  needs harbor / megatron / fsdp (vLLM 0.19+). The launcher's
  `uv run --isolated --with ...` pattern (already used by every other
  Arctic launcher) sidesteps this without any pyproject churn.
