# Arctic RL Integration for SkyRL

Routes SkyRL's GRPO training loop through [Arctic RL](https://github.com/Snowflake-AI-Research/Arctic-Platform) — **all GPU operations** (training, generation, log-probs, weight sync) run on Arctic's Ray actors. The SkyRL driver process holds 0 GPUs and orchestrates via Ray actor calls.

```
                       Single Ray cluster
 +----------------------------------------------------------------+
 |                                                                |
 |   SkyRL driver (num_gpus=0)        Arctic RL Ray actors        |
 |   - Data loading                   - ArcticRLRayServerState    |
 |   - skyrl-gym reward scoring       - DeepSpeedWorker (xN)      |
 |   - generate -> score -> train     - InferenceWorker (xM)      |
 |                                      = ArcticInference vLLM    |
 |          |                                  ^                  |
 |          +---- ray.get(actor.x.remote()) ---+                  |
 |                                                                |
 |   NCCL weight sync runs directly between DeepSpeedWorker       |
 |   and InferenceWorker GPUs (no driver involvement).            |
 +----------------------------------------------------------------+
```

`arctic-platform` supports both Ray and HTTP transports (`comm_protocol="ray"` vs `"http"`); the HTTP path is for remote dss-platform deployments. This integration pins `comm_protocol="ray"` so the SkyRL driver and the Arctic server actors live in the same Ray cluster and talk via in-cluster actor calls — no HTTP server, no separate process group.

## Quick Start (GSM8K, 4 GPUs, single node)

The fastest way to verify a clean install: GSM8K trains end-to-end on 4 GPUs with no manual data prep. Auto-downloads `Qwen/Qwen3-0.6B` from HuggingFace and writes GSM8K parquets on first launch.

### 1. Clone and install

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL
uv sync --extra arctic-rl
uv pip install --upgrade 'transformers>=4.57,<5'
```

The `arctic-rl` extra pulls `arctic-platform`, `arctic-inference[vllm]`, and `liger-kernel` from their public `main` branches; vLLM and torch arrive transitively via `arctic-inference[vllm]`. The `transformers` line is a one-time pin: SkyRL's base extra requires `>=5.0`, but vLLM 0.18 (what arctic-inference targets) requires `<5`.

### 2. Start Ray

```bash
uv run ray start --head --num-gpus=4
uv run ray status   # should show 4/4 GPUs
```

### 3. Run

```bash
bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh
```

That's it. The recipe auto-preps GSM8K parquets to `$HOME/data/gsm8k/` and starts training. Pass `LOGGER=wandb WANDB_API_KEY=...` to log to W&B; the default `LOGGER=console` runs with no credentials.

#### What to expect

On 4 H200 GPUs (1 inference engine, 2 training, 1 log-prob), at default batch sizes (`train_batch_size=256`, `n_samples_per_prompt=4`):

| Phase | Wall clock |
| --- | --- |
| Cold eval (6 iters) | ~2 min |
| Step 1 (cold) | ~56s (generate + train + sync) |
| `sync_weights` (steady state) | <0.1s |
| `train_critic_and_policy` | ~12s |

Reward signal at the first four steps from a clean run on locked public-main `arctic-platform` + `arctic-inference`:

```
step 0  avg_final_rewards: 0.229
step 1  avg_final_rewards: 0.247
step 2  avg_final_rewards: 0.280
step 3  avg_final_rewards: 0.337   loss=0.104  grad_norm=0.22
```

## Enabling Arctic RL on Your Own Recipe

Any stock SkyRL recipe becomes an Arctic RL recipe by appending a single override:

```bash
uv run -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.zero_stage=2 \
    <... your existing recipe overrides ...>
```

`trainer.override_entrypoint` tells `skyrl.train.entrypoints.main_base` to dispatch into `integrations/arctic_rl/entrypoint.py:main()` after Hydra parsing. No changes to `main_base.py` required; no `PYTHONPATH` setup either — `integrations/` is a PEP-420 namespace package, reachable from the SkyRL repo root.

### Arctic-specific knobs

All under `trainer.arctic_rl.*`. Defaults are filled in by `integrations/arctic_rl/config.py` if unset.

| Knob | Default | Purpose |
| --- | --- | --- |
| `zero_stage` | `0` | DeepSpeed ZeRO stage. Use `2` for ~1B params, `3` for ≥7B. ZeRO ≥ 1 is required for bf16 + fp32 grad accumulation. |
| `offload_optimizer` | `false` | Offload optimizer state to CPU (needs `zero_stage>=2`). |
| `offload_param` | `false` | Offload ZeRO-3 param shards to CPU (needs `zero_stage>=3`). |
| `colocate` | `false` | Share GPUs between training and sampling. |
| `log_prob_gpus` | `0` | Dedicated log-prob GPUs (0 = colocate with sampling vLLM). |
| `use_zorro` | `false` | Enable Snowflake's ZoRRo trainer (chunked / fused GRPO loss). |
| `use_liger` | `false` | Enable Liger fused kernels in the policy fwd/bwd. |
| `attn_implementation` | `flash_attention_2` | `flash_attention_2` or `flash_attention_3` (Hopper, needs FA3 wheel). |
| `use_arctic_inference` | `false` | Use Arctic's vLLM wrapper (required for FCA / Arctic spec-dec). |
| `cuda_ipc_weight_sync` | `false` | Zero-copy IPC weight sync (colocate-only). |
| `vllm_enforce_eager` | `false` | Disable vLLM cudagraph compilation. |
| `vllm_max_num_seqs` | `256` | vLLM batching cap. |
| `vllm_config` | `None` | Raw `vllm.AsyncEngineArgs` overrides — see below. |

### Raw vLLM overrides via `vllm_config`

Performance knobs that the high-level schema doesn't model go through `trainer.arctic_rl.vllm_config`. The dict is merged on top of the integration's vLLM defaults and forwarded verbatim to `vllm.AsyncEngineArgs`:

```bash
trainer.arctic_rl.vllm_config="{
    compilation_config: {
        cudagraph_mode: PIECEWISE,
        pass_config: {fuse_allreduce_rms: false},
    },
    speculative_config: {
        method: arctic,
        model: <hf-id-or-local-path>,
        num_speculative_tokens: 3,
    },
    forest_cascade_attn_configs: '{}',
}"
```

This routes through `arctic-platform`'s public-main `ArcticRLClientConfig.vllm_config` field, which forwards every key into vLLM — typed fields land on `ModelConfig`, unknown keys land in `extra_engine_kwargs`. Use this for things like:

- `compilation_config` — pin `cudagraph_mode` or disable a specific compile pass (`fuse_allreduce_rms` collides with multi-replica/node FlashInfer; turn it off when you see `Flashinfer workspace must be initialized` asserts).
- `speculative_config` — point at an Arctic draft head for inference-time spec-decoding.
- `forest_cascade_attn_configs` — enable Arctic's FCA attention pass (`"{}"` uses defaults).

OmegaConf parses flow-style YAML, so leave a space after every `:` and quote string values that contain colons.

## Multi-node setup

The integration is identical for multi-node — only the Ray bringup changes.

```bash
# On the head node:
uv run ray start --head --port=6379 --num-gpus=8

# On each worker node (same SkyRL checkout, same env):
uv run ray start --address=<HEAD_IP>:6379 --num-gpus=8

# Confirm:
uv run ray status
```

GPU layout is derived from standard SkyRL knobs and `trainer.arctic_rl.*`:

- Training GPUs: `trainer.placement.policy_num_gpus_per_node * trainer.placement.policy_num_nodes`
- Sampling GPUs: `generator.inference_engine.num_engines * generator.inference_engine.tensor_parallel_size`
- Log-prob GPUs: `trainer.arctic_rl.log_prob_gpus` (0 = colocate with sampling)

When `trainer.arctic_rl.colocate=true`, training and sampling share the same GPU set; otherwise the two sets must be disjoint.

### (Optional) FlashAttention-3 on Hopper

The 8B / 32B recipes target `flash_attention_3` for best throughput. PyTorch publishes [official FA3 wheels](https://dev-discuss.pytorch.org/t/flash-attention-3-wheels/3322), ABI-stable for any Python ≥ 3.9, torch ≥ 2.9:

```bash
uv pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128
```

(`cu126` / `cu130` indices also available.) If you don't install FA3, set `trainer.arctic_rl.attn_implementation=flash_attention_2` on the launcher.

## BIRD-SQL recipes

`integrations/arctic_rl/examples/run_bird_grpo_*` ships 8B and 32B BIRD-SQL recipes. These currently require manual BIRD data prep (raw download + `integrations/arctic_rl/envs/preprocess_bird.py`); a one-command HuggingFace-hosted parquet bundle is on the roadmap. See the launcher source for the current path.

## File structure

```
integrations/arctic_rl/                # importable as integrations.arctic_rl
├── README.md
├── __init__.py                        # exports ArcticPPOTrainer, ArcticGenerator
├── trainer.py                         # ArcticPPOTrainer: routes training to server
├── generator.py                       # ArcticGenerator: routes generation to server vLLM
├── config.py                          # ArcticRLTrainerConfig + build_rl_config
├── entrypoint.py                      # dispatched here from main_base
├── envs/
│   ├── bird.py                        # skyrl-gym BIRD SQL env
│   ├── bird_reward.py                 # vendored verl PR #6 SQL reward fn
│   └── preprocess_bird.py             # raw BIRD -> verl parquets
└── examples/
    ├── run_gsm8k_grpo_4gpu.sh         # 4-GPU GSM8K smoke / reference recipe
    └── run_bird_grpo_*.sh             # BIRD-SQL recipes (require manual data prep)
```

## How it works

1. **`arctic_rl.entrypoint.main`** pre-initializes the Arctic RL Ray actors on the driver (one `ArcticRLRayServerState` actor that owns `DeepSpeedWorker` and `InferenceWorker` child actors), grabs their handles via `pre_client.get_server_state()`, then launches the SkyRL trainer as another Ray task in the same cluster, passing the actor handles in as `server_state`. The trainer reattaches via `create_arctic_rl_client(reconnect_cfg, server_state)` so it talks to the same actors the driver started.

2. **`ArcticPPOTrainer`** replaces three steps of SkyRL's training loop:
   - `fwd_logprobs_values_reward` → no-op (server actors compute old log-probs internally)
   - `compute_advantages_and_returns` → no-op (server actors compute GRPO advantages from rewards)
   - `train_critic_and_policy` → ships batches to the server actors via `ray.get(actor.train.remote(...))`; the actors run GRPO loss + backward + step

3. **`ArcticGenerator`** routes generation to the `InferenceWorker` (vLLM) actors and scores completions client-side via `skyrl-gym`.

4. **Server-side GRPO loss** is self-contained: per-token log-probs with causal shift, old log-probs by detaching (correct for `update_epochs_per_batch=1`), group-relative advantages from per-sequence rewards, PPO clipped surrogate.

The SkyRL ↔ Arctic RL protocol is minimal: server actors take `(sequences, rewards, loss_mask)` and return updated weights via NCCL sync directly to the `InferenceWorker` GPUs (the driver never sees the weights). All reward scoring and data orchestration live on the SkyRL side, all GPU compute lives on the Arctic actors.
