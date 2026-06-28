# Arctic RL Integration for SkyRL

**Bring [ZoRRo](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/)'s 3.5× RL speedup to any SkyRL recipe with a single CLI flag.**

[ZoRRo (Zero Redundancy Rollouts)](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/) is Snowflake AI Research's RL acceleration stack — prompt-deduplicated split attention for training, Forest Cascade Attention for inference, and Arctic speculative decoding for generation. On Arctic-Text2SQL-R2 it delivers **3.5× faster end-to-end training**, **6× faster actor update**, **5× faster log-prob**, **1.7× faster rollout generation**, and **3.2× longer context**.

This integration routes SkyRL's GRPO loop through the open-source [Arctic RL](https://github.com/Snowflake-AI-Research/Arctic-Platform) server — the same backend that produced those numbers — so you get all of ZoRRo's optimizations on your existing recipes, untouched.

## Install

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL
```

That's it. No `uv sync --extra`, no `uv pip install`. Every launcher in `examples/` uses `uv run --isolated --extra skyrl-train --with arctic-platform --with 'arctic-inference[vllm]' --with liger-kernel --with 'transformers==4.57.6' --with flash-attn@<wheel>` — the same pattern as `examples/train/flash_rl` and `examples/train_integrations/harbor`. uv resolves the env on first launch (~5 min, cached after) and SkyRL's uv+Ray plugin replicates the exact invocation on every worker.

Why the `--with` overrides: `arctic-inference[vllm]` pulls vLLM 0.18, which requires `transformers<5` and a torch-2.10 ABI flash-attn wheel. SkyRL's `pyproject.toml` pins `transformers>=5.6.1` and ships a torch-2.11 flash-attn (both needed by the fsdp/megatron backends). `--isolated` + `--with` overrides resolve to vLLM 0.18's stack for this run only, leaving the project `.venv` untouched.

## 30-second smoke test (GSM8K, 4 GPUs)

```bash
bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh
```

(Single-node — the launcher's `python -m skyrl.train.entrypoints.main_base` auto-starts a local Ray cluster.)

Auto-downloads `Qwen/Qwen3-0.6B`, auto-preps GSM8K parquets. First reward signal lands in ~3 min from a cold start. On a clean public-mains run we see:

```
step 0  avg_final_rewards: 0.229
step 1  avg_final_rewards: 0.247
step 2  avg_final_rewards: 0.280
step 3  avg_final_rewards: 0.337    loss=0.104  grad_norm=0.22
step 1 wall clock: 56s  (generate + train + sync_weights)
sync_weights steady state: <0.1s
```

## Drop into your own recipe

Any stock SkyRL recipe becomes an Arctic RL recipe by appending **one flag** and running through the same `uv run --isolated` driver the launchers use:

```bash
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

uv run --isolated --extra skyrl-train \
    --with arctic-platform \
    --with 'arctic-inference[vllm]' \
    --with liger-kernel \
    --with 'transformers==4.57.6' \
    --with "flash-attn@${FLASH_ATTN_WHL}" \
    -- python -m skyrl.train.entrypoints.main_base \
        trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
        <... your existing recipe overrides ...>
```

That's it. ZoRRo (training-side dedup + memory-chunked logits), Forest Cascade Attention (rollout), Liger fused kernels, and the multi-replica FlashInfer workaround are all **on by default**. Add `trainer.arctic_rl.speculative_model=<hf-id-or-path>` to also turn on Arctic speculative decoding.

`trainer.override_entrypoint` tells `main_base` to dispatch into the Arctic entrypoint after Hydra parsing. No `PYTHONPATH` setup, no edits to `main_base.py`.

### `trainer.arctic_rl.*` knobs

Defaults assume you came here to use Arctic RL; opt **out** of optimizations explicitly if you need a baseline comparison.

| Knob | Default | Purpose |
| --- | --- | --- |
| `use_arctic_inference` | **`true`** | Forest Cascade Attention in the rollout (auto-injects the multi-replica `fuse_allreduce_rms` workaround when needed). |
| `use_zorro` | **`true`** | ZoRRo split-attention + prompt dedup in the trainer. |
| `logits_optimization` | **`"memory"`** | Chunked logits compute on the server (ZoRRo only). |
| `use_liger` | **`true`** | Liger fused MLP/RMSNorm kernels. |
| `speculative_model` | `None` | HF id / local path of an Arctic draft head — set this to enable Arctic speculative decoding. |
| `num_speculative_tokens` | `3` | Draft tokens per target-model step (only when `speculative_model` is set). |
| `zero_stage` | `0` | DeepSpeed ZeRO (use `2` for ~1B, `3` for ≥7B; required for bf16 + fp32 grad). |
| `offload_optimizer` | `false` | CPU offload optimizer state (`zero_stage≥2`). |
| `offload_param` | `false` | CPU offload ZeRO-3 param shards. |
| `colocate` | `false` | Share GPUs between training and sampling. |
| `attn_implementation` | `flash_attention_2` | `flash_attention_2` or `flash_attention_3` (Hopper, optional wheel). |
| `cuda_ipc_weight_sync` | `false` | Zero-copy IPC weight sync (colocate-only). |
| `vllm_max_num_seqs` | `256` | vLLM batching cap. |
| `vllm_config` | `None` | Escape hatch for raw `vllm.AsyncEngineArgs` keys not covered above (see below). |

### Escape hatch — `trainer.arctic_rl.vllm_config`

The integration translates `arctic_rl.*` knobs into the corresponding `vllm.AsyncEngineArgs` keys for you — FCA, the multi-replica FlashInfer workaround, and Arctic speculative decoding are all auto-injected when their typed flag is on.

`vllm_config` is the escape hatch for vLLM knobs the typed fields don't yet cover — e.g. a non-default `cudagraph_mode`, a custom `compilation_config` pass, or any future vLLM field arctic-inference doesn't model. The dict is forwarded verbatim to `vllm.AsyncEngineArgs` via arctic-platform's `ArcticRLClientConfig.vllm_config` (public main, unmodified), and **user keys win on conflict** with the auto-injected defaults:

```bash
trainer.arctic_rl.vllm_config="{compilation_config: {cudagraph_mode: FULL}}"
```

You should not need this in the happy path. If you reach for it, the typed-knob layer probably should grow a new field — file a follow-up.

## Architecture

```
                 Single Ray cluster (no HTTP, no subprocess)
 +----------------------------------------------------------------+
 |  SkyRL driver (num_gpus=0)        Arctic RL Ray actors         |
 |  - Data loading                   - ArcticRLRayServerState     |
 |  - skyrl-gym reward scoring       - DeepSpeedWorker (xN)       |
 |  - generate -> score -> train     - InferenceWorker (xM)       |
 |                                     = ArcticInference vLLM     |
 |         |                                  ^                   |
 |         +---- ray.get(actor.x.remote()) ---+                   |
 |                                                                |
 |  NCCL weight sync runs GPU-to-GPU between DeepSpeedWorker and  |
 |  InferenceWorker (driver never sees weights).                  |
 +----------------------------------------------------------------+
```

The integration pins `comm_protocol="ray"` — the SkyRL driver and Arctic server actors share one Ray cluster. arctic-platform's HTTP transport is for remote dss-platform deployments and isn't used here.

## Multi-node and FA3

Same `ray start` pattern as every other SkyRL recipe — `uv run --isolated --extra skyrl-train ray start ...` on each node:

```bash
# Head node:
uv run --isolated --extra skyrl-train ray start --head --port=6379 --num-gpus=8
# Each worker (same SkyRL checkout):
uv run --isolated --extra skyrl-train ray start --address=<HEAD_IP>:6379 --num-gpus=8
uv run --isolated --extra skyrl-train ray status   # confirms total GPUs
```

The launcher's `uv run --isolated --extra skyrl-train --with arctic-platform ...` invocation is replicated on every Ray worker via the uv+Ray `py_executable` integration, so workers automatically get the same arctic stack.

GPU layout follows standard SkyRL knobs:
- Training: `trainer.placement.policy_num_gpus_per_node * policy_num_nodes`
- Sampling: `generator.inference_engine.num_engines * tensor_parallel_size`
- Log-prob: `trainer.arctic_rl.log_prob_gpus` (0 = colocate with sampling)

Optional FA3 on Hopper (recommended for the largest speedups):

```bash
uv pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128
```


## File layout

```
integrations/arctic_rl/
├── README.md
├── trainer.py         ArcticPPOTrainer — routes training to Arctic server actors
├── generator.py       ArcticGenerator — routes generation to vLLM, scores via skyrl-gym
├── config.py          ArcticRLTrainerConfig + build_rl_config
├── entrypoint.py      Dispatched from main_base via trainer.override_entrypoint
├── envs/
│   ├── bird.py
│   ├── bird_reward.py
│   └── preprocess_bird.py
└── examples/
    ├── run_gsm8k_grpo_4gpu.sh
    └── run_bird_grpo_*.sh      (BIRD recipes — manual data prep today)
```

## BIRD-SQL: 32B Qwen3 on 32 × H200

`examples/run_bird_grpo_32b_32gpu.sh` reproduces the 32B Text2SQL setup from the [ZoRRo blog](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/). End-to-end from a clean machine:

```bash
# 1. Clone (first launcher invocation resolves the env in ~5 min, cached after)
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL

# 2. Ray up (4-node example; repeat on each worker)
uv run --isolated --extra skyrl-train ray start --head --port=6379 --num-gpus=8
# on each worker:
#   uv run --isolated --extra skyrl-train ray start --address=<head>:6379 --num-gpus=8

# 3. (Hopper) FA3 wheel
uv pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128

# 4. BIRD data — raw download + preprocess (a one-command HF bundle is on the roadmap)
mkdir -p /data/bird && cd /data/bird
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip && unzip train.zip
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip   && unzip dev.zip
cd -
python integrations/arctic_rl/envs/preprocess_bird.py \
    --bird_dir   /data/bird \
    --output_dir $HOME/data/bird \
    --max_tokens 32768 \
    --tokenizer  Qwen/Qwen3-1.7B

# 5. Run
DATA_DIR=$HOME/data/bird WANDB_API_KEY=<key> \
    bash integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh
```

Notes:

- BIRD raw data is gated behind <https://bird-bench.github.io>; the `wget` URLs above are the public mirrors arctic-platform's txt2sql README uses.
- `--max_tokens 32768` matches the recipe's `trainer.max_prompt_length=32768`. Drop this lower only if you also lower the recipe's prompt length.
- `preprocess_bird.py` writes verl-format parquets whose `extra_info.db_path` is absolute. The SQLite files at those paths must be readable on every node at training time — either a shared filesystem, or mirror `/data/bird/` to each worker.
- An 8B variant `run_bird_grpo_8b_32gpu.sh` ships for quick smoke tests on the same cluster.
