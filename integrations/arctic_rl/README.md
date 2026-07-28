# Arctic RL Integration for SkyRL

Routes SkyRL's GRPO loop through the [Arctic RL](https://github.com/Snowflake-AI-Research/Arctic-Platform) server. Adds the ZoRRo / Forest Cascade Attention / Arctic speculative-decoding stack ([ZoRRo blog](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/)) under SkyRL's existing trainer / Hydra / Ray plumbing.

## Install

Clone SkyRL:

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL
```

There is no `uv sync` step. Each launcher in `integrations/arctic_rl/examples/` invokes:

```
uv run --isolated --extra skyrl-train \
    --with arctic-platform \
    --with 'arctic-inference[vllm]' \
    --with liger-kernel \
    --with 'transformers==4.57.6' \
    --with "flash-attn@<torch-2.10 cu128 wheel URL>" \
    --with "flash-attn-3@<cu128 wheel URL>" \
    -- python -m ...
```

What each flag does:

- `--isolated` — resolves a fresh env from scratch instead of using SkyRL's lockfile. Required because `tool.uv.sources` in `pyproject.toml` pins vLLM to the cu129 index (vLLM 0.23 only), but `arctic-inference[vllm]` needs vLLM 0.18.
- `--extra skyrl-train` — pulls SkyRL's training deps (`ray`, `deepspeed`, `hydra-core`, …).
- `--with arctic-platform`, `--with 'arctic-inference[vllm]'`, `--with liger-kernel` — the three Arctic packages.
- `--with 'transformers==4.57.6'` — vLLM 0.18 needs transformers `<5`; pinned exact (not `<5`) because Ray's worker-spawn shell parses `<5` as a redirect from fd 5.
- `--with "flash-attn@<URL>"` — overrides the torch-2.11 flash-attn wheel that `tool.uv.sources` selects, replacing it with a torch-2.10 ABI wheel that matches vLLM 0.18's torch pin.
- `--with "flash-attn-3@<URL>"` — FA3 wheel from PyTorch's cu128 index (`cp39-abi3`). Default attention backend on the BIRD recipes; this is what produced the 2.38× number on H200.

Ray workers replay the same `uv run --isolated --with ...` via the [uv+Ray `py_executable`](https://www.anyscale.com/blog/uv-ray-pain-free-python-dependencies-in-clusters) integration, so the Arctic stack lands on every worker without a separate install.

First launch resolves and downloads the wheels (~5 min). Subsequent launches hit the uv cache.

## Smoke test — GSM8K, 4 GPUs

```bash
bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh
```

Downloads `Qwen/Qwen3-0.6B`, preps GSM8K parquets, starts a local Ray cluster. First reward in ~3 min from a cold uv cache.

Recent run (public mains, 4 GPUs):

```
step 0  avg_final_rewards: 0.229
step 1  avg_final_rewards: 0.247
step 2  avg_final_rewards: 0.280
step 3  avg_final_rewards: 0.337    loss=0.104  grad_norm=0.22
sync_weights steady state: <0.1s
```

## Use Arctic RL in a recipe

Add one flag to the SkyRL `main_base` invocation:

```
trainer.override_entrypoint=integrations.arctic_rl.entrypoint
```

This dispatches into `integrations/arctic_rl/entrypoint.py` after Hydra parsing. ZoRRo, FCA, Liger, and the multi-replica FlashInfer workaround are on by default; opt out via the `trainer.arctic_rl.*` knobs below.

Wrapped in the launcher driver:

```bash
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN3_WHL="https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"

uv run --isolated --extra skyrl-train \
    --with arctic-platform --with 'arctic-inference[vllm]' --with liger-kernel \
    --with 'transformers==4.57.6' \
    --with "flash-attn@${FLASH_ATTN_WHL}" \
    --with "flash-attn-3@${FLASH_ATTN3_WHL}" \
    -- python -m skyrl.train.entrypoints.main_base \
        trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
        trainer.arctic_rl.attn_implementation=flash_attention_3 \
        <existing recipe overrides>
```

Add `trainer.arctic_rl.speculative_model=<hf-id-or-path>` to enable Arctic speculative decoding.

### `trainer.arctic_rl.*` knobs

| Knob | Default | Effect |
| --- | --- | --- |
| `use_arctic_inference` | `true` | Forest Cascade Attention in the rollout; auto-injects the multi-replica `fuse_allreduce_rms` workaround when `num_engines > 1`. |
| `use_zorro` | `true` | ZoRRo split-attention + prompt dedup on the trainer side. |
| `logits_optimization` | `"memory"` | Chunked logits compute on the server (ZoRRo only). |
| `use_liger` | `true` | Liger fused MLP/RMSNorm kernels. |
| `speculative_model` | `None` | HF id or local path of an Arctic draft head. Enables Arctic speculative decoding when set. |
| `num_speculative_tokens` | `3` | Draft tokens per target-model step. Only used when `speculative_model` is set. |
| `zero_stage` | `0` | DeepSpeed ZeRO stage. `2` for ~1B models, `3` for ≥7B. Required to be ≥1 for bf16 model + fp32 grad. |
| `offload_optimizer` | `false` | CPU offload optimizer state. Needs `zero_stage ≥ 2`. |
| `offload_param` | `false` | CPU offload ZeRO-3 parameter shards. |
| `colocate` | `false` | Share GPUs between training and sampling. |
| `attn_implementation` | `flash_attention_3` on BIRD launchers, `flash_attention_2` on GSM8K | Set to `flash_attention_2` on A100 / L40S. |
| `cuda_ipc_weight_sync` | `false` | Zero-copy IPC weight sync. Requires `colocate=true`. |
| `vllm_max_num_seqs` | `256` | vLLM batching cap. |
| `vllm_config` | `None` | Raw `vllm.AsyncEngineArgs` dict, forwarded verbatim. See below. |

### `trainer.arctic_rl.vllm_config`

`trainer.arctic_rl.*` covers the vLLM knobs Arctic RL needs. `vllm_config` is for raw vLLM fields the typed knobs don't model:

```bash
trainer.arctic_rl.vllm_config="{compilation_config: {cudagraph_mode: FULL}}"
```

The dict is shallow-merged after auto-injected defaults, so user keys win on conflict.

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

`comm_protocol="ray"` is pinned in the integration. arctic-platform's HTTP transport is for remote dss-platform deployments and is not used here.

## Multi-node

`ray start` on each node, same pattern as every other SkyRL recipe:

```bash
# Head node
uv run --isolated --extra skyrl-train ray start --head --port=6379 --num-gpus=8

# Each worker
uv run --isolated --extra skyrl-train ray start --address=<HEAD_IP>:6379 --num-gpus=8

uv run --isolated --extra skyrl-train ray status
```

GPU layout:

- Training: `trainer.placement.policy_num_gpus_per_node * policy_num_nodes`
- Sampling: `generator.inference_engine.num_engines * tensor_parallel_size`
- Log-prob: `trainer.arctic_rl.log_prob_gpus` (`0` colocates with sampling)

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

`examples/run_bird_grpo_32b_32gpu.sh` runs the 32B Text2SQL setup from the [ZoRRo blog](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/).

```bash
# 1. Clone
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL

# 2. Ray cluster (4-node example; repeat on each worker)
uv run --isolated --extra skyrl-train ray start --head --port=6379 --num-gpus=8
# worker:
#   uv run --isolated --extra skyrl-train ray start --address=<head>:6379 --num-gpus=8

# 3. BIRD raw data + preprocess
mkdir -p /data/bird && cd /data/bird
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip && unzip train.zip
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip   && unzip dev.zip
cd -
python integrations/arctic_rl/envs/preprocess_bird.py \
    --bird_dir   /data/bird \
    --output_dir $HOME/data/bird \
    --max_tokens 32768 \
    --tokenizer  Qwen/Qwen3-1.7B

# 4. Launch
DATA_DIR=$HOME/data/bird WANDB_API_KEY=<key> \
    bash integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh
```

Notes:

- BIRD raw data is gated behind <https://bird-bench.github.io>; the `wget` URLs above are the mirrors arctic-platform's txt2sql README uses.
- `--max_tokens 32768` matches the recipe's `trainer.max_prompt_length=32768`.
- `preprocess_bird.py` emits verl-format parquets with absolute `extra_info.db_path`. The SQLite files at those paths must be readable on every Ray node — shared filesystem, or mirror `/data/bird/` to each worker.
- `run_bird_grpo_8b_32gpu.sh` ships an 8B variant on the same cluster shape for faster iteration.
