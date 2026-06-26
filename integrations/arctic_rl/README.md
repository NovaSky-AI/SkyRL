# Arctic RL Integration for SkyRL

**Bring [ZoRRo](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/)'s 3.5× RL speedup to any SkyRL recipe with a single CLI flag.**

[ZoRRo (Zero Redundancy Rollouts)](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/) is Snowflake AI Research's RL acceleration stack — prompt-deduplicated split attention for training, Forest Cascade Attention for inference, and Arctic speculative decoding for generation. On Arctic-Text2SQL-R2 (Qwen3-32B, 32 × H200), it delivers **3.5× faster end-to-end training**, **6× faster actor update**, **5× faster log-prob**, **1.7× faster rollout generation**, and **3.2× longer context**.

This integration routes SkyRL's GRPO loop through the open-source [Arctic RL](https://github.com/Snowflake-AI-Research/Arctic-Platform) server — the same backend that produced those numbers — so you get all of ZoRRo's optimizations on your existing recipes, untouched.

## 30-second smoke test (GSM8K, 4 GPUs)

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL
uv sync --extra arctic-rl
uv pip install --upgrade 'transformers>=4.57,<5'   # vllm 0.18 pin
uv run ray start --head --num-gpus=4
bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh
```

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

Any stock SkyRL recipe becomes an Arctic RL recipe by appending one flag:

```bash
uv run -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.use_arctic_inference=true \
    trainer.arctic_rl.zero_stage=2 \
    <... your existing recipe overrides ...>
```

`trainer.override_entrypoint` tells `main_base` to dispatch into the Arctic entrypoint after Hydra parsing. No `PYTHONPATH` setup, no edits to `main_base.py`.

### `trainer.arctic_rl.*` knobs

| Knob | Default | Purpose |
| --- | --- | --- |
| `use_zorro` | `false` | Enable ZoRRo split-attention + prompt dedup in the trainer. |
| `use_arctic_inference` | `false` | Enable Forest Cascade Attention + Arctic spec-dec in the rollout. |
| `use_liger` | `false` | Liger fused kernels in the policy fwd/bwd. |
| `zero_stage` | `0` | DeepSpeed ZeRO (use `2` for ~1B, `3` for ≥7B; required for bf16 + fp32 grad). |
| `offload_optimizer` | `false` | CPU offload optimizer state (`zero_stage≥2`). |
| `offload_param` | `false` | CPU offload ZeRO-3 param shards. |
| `colocate` | `false` | Share GPUs between training and sampling. |
| `attn_implementation` | `flash_attention_2` | `flash_attention_2` or `flash_attention_3` (Hopper, optional wheel). |
| `cuda_ipc_weight_sync` | `false` | Zero-copy IPC weight sync (colocate-only). |
| `vllm_max_num_seqs` | `256` | vLLM batching cap. |
| `vllm_config` | `None` | Raw `vllm.AsyncEngineArgs` overrides (see below). |

### Escape hatch — `trainer.arctic_rl.vllm_config`

The simple case (`use_zorro=true use_arctic_inference=true`) is enough for most users — cudagraph mode, attention pass selection, KV-cache settings, weight-sync wiring are all handled inside arctic-platform + arctic-inference. You shouldn't need to think about vLLM flags.

For knobs that aren't yet exposed as typed `trainer.arctic_rl.*` fields — currently Forest Cascade Attention configs, Arctic speculative decoding, and one multi-replica fused-allreduce workaround — `vllm_config` is a dict forwarded verbatim to `vllm.AsyncEngineArgs` via `arctic-platform`'s `ArcticRLClientConfig.vllm_config` (public main, unmodified):

```bash
trainer.arctic_rl.vllm_config="{
    forest_cascade_attn_configs: '{}',
    speculative_config: {method: arctic, model: <hf-id>, num_speculative_tokens: 3},
    compilation_config: {pass_config: {fuse_allreduce_rms: false}},
}"
```

Why each of those three appears here today, and why we expect them to disappear:

- **`forest_cascade_attn_configs`** — should be implied by `use_arctic_inference=true`. Will be folded into the high-level knob in a follow-up.
- **`speculative_config`** — should be a typed `trainer.arctic_rl.speculative_model` field. arctic-platform's `arctic_inference_config` schema already has slots for this; we'll wire SkyRL to it once the public-main schema reconciles.
- **`compilation_config.pass_config.fuse_allreduce_rms: false`** — arctic-inference's `use_fca=True` default is `true`, which collides with FlashInfer workspace allocation on multi-replica/multi-node setups (per-process IPC port collision). The right fix is inside arctic-inference (auto-disable when `world_size > tp_size`); this `vllm_config` override is the interim workaround on multi-node runs.

In short: `vllm_config` exists for the long tail; it shouldn't be in the critical path of getting the speedup.

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

```bash
# Head node:
uv run ray start --head --port=6379 --num-gpus=8
# Each worker (same SkyRL checkout, same env):
uv run ray start --address=<HEAD_IP>:6379 --num-gpus=8
uv run ray status   # confirms total GPUs
```

GPU layout follows standard SkyRL knobs:
- Training: `trainer.placement.policy_num_gpus_per_node * policy_num_nodes`
- Sampling: `generator.inference_engine.num_engines * tensor_parallel_size`
- Log-prob: `trainer.arctic_rl.log_prob_gpus` (0 = colocate with sampling)

Optional FA3 on Hopper (recommended for the largest speedups):

```bash
uv pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128
```

Skip and set `trainer.arctic_rl.attn_implementation=flash_attention_2` if you don't want FA3.

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

`examples/run_bird_grpo_32b_32gpu.sh` reproduces the 32B Arctic-Text2SQL-R2 setup from the [ZoRRo blog](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/). End-to-end from a clean machine:

```bash
# 1. Clone + install
git clone https://github.com/Snowflake-AI-Research/SkyRL.git && cd SkyRL
uv sync --extra arctic-rl
uv pip install --upgrade 'transformers>=4.57,<5'   # vllm 0.18 pin

# 2. Ray up (4-node example; repeat on each worker)
uv run ray start --head --port=6379 --num-gpus=8
# on each worker:
#   uv run ray start --address=<head>:6379 --num-gpus=8

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
