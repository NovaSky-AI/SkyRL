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

### Raw vLLM overrides — `trainer.arctic_rl.vllm_config`

For anything the high-level schema doesn't model — `compilation_config`, `speculative_config`, Forest Cascade Attention configs — pass a dict that's forwarded verbatim to vLLM:

```bash
trainer.arctic_rl.vllm_config="{
    compilation_config: {cudagraph_mode: PIECEWISE,
                         pass_config: {fuse_allreduce_rms: false}},
    speculative_config: {method: arctic, model: <hf-id>, num_speculative_tokens: 3},
    forest_cascade_attn_configs: '{}',
}"
```

This routes through `arctic-platform`'s `ArcticRLClientConfig.vllm_config` (public main, unmodified) — typed keys land on `ModelConfig`, unknown keys land in `extra_engine_kwargs`. No patches to arctic-platform or arctic-inference required.

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

## BIRD-SQL recipes

`examples/run_bird_grpo_8b_32gpu.sh` and `run_bird_grpo_32b_32gpu.sh` reproduce the 32B Arctic-Text2SQL-R2 setup. They currently require manual BIRD raw-data download + `python integrations/arctic_rl/envs/preprocess_bird.py`; a single-command HuggingFace-hosted parquet bundle is on the roadmap. See the launcher source for the current path.
