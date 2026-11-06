# DeepSeek-V4.1-Flash LoRA RL: long multi-turn agent workload

Fully async RL: vLLM sampling and Megatron training run at the same time, on separate GPUs. The rollouts look like a
long tool-use agent: 69 turns, contexts up to ~200k tokens. The run needs a Ray cluster with two 8-GPU nodes (B200
or B300): one trainer node and one vLLM node.

```bash
# once (CPU only): download the model and make it text-only in place, then build the 69-turn dataset.
# Put both on storage every node can read.
MODEL_PATH=/mnt/shared/DeepSeek-V4.1-Flash DATA_DIR=/mnt/shared/data/gsm8k_long_horizon_t69 \
  bash examples/train/deepseek_v4_flash/long_horizon/setup_dsv41_long_horizon.sh

# the run: ~50 min startup, ~1.5-2 h of 69-turn rollouts before the first step
MODEL_PATH=/mnt/shared/DeepSeek-V4.1-Flash DATA_DIR=/mnt/shared/data/gsm8k_long_horizon_t69 \
  bash examples/train/deepseek_v4_flash/long_horizon/run_dsv41_long_horizon.sh
```

Logs go to `$OUT/run.log` (default `~/skyrl-logs/dsv41_long_horizon`) and to the W&B project `dsv41_long_horizon`.
Use `LOGGER=console` to skip W&B.

## Configuration

| | |
|---|---|
| rollout | 69 turns. Each turn generates exactly 1,065 tokens (`ignore_eos`), then gets a ~1,870-token tool observation. |
| concurrency | 80 rollouts in flight per vLLM GPU (640 on 8). 128 rollouts (16 prompts x 8) per step, so ~5 steps of rollouts in flight (staleness 4). |
| trainer | TP1 PP1 CP8 EP8. One 200k-token sequence spans the node's 8 GPUs; CP4 runs out of memory on B200 and fits on B300. Packed sequences, stateful full recompute with layer inputs offloaded to pinned host memory, frozen base weights stored as MXFP8, token-balanced DP, fused LM-head logprob. |
| old logprobs | `trainer.old_logprobs_from_first_minibatch=true`. With one mini-batch per step, the old logprobs come from the training forward itself. This skips a separate forward pass that took ~25-35% of a 200k step. |
| sampler | vLLM DP8 + EP8 (data-parallel attention, expert-parallel MoE). Each GPU caches only its own sessions' KV: ~93M tokens per B200 node, vs ~10M with TP8. Serves the released FP8 / MXFP4 weights. |
| MoE + LoRA kernels | MXFP4 MoE + LoRA on FlashInfer's TRT-LLM kernel, two-kernel (shrink + expand) MoE-LoRA with tuned tiles (`vllm_tuned_configs/`), int64 offsets and global LoRA metadata under EP. CUDA graphs: `FULL_DECODE_ONLY`. |
| routing | Each rollout sticks to the DP rank that holds its prefix cache (`SKYRL_SESSION_STICKY_DIRECT=1`, consistent-hash router). Round-robin drops prefix-cache hits to ~5-22%. |
| weight sync | LoRA only, over NCCL, after every step. Under DP the engines pause in "wait" mode: in-flight turns finish first. "Keep" mode can deadlock the DP ranks during the update. |
| algorithm | LoRA r32 on attention, shared experts and routed experts. GRPO advantages, rollout-anchored masked importance sampling (MIS-style), routing replay (R3), score centering. |

Most of these are now defaults of `../run_async_multiturn_dsv41_flash_lora.sh`, the general multi-turn recipe.
`VLLM_DP=1` switches it back to TP8 serving.

## Measured (16x B200, defaults)

Each step trains 128 rollouts of ~200.7k tokens, ~25.7M tokens in total.

| | |
|---|---|
| decode, 20k-108k context | ~1,420 tok/s/GPU |
| decode, 120k-196k context | ~1,130-1,340 tok/s/GPU |
| trainer forward + backward | ~2,300-2,600 tok/s/GPU |
| trainer peak memory | ~130 GB of 178 |
| trainer vs. sampler logprob gap (mean abs, 200k rollouts) | 0.0087 |

How to read these numbers off a run:

- **Decode:** every `/skyrl/v1/generate` request is one turn of exactly 1,065 tokens. Take the
  `http_requests_total{handler="/skyrl/v1/generate"}` counters (`/metrics` on each DP server) over a time window,
  multiply by 1,065, and divide by 8 GPUs. vLLM's `Avg generation throughput` log lines undercount under DP8,
  because Ray deduplicates the 8 engines' near-identical lines.
- **Trainer:** `generate/avg_num_tokens` x 128, divided by `timing/policy_forward_backward`, divided by 8 GPUs.

## Benchmarks without the full loop

- `dsv41_inference_bench.py`: one vLLM engine on this workload shape, with synthetic contexts at any depth. Pass
  `--lora-adapter <dir>` to activate an adapter; without one, the LoRA kernels don't run.
- `dsv41_train_bench.sh`: the trainer step alone, on synthetic multi-turn sequences.

## Caveats

- The workload is synthetic: GSM8k prompts, generated filler observations, and `ignore_eos` turns. It measures
  throughput; rewards and learning are not meaningful. For a real environment, use
  `../run_async_multiturn_dsv41_flash_lora.sh` with your own `ENV_CLASS` / `DATA_DIR` and the trainer overrides from
  `run_dsv41_long_horizon.sh`.
- All 640 rollouts start together, so their contexts grow in lockstep. A production fleet is spread over 0-200k,
  so decode at ~100k is the closest match to a fleet averaging ~100k.
- Numerics: the trainer computes in BF16, with base weights dequantized per layer from the released FP8 / MXFP4.
  The sampler runs the released FP8 / MXFP4 weights.
