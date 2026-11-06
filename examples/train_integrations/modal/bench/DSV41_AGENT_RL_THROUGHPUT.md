# DeepSeek-V4.1-Flash LoRA RL on Modal: long multi-turn agent workload, 16 GPUs

An end-to-end fully-async RL run (vLLM sampling and Megatron training at the same time, on separate GPUs) shaped
like a long tool-use agent workload, which reports sampler and trainer throughput from the same run.

| | this run |
|---|---|
| GPUs | 16x B300: 1 Megatron trainer node + 1 vLLM node |
| model | `deepseek-ai/DeepSeek-V4.1-Flash` (released FP8 / MXFP4 weights), LoRA r32 on attention, shared and routed experts |
| rollout | 16 turns; each turn the policy generates exactly 1,065 tokens (`ignore_eos`) and the environment answers with a ~1,870-token tool observation; ~45k tokens per rollout |
| concurrency | 640 rollouts in flight = 80 per vLLM GPU (16 prompts x 8 samples per step, up to 4 steps stale) |
| trainer | context parallel 4 x data parallel 2, expert parallel 8, packed sequences, full recompute, frozen base weights stored as MXFP8 |
| sampler | vLLM TP8, MXFP4 MoE + LoRA on FlashInfer's TRT-LLM kernel, prefix caching, 32k prefill budget, mixed-step CUDA graphs |
| algorithm | GRPO-style group advantages, rollout-anchored importance-sampling loss, routing replay (R3), score centering |

## Run it

Prerequisites: the Modal CLI logged in, a W&B secret named `wandb` (or `MODAL_SECRETS= LOGGER=console`).

```bash
# once (~1-2 h, CPU only): download the model to the `dsv41-models` volume and make it text-only in place,
# build the dataset, stage FlashInfer's kernel package on the volume
bash examples/train_integrations/modal/bench/launch_depthfirst_sim.sh setup

# the run (detached; ~45 min startup, then a step every few minutes)
bash examples/train_integrations/modal/bench/launch_depthfirst_sim.sh run
```

Logs: `skyrl-logs/dsv41_depthfirst_sim/run.log` on the `skyrl-data` volume, W&B project `dsv41_long_horizon`.
Knobs (env): `MAX_TURNS`, `MAX_MODEL_LEN`, `TRAIN_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `MAX_STALENESS_STEPS`,
`MAX_TRAINING_STEPS`, `GEN_STAGGER_S`, `MODELS_VOLUME`, `GPU`.

## What to read off

- **Sampler**: vLLM's `Avg generation throughput` (per 10 s, in `run.log`) divided by 8 GPUs.
- **Trainer**: tokens per step (`generate/avg_num_tokens` x 128) over `timing/policy_forward_backward` (forward +
  backward) or over `timing/run_training` (also the old-logprob forward pass), divided by 8 GPUs.

## Measured

| | per GPU |
|---|---|
| sampler, B300 | TBD |
| trainer fwd+bwd, B300 | TBD |
| trainer fwd+bwd, B200 (validation run) | ~1,960 tok/s (5.77M tokens / 368 s), peak 102 GB |

## Caveats

- Synthetic workload: GSM8k prompts, generated filler observations, and `ignore_eos` turns. It is a throughput run;
  rewards and learning are not meaningful.
- ~45k-token rollouts, not ~200k: 640 rollouts of ~200k tokens do not fit the TP8 KV cache. For ~200k rollouts set
  `MAX_TURNS=69 MAX_MODEL_LEN=262144` and fewer rollouts in flight (e.g. `MAX_STALENESS_STEPS=1`); the trainer config
  is the one validated at 200k on B300 (~3,500 tok/s/GPU forward + backward, 198 GB peak).
- Numerics: BF16 compute with the base weights dequantized from the released FP8 / MXFP4 per layer; the sampler
  runs the released FP8 / MXFP4 weights.
