# DeepSeek-V4.1-Flash LoRA RL on Modal: long multi-turn agent workload, 16 GPUs

An end-to-end fully-async RL run (vLLM sampling and Megatron training at the same time, on separate GPUs) shaped
like a long tool-use agent workload at full context, which reports sampler and trainer throughput from the same run.

| | this run (defaults) |
|---|---|
| GPUs | 16x B200 (`GPU=B300:8` for B300): 1 Megatron trainer node + 1 vLLM node |
| model | `deepseek-ai/DeepSeek-V4.1-Flash` (released FP8 / MXFP4 weights), LoRA r32 on attention, shared and routed experts |
| rollout | 69 turns; each turn the policy generates exactly 1,065 tokens (`ignore_eos`) and the environment answers with a ~1,870-token tool observation; contexts grow to ~200k tokens |
| concurrency | 640 rollouts in flight = 80 per vLLM GPU; 128 rollouts (16 prompts x 8) per optimizer step, so ~5 steps of async policy lag |
| trainer | context parallel 8 (one 200k-token sequence per 8-GPU node), expert parallel 8, packed sequences, stateful full recompute, frozen base weights stored as MXFP8 |
| sampler | vLLM with data-parallel attention + expert parallelism (DP8 + EP8: each GPU caches only its own sessions' KV, ~93M tokens per B200 node), MXFP4 MoE + LoRA on FlashInfer's TRT-LLM kernel, two-kernel MoE-LoRA with tuned tiles, prefix caching with session-sticky routing |
| algorithm | GRPO-style group advantages, rollout-anchored importance-sampling loss, routing replay (R3), score centering |

## Run it

Prerequisites: the Modal CLI logged in, a W&B secret named `wandb` (or `MODAL_SECRETS= LOGGER=console`).

```bash
# once (CPU only): download the model to the `dsv41-models` volume and make it text-only in place,
# build the 69-turn dataset, stage FlashInfer's kernel package on the volume
bash examples/train_integrations/modal/bench/launch_depthfirst_sim.sh setup

# the run (detached): ~50 min startup, ~1.5-2 h of 69-turn rollouts, then a training step every ~35 min
bash examples/train_integrations/modal/bench/launch_depthfirst_sim.sh run
```

Logs: `skyrl-logs/<RUN_NAME>/run.log` on the `skyrl-data` volume (RUN_NAME defaults to `dsv41_depthfirst_sim`), W&B
project `dsv41_long_horizon`. Knobs (env): `GPU`, `MAX_TURNS`, `MAX_MODEL_LEN`, `CP`, `VLLM_DP`, `TRAIN_BATCH_SIZE`,
`N_SAMPLES_PER_PROMPT`, `MAX_STALENESS_STEPS`, `MAX_TRAINING_STEPS`, `MODELS_VOLUME`, `TIMEOUT`.

## What to read off

- **Sampler**: every `/skyrl/v1/generate` request is one turn of exactly 1,065 generated tokens, so the most exact rate
  is the vLLM servers' `http_requests_total{handler="/skyrl/v1/generate"}` counters (`/metrics` on each DP server)
  over a time window x 1,065 / 8 GPUs. vLLM's `Avg generation throughput` log lines undercount under DP8: Ray
  deduplicates the 8 engines' near-identical lines.
- **Trainer**: tokens per step (`generate/avg_num_tokens` x 128) over `timing/policy_forward_backward` (forward +
  backward) or over `timing/run_training` (also the old-logprob forward pass), divided by 8 GPUs.

## Measured (16x B200, defaults, depthfirst-length rollouts)

Each training step: 128 rollouts x ~200.7k tokens = ~25.7M tokens.

| | Modal B200 (this launcher) | Anyscale B200 (same config) |
|---|---|---|
| decode, ~20k-108k context | | **~1,420 tok/s/GPU** |
| decode, ~120k-196k context | | ~1,130-1,340 tok/s/GPU |
| trainer forward + backward (step 2) | **~2,600 tok/s/GPU** (1,234 s) | ~2,300 tok/s/GPU (1,406 s) |
| trainer, whole step incl. the old-logprob forward pass (step 2) | **~1,890 tok/s/GPU** (1,699 s) | ~1,570 tok/s/GPU (2,049 s) |
| trainer peak memory | | 129.5 GB (of 178) |
| trainer vs. sampler logprob gap (mean abs, 200k rollouts) | | 0.0087 |

Decode is counted from the vLLM servers' completed turns; on Modal it is in W&B / `run.log`.

The "whole step" rows above include a separate old-logprob forward pass (~25-35% of a 200k step). The launcher now
sets `trainer.old_logprobs_from_first_minibatch=true`: with one mini-batch per step the old logprobs come from the
training forward itself and that pass is skipped (validated on B200: `run_training` time = `policy_train` time), so a
step costs about the forward + backward time.

## Caveats

- Weight sync under data-parallel serving (VLLM_DP > 1): vLLM's DP ranks step in lockstep, so pausing them in "keep"
  mode (in-flight requests frozen) and then pushing the new LoRA weights over NCCL can deadlock (ranks still in the
  DP all-reduce never join the weight broadcast). It shows up at the first sync that happens while rollouts are being
  generated (step 2 here). `RemoteInferenceClient.pause_generation` now pauses in "wait" mode under DP (in-flight turns
  finish first, so every rank is idle before the update); `SKYRL_DP_PAUSE_MODE` overrides it.

- Synthetic workload: GSM8k prompts, generated filler observations, and `ignore_eos` turns. It is a throughput run;
  rewards and learning are not meaningful.
- All 640 rollouts start together, so their contexts move in lockstep (a production fleet is spread over 0-200k);
  decode at ~100k is the closest match to a fleet averaging ~100k.
- Numerics: BF16 compute with the base weights dequantized from the released FP8 / MXFP4 per layer; the sampler
  runs the released FP8 / MXFP4 weights.
