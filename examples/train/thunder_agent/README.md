# ThunderAgent GSM8K Run

This directory contains the ThunderAgent fully-async GSM8K training script:

- Baseline: [examples/train/fully_async/fully_async_run_gsm8k.sh](/home/ergt/SkyRL/examples/train/fully_async/fully_async_run_gsm8k.sh)
- ThunderAgent: [examples/train/thunder_agent/fully_async_run_gsm8k.sh](/home/ergt/SkyRL/examples/train/thunder_agent/fully_async_run_gsm8k.sh)

## Run

Prepare GSM8K first:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

Run the baseline:

```bash
bash examples/train/fully_async/fully_async_run_gsm8k.sh
```

Run the ThunderAgent version:

```bash
bash examples/train/thunder_agent/fully_async_run_gsm8k.sh
```

The ThunderAgent script uses:

- `examples.train.thunder_agent.main_fully_async_thunder_agent`
- the new HTTP inference layer (`_SKYRL_USE_NEW_INFERENCE=1`)
- `generator.batched=false`
- non-colocated training / inference workers

With the default script values, it expects:

- `NUM_POLICY_GPUS=2`
- `NUM_INFERENCE_GPUS=2`

That means 8 GPUs total:

- 2 for policy
- 2 for critic
- 2 for ref
- 2 for inference

## Rollout Speedup

Use `examples/train/fully_async/fully_async_run_gsm8k.sh` as the baseline rollout path.

ThunderAgent is meant to improve rollout throughput by routing requests through program-aware scheduling and by coordinating rollout blocking around weight sync. Fill in the measured result here once you have your final benchmark number:

- Baseline rollout throughput: `<fill in>`
- ThunderAgent rollout throughput: `<fill in>`
- Speedup vs baseline rollout: `<fill in>x`
