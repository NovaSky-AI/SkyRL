# ThunderAgent-SkyRL Integration

## Run

Prepare GSM8K first:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

Run the fully async training script with Thunderagent:

```bash
bash examples/train/thunder_agent/run_thunder_agent_gsm8k.sh
```

## Rollout Speedup

### Settings

- Datasets: DCAgent/exp-rdb-r2egym.
- Hardware: `4 x 8 H100` training nodes, `1 x 8 H100` rollout node.
- Training strategy: fully-async GRPO with n_samples_per_prompt=4, train_batch_size=64(256 sampled trajectories per update).
- 40 steps and 10 epochs.

### Speedup

![TA vs no-TA stitched 40-step Harbor timeline](./docs/speedup.png)
