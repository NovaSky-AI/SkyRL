## Guide: Mini-SWE-Agent + SkyRL

This directory contains an integration to train a coding agent on the SWE-Bench task using [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) and SkyRL.

To start training, follow three simple steps:
1) Prepare the SWE-Gym dataset.
2) Configure your environment backend (Podman).
3) Launch training!

Start by following the SkyRL [installation instructions](https://skyrl.readthedocs.io/en/latest/getting-started/installation.html), then enter the `skyrl-train` directory:
```bash
cd SkyRL/skyrl-train
```

## How it works

The Mini-SWE-Agent integration implements a custom `MiniSweAgentGenerator` that uses Mini-SWE-Agent to generate trajectories for SWE-Bench instances. The workflow consists of:

1. **Generation**: Initialize a sandbox environment and generate a trajectory using Mini-SWE-Agent configured with SkyRL's HTTP endpoint, producing a git patch.
2. **Evaluation**: Apply the generated patch to a fresh environment and run the evaluation script to determine if the instance was resolved.

We launch a Ray task per trajectory to scale this across all nodes in the cluster.

### 1) Prepare the dataset

We use [SWE-Gym](https://huggingface.co/SWE-Gym), specifically the subset from [SumanthRH/SWE-Gym-Subset](https://huggingface.co/datasets/SumanthRH/SWE-Gym-Subset).

Execute the following command:
```bash
uv run --isolated examples/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset # or modify to our desired path
```

### 2) Configure environment backend

**Prerequisites**: Install the required environment backend. By default, we use [Podman](https://podman.io/docs). This can be modified in `examples/mini_swe_agent/swebench.yaml`.

### 3) Launch training

We provide example scripts for different model sizes:

**Qwen3-8B** (requires 1x 8xH100 node):
```bash
bash examples/mini_swe_agent/run_mini_swe_8B.sh
```

**Qwen3-Coder-30B** (requires 2x 8xH100 nodes):
```bash
bash examples/mini_swe_agent/run_mini_swe_30B.sh
```

Make sure to update the `DATA_DIR` variable in the bash script if you saved the data to a custom path.

All training parameters can be modified in the run scripts, such as model choice, GRPO group size, or training batch size.

## Troubleshooting

For issues with SkyRL or the Mini-SWE-Agent integration, please [open an Issue](https://github.com/NovaSky-AI/SkyRL/issues/new).

### Common Issues

- **Context length errors**: If you see `ValueError: The decoder prompt (length xxxx) is longer than the maximum model length`, increase `max_input_length` and `max_generate_length` or reduce steps in `swebench.yaml`.

- **All zero rewards**: If rewards are consistently zero, the task may be too difficult. Consider:
  - Filtering data for a better mix of easy/hard samples
  - Using a stronger base model
  - Increasing `step_limit` in `swebench.yaml`

- **Argument list too long**: For very large git patches, you might notice evaluation errors such as `Argument list too long: 'podman'`. This is because we apply the model's git patch by passing it as a CLI argument, and for large patches, you can hit the system's `ARG_MAX` limits. On modern systems, this limit is about ~1MB. We make a simple assumption that such large patches are meant to be incorrect.

- **Podman UID errors**: If running podman within a container, you might hit errors due to insufficient UIDs. To resolve this, you have two options on Linux-based machines:
  1. Edit the `/etc/subuid` and `/etc/subgid` files to use a larger range of UIDs, like `100000-1100000`
  2. Set `ignore_chown_errors=true` in Podman's containers.conf

## Configuration

Beyond the configuration for SkyRL in the training script, the task-specific configuration file is `examples/mini_swe_agent/swebench.yaml`, which controls:
- Environment backend settings
- Step limits for agent execution
- Tool configurations for Mini-SWE-Agent

For more details, refer to the [documentation](https://skyrl.readthedocs.io/en/latest/examples/mini_swe_agent.html).
