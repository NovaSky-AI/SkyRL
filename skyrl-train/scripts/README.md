# Fleet Training Scripts

Shared scripts for Fleet GRPO training via SkyPilot. These replace the inline shell in each task YAML with reusable, multi-node-aware scripts.

## Scripts

| Script | Purpose |
|--------|---------|
| `fleet-common-setup.sh` | Env validation, Python venv, pip deps, OpenEnv, S3 dataset download, prepare_dataset |
| `fleet-common-run.sh` | Ray cluster setup (multi-node), wandb login, training launch with common hydra overrides |
| `fleet-qwen35-extra-setup.sh` | Qwen3.5-specific: transformers nightly, flash-attn wheel, CUDA toolkit, writes `$HOME/.cuda_env` |

## Usage

### Single-node (e.g., 9B)

```yaml
setup: |
  bash skyrl-train/scripts/fleet-common-setup.sh \
    --openenv-branch deniz/fleet_client \
    --extra-setup skyrl-train/scripts/fleet-qwen35-extra-setup.sh

run: |
  bash skyrl-train/scripts/fleet-common-run.sh \
    --use-python-direct --cuda-env "$HOME/.cuda_env" \
    --set-ulimit --no-pytorch-alloc-conf -- \
    trainer.policy.model.path="Qwen/Qwen3.5-9B" \
    trainer.epochs=20 ...
```

Data root is auto-detected: `/workspace` if writable (RunPod), otherwise `$HOME` (GCP, Lambda, etc.). Override with `--data-root DIR`.

### Multi-node (e.g., 35B on 2 nodes)

Set `num_nodes: 2` in the YAML. The run script auto-detects `SKYPILOT_NODE_RANK`:
- **Rank 0 (head):** starts Ray head, calculates `TOTAL_GPUS = gpus_per_node * num_nodes`, launches training
- **Rank > 0 (workers):** joins Ray cluster, sleeps

No changes needed in the run block — multi-node works automatically.

## Setup Script Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--openenv-branch BRANCH` | `deniz/fleet_client` | OpenEnv git ref to install |
| `--extra-setup SCRIPT` | *(none)* | Script to source after `uv sync` (model-specific deps) |
| `--data-root DIR` | auto-detect | Root for dataset download (`DIR/data/fleet/`). Uses `/workspace` if writable, else `$HOME` |
| `--skip-uv-isolated` | `false` | Flag for configs that use `python` directly |

## Run Script Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root DIR` | auto-detect | Root for dataset files. Uses `/workspace` if writable, else `$HOME` |
| `--ckpt-root DIR` | same as data-root | Root for checkpoints, exports, and tmp dir |
| `--use-python-direct` | `false` | Use `python -m` instead of `uv run --isolated` |
| `--cuda-env FILE` | *(none)* | Source this file for CUDA_HOME (e.g., `$HOME/.cuda_env`) |
| `--set-ulimit` | `false` | Set `ulimit -n 65536` (needed for Ray+vLLM) |
| `--no-pytorch-alloc-conf` | `false` | Skip setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `--nccl-heartbeat SEC` | *(none)* | Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (useful for multi-node) |
| `--` | | Everything after this is passed as hydra overrides |

## Common Hydra Overrides (auto-injected by run script)

These are always set by `fleet-common-run.sh` — don't repeat them in the YAML:

- `data.train_data`, `data.val_data` (from `--data-root`)
- `environment.env_class=fleet_task`
- `environment.skyrl_gym.fleet_task.tasks_file`
- `trainer.placement.colocate_all=true`, `trainer.strategy=fsdp2`
- `trainer.placement.{policy,ref}_num_gpus_per_node`, `{policy,ref}_num_nodes`
- `generator.num_inference_engines=$TOTAL_GPUS`

## Environment Variables

Required in SkyPilot YAML `envs:`:

| Variable | Description |
|----------|-------------|
| `FLEET_API_KEY` | Fleet API key |
| `WANDB_API_KEY` | Weights & Biases key |
| `AWS_ACCESS_KEY_ID` | S3 dataset download |
| `AWS_SECRET_ACCESS_KEY` | S3 dataset download |
| `MODALITY` | `tool_use` or `computer_use` |
| `DATA_VERSION` | S3 dataset version (e.g., `v54`) |
| `S3_DATASET_BUCKET` | Dataset bucket name |
| `INFERENCE_BACKEND` | `vllm` or `sglang` |

## Adding a New Model Config

1. Copy an existing YAML (e.g., `openenv-fleet-grpo-qwen3_5-9b.yaml`)
2. If the model needs special deps, create a new `fleet-<model>-extra-setup.sh`
3. Update `setup:` to reference your extra-setup script
4. Update `run:` with model-specific hydra overrides after `--`
5. Add the task name to `.github/workflows/openenv-fleet-train.yaml`

## GCP Support

GCP requires the correct GPU image for H200/B200 (default SkyPilot image has driver 535, needs 550+):

```yaml
- accelerators: H200:8
  cloud: gcp
  use_spot: true
  image_id: projects/deeplearning-platform-release/global/images/common-cu128-ubuntu-2204-nvidia-570-v20260305
```
