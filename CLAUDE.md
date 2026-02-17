# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SkyRL is a full-stack reinforcement learning library for LLM post-training. It consists of four main packages:

- **skyrl-train**: The core training framework for RL (PPO, GRPO, etc.) with Ray-based distributed training
- **skyrl-gym**: Gymnasium-style environment interface for RL tasks (math, code, search, SQL)
- **skyrl-agent**: Agent layer for multi-turn, long-horizon agent training
- **skyrl-tx**: Cross-platform REST API for model post-training (Tinker-like)

## Build and Development Commands

### Installation (skyrl-train)
```bash
cd skyrl-train
uv sync --extra vllm  # or --extra sglang, --extra mcore for Megatron
source .venv/bin/activate
```

### Running Tests
```bash
# CPU tests (skyrl-train)
cd skyrl-train
uv run --frozen pytest tests/cpu/

# Single test file
uv run --isolated --extra dev pytest -s tests/cpu/test_config.py

# GPU tests (requires GPUs)
uv run --isolated --extra dev --extra vllm pytest -s tests/gpu/gpu_ci -m "not (sglang or integrations or megatron)"

# skyrl-gym tests
cd skyrl-gym
uv run --frozen pytest tests/
```

### Linting and Formatting
```bash
# From repo root
bash format.sh  # Runs pre-commit with ruff + black
```

### Running Training
```bash
cd skyrl-train
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Example: GSM8K training
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash examples/gsm8k/run_gsm8k.sh
```

## Architecture

### Configuration System
The codebase is migrating from Hydra/YAML to Python dataclasses for configuration:
- `skyrl_train/config/config.py`: Typed configuration dataclasses (`SkyRLConfig`, `TrainerConfig`, `GeneratorConfig`, etc.)
- `skyrl_train/config/ppo_base_config.yaml`: Default YAML config (still used as base)
- Configs can be constructed via `SkyRLConfig.from_dict_config()` or `SkyRLConfig.from_cli_overrides()`

### Core Training Components
- `skyrl_train/trainer.py`: `RayPPOTrainer` - main training loop orchestration
- `skyrl_train/entrypoints/main_base.py`: `BasePPOExp` - experiment setup and entry point
- `skyrl_train/workers/worker.py`: Ray actor-based distributed workers
- `skyrl_train/workers/fsdp/`: FSDP/FSDP2 training backend workers
- `skyrl_train/workers/megatron/`: Megatron training backend workers

### Generation and Inference
- `skyrl_train/generators/base.py`: `GeneratorInterface` - abstract generator interface
- `skyrl_train/generators/skyrl_gym_generator.py`: `SkyRLGymGenerator` - integrates skyrl-gym environments
- `skyrl_train/inference_engines/`: vLLM/SGLang inference engine wrappers
- `skyrl_train/inference_engines/inference_engine_client.py`: Client for managing inference engines

### Environment System (skyrl-gym)
- `skyrl_gym/core.py`: `Env` base class with `init()`, `step()`, `close()` methods
- `skyrl_gym/envs/`: Environment implementations (gsm8k, sql, search, lcb, etc.)
- Environments return `EnvStepOutput` with observations, reward, done flag, and metadata

### Training Strategies
- `fsdp2` (default): FSDP2 for distributed training
- `fsdp`: Original FSDP
- `megatron`: Megatron-Core for 5D parallelism (TP, PP, CP, EP, DP)

### Placement Configurations
- `colocate_all=true`: Training and inference share the same GPUs
- `colocate_all=false`: Separate GPU pools for training vs inference

## Code Style
- Line length: 120 characters
- Formatters: ruff (with --fix) and black
- Python 3.12 required for skyrl-train

## Test Markers
Tests use pytest markers for selective execution:
- `vllm`: vLLM-specific tests
- `sglang`: SGLang-specific tests
- `megatron`: Megatron backend tests
- `integrations`: External integration tests
