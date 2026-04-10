# Architecture

## Project Structure

```
skyrl/                  # Core library
├── backends/           # Backend implementations
│   └── skyrl_train/    # FSDP/Megatron training backend
│       ├── distributed/        # Dispatch, FSDP/Megatron strategies
│       ├── inference_engines/  # Legacy inference path
│       ├── inference_servers/  # New inference path
│       ├── weight_sync/        # Weight extraction and transfer
│       └── workers/            # FSDP/Megatron workers
├── train/              # Training entrypoints, config, dataset, generators, trainer
│   ├── config/         # Hydra YAML configs (ppo_base, megatron, skyrl_gym)
│   └── entrypoints/    # main_base.py is the primary training entrypoint
├── tinker/             # Tinker API server (FastAPI + SQLModel)
├── tx/                 # JAX-native model implementations (Flax/NNX)
└── utils/

skyrl-gym/              # RL environment package (separate sub-package)
├── skyrl_gym/envs/     # Environments: gsm8k, aime, lcb, search, sql, etc.

tests/                  # Mirrors skyrl/ structure
├── backends/skyrl_train/gpu/gpu_ci/   # GPU CI tests
examples/train/         # Example training scripts per model/backend
```

## Key Patterns

- Refer to `docs/content/docs/getting-started/overview.mdx` for a detailed system overview of SkyRL's training backend (non-Jax). 
- Refer to `docs/content/docs/tinker/architecture.mdx` for an overview of SkyRL's tinker API server implementation.

- **Ray orchestration**: Training workers and inference engines run as Ray actors.
- **Config hierarchy**: `SkyRLTrainConfig` → `TrainerConfig`, `GeneratorConfig`, `DataConfig`, `EnvironmentConfig`. Accessed as `cfg.trainer.*`, `cfg.generator.*`, etc.
- **CLI**: Config uses Hydra/OmegaConf — Hydra for YAML config composition, OmegaConf for CLI parsing and merging. Pass overrides as `key=value` CLI args.
- **Backend selection**: `trainer.strategy` chooses backend — `fsdp2` (default), `fsdp`, `megatron`, or `jax`.

## Weight Sync

Training weights are synced to inference engines via:
- **Broadcast strategy**: NCCL-based, for non-colocated setups.
- **CUDA IPC strategy**: For colocated setups (`colocate_all=true`).

## Environments

Defined in `skyrl-gym/skyrl_gym/envs/`. Each env extends `BaseTextEnv` with `step()` (and typically a `_get_reward()` helper). Available: gsm8k, aime, lcb, search, searchcode, sql.