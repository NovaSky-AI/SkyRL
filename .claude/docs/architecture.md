# Architecture

- Refer to `docs/content/docs/getting-started/overview.mdx` for a detailed system overview of SkyRL's training backend (non-Jax). 
- Refer to `docs/content/docs/tinker/architecture.mdx` for an overview of SkyRL's tinker API server implementation.

## Project Structure

```
skyrl/                  # Core library
├── backends/           # Backend implementations
│   └── skyrl_train/    # FSDP/Megatron training backend
│       ├── distributed/        # Dispatch, FSDP/Megatron strategies
│       ├── inference_servers/  # HTTP inference path (RemoteInferenceClient, vLLM servers, router); base.py + engine_utils.py hold shared interfaces/helpers
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

- **Ray orchestration**: Training workers and inference engines run as Ray actors.
- **Config hierarchy**: `SkyRLTrainConfig` → `TrainerConfig`, `GeneratorConfig`, `DataConfig`, `EnvironmentConfig`. Accessed as `cfg.trainer.*`, `cfg.generator.*`, etc.
- **CLI**: Config uses OmegaConf + dataclasses — OmegaConf for CLI parsing and merging, loaded into dataclasses for better typing. Pass overrides as `key=value` CLI args.
- **Backend selection**: `trainer.strategy` chooses backend — `fsdp` (default), `megatron`, or `jax`.

## Weight Sync

Training weights are synced to inference engines via:
- **Broadcast strategy**: NCCL-based, for non-colocated setups.
- **CUDA IPC strategy**: For colocated setups (`colocate_all=true`).

## Environments

Defined in `skyrl-gym/skyrl_gym/envs/`. Each env extends `BaseTextEnv` with `step()` (and typically a `_get_reward()` helper).