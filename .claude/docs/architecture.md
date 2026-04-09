# Architecture

## Project Structure

```
skyrl/                  # Core library
├── backends/           # Backend implementations
│   └── skyrl_train/    # GPU training backend
│       ├── distributed/        # Dispatch, FSDP/Megatron strategies
│       ├── inference_engines/  # vLLM engine clients (legacy)
│       ├── inference_servers/  # Remote inference routing (new)
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

- **Ray orchestration**: Training workers and inference engines run as Ray actors. Use `ray.remote()` with `spawn` start method (not fork).
- **Config hierarchy**: `SkyRLTrainConfig` → `TrainerConfig`, `GeneratorConfig`, `DataConfig`, `EnvironmentConfig`. Accessed as `cfg.trainer.*`, `cfg.generator.*`, etc.
- **Hydra**: Config uses OmegaConf. Pass overrides as `key=value` CLI args.
- **Backend selection**: `trainer.strategy` chooses backend — `fsdp2` (default), `fsdp`, `megatron`, or `jax`.

## Weight Sync

Training weights are synced to inference engines via:
- **Broadcast strategy**: NCCL-based, for non-colocated setups.
- **CUDA IPC strategy**: For colocated setups (`colocate_all=true`).
- Four-phase protocol: `init_weight_transfer_engine` → `start_weight_update` → `update_weights` → `finish_weight_update`.

## Environments

Defined in `skyrl-gym/skyrl_gym/envs/`. Each env implements `BaseTextEnv` with `_get_reward()` and `step()`. Available: gsm8k, aime, lcb, search, searchcode, sql.
