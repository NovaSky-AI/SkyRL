# CI

## GitHub Actions

Workflows in `.github/workflows/`. GPU tests run on Anyscale clusters via `anyscale job submit`.

## CI Directory

- `ci/` contains job YAML configs and run scripts. Uses Anyscale `l4_ci` compute config.
- GPU CI scripts: 
    - SkyRL-Train (FSDP): `ci/gpu_ci_run_skyrl_train.sh`
    - SkyRL-Train (Megatron): `ci/gpu_ci_run_skyrl_train_megatron.sh`
    - JAX: `ci/gpu_ci_run.sh`
- GPU CI images:
    - FSDP/default: `novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`
    - Megatron: `novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8-megatron`
