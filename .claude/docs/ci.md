# CI

## GitHub Actions

Workflows in `.github/workflows/`. GPU tests run on Anyscale clusters via `anyscale job submit`.

## CI Directory

- `ci/` contains job YAML configs and run scripts.
- GPU CI script: `ci/gpu_ci_run_skyrl_train.sh` — runs broad test suite, then new-inference-specific tests.
- GPU CI image: `novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`.

## CI Test Phases

1. **Phase 1**: Broad test suite with no `_SKYRL_USE_NEW_INFERENCE` env var (uses default=False, legacy codepath).
2. **Phase 2**: Test files run explicitly with `_SKYRL_USE_NEW_INFERENCE=1` (new inference codepath).
