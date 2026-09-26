# SkyRL

SkyRL is a full-stack reinforcement learning library for training LLMs, designed for modularity and extensibility.

## Critical Rules

- **Always use `uv run --isolated`** to run commands. Never use bare `python`, `pip`, or `pip install`.
- **Log output to files**: `<cmd> > /tmp/results_1.log 2>&1` for persistence.
- Backend extras (`fsdp`, `megatron`, `jax`) conflict with each other -- never combine them.
- Always read the relevant documentation files in `.agents/docs` before troubleshooting or working on any changes. Follow the routing rules below.

## Test Commands

These mirror `.github/workflows/cpu_skyrl.yaml` and `cpu_jax.yaml`; keep them in sync.

```bash
# CPU tests -- skyrl_train. Needs a backend extra for `ray`: without one, collection fails
# in tests/backends/skyrl_train/conftest.py with "No module named 'ray'". CI splits on the
# `vllm` marker because the two halves need different extras.
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "not vllm"
uv run --isolated --extra fsdp --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "vllm"

# CPU tests -- tx / tinker / utils
uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tx tests/backends/test_jax_backend.py --ignore=tests/tx/gpu
uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tinker tests/utils --ignore=tests/tinker/skyrl_train
uv run --isolated --extra fsdp --extra tinker --extra dev pytest tests/tinker/skyrl_train/

# GPU tests (requires Ray cluster with GPUs)
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py

# The opt-in h100 GPU marker is auto-skipped unless requested by name:
uv run --isolated --extra dev --extra megatron pytest -m h100 tests/backends/skyrl_train/gpu/gpu_ci/megatron/

# Lint / format (needs pre-commit; `bash format.sh` fails if it is not on PATH)
uv run --isolated --extra dev pre-commit run --all-files
```

Tests that connect to Ray call bare `ray.init()`, which attaches to any cluster already
running on the box -- including a live training cluster, whose workers then die. Run them
with `RAY_ADDRESS=local` on a machine that has one up.

## Training Quick Start

```bash
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron trainer.policy.model.path=<model> environment.env_class=gsm8k ...
```

## Routing Rules

When working on these areas, read the corresponding doc first:

| Area | Read first |
|------|-----------|
| Package management, uv, formatting | `.agents/docs/development.md` |
| Overall guide for modifying or working with SkyRL | `.agents/docs/contributing.md` |
| Tests, fixtures, CI quirks | `.agents/docs/testing.md` |
| Project layout, Ray actors, config | `.agents/docs/architecture.md` |
| Training entrypoints, configs | `.agents/docs/training.md` |
| Inference engines, vLLM, PD disagg | `.agents/docs/inference.md` |
| GitHub Actions, Anyscale CI | `.agents/docs/ci.md` |
| Tinker API server | `.agents/docs/tinker.md` |
| Megatron backend | `.agents/docs/backends/megatron.md` |
| FSDP backend | `.agents/docs/backends/fsdp.md` |
| JAX/TPU backend | `.agents/docs/backends/jax.md` |
| Weight sync | `.agents/docs/weight_sync.md` |
| Bumping megatron-core / megatron-bridge, or Megatron patches / vendored code | `skyrl/backends/skyrl_train/patches/megatron/README.md` |


## Troubleshooting

For troubleshooting training runs with SkyRL:

1. Go through the troubleshooting section in the docs for known errors: `docs/content/docs/troubleshooting/troubleshooting.mdx`
2. Go through the contributing guide for overall guidelines: `.agents/docs/contributing.md`
