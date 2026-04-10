# SkyRL

SkyRL is a full-stack reinforcement learning library for training LLMs, designed for modularity and extensibility.

## Critical Rules

- **Always use `uv run --isolated`** to run commands. Never use bare `python`, `pip`, or `pip install`.
- **Log output to files**: `<cmd> > /tmp/results_1.log 2>&1` for persistence.
- Backend extras (`fsdp`, `megatron`, `jax`, `flashrl`) conflict with each other -- never combine them.

## Test Commands

```bash
# CPU tests
uv run --extra dev --extra jax pytest tests/tx/ tests/tinker/ tests/utils/
uv run --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu/

# GPU tests (requires Ray cluster with GPUs)
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py

# Lint / format
bash format.sh
```

## Training Quick Start

```bash
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron trainer.policy.model.path=<model> environment.env_class=gsm8k ...
```

## Routing Rules

When working on these areas, read the corresponding doc first:

| Area | Read first |
|------|-----------|
| Package management, uv, formatting | `.claude/docs/development.md` |
| Tests, fixtures, CI quirks | `.claude/docs/testing.md` |
| Project layout, Ray actors, config | `.claude/docs/architecture.md` |
| Training entrypoints, Hydra configs | `.claude/docs/training.md` |
| Inference engines, vLLM, PD disagg | `.claude/docs/inference.md` |
| GitHub Actions, Anyscale CI | `.claude/docs/ci.md` |
| New model support, tokenizer quirks | `.claude/docs/contributing.md` |
| Megatron backend | `.claude/docs/backends/megatron.md` |
| FSDP backend | `.claude/docs/backends/fsdp.md` |
| JAX/TPU backend | `.claude/docs/backends/jax.md` |
