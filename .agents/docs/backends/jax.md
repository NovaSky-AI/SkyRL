# JAX/TPU Backend

## Overview

JAX-native model implementations in `skyrl/tx/`:

```
skyrl/tx/
├── layers/     # Custom layers
├── models/     # Model implementations (llama3, qwen3, qwen3_5, deepseekv3)
├── loaders/    # Weight loading from HF checkpoints
└── utils/      # JAX utilities
```

Uses Flax/NNX for model definitions.

## Extras

- `jax` extra for JAX dependencies.
- `gpu` extra for JAX on GPU (CUDA).
- `tpu` extra for JAX on TPU (uses custom index `jax-tpu`).
- `mps` extra for JAX on Apple Silicon through `jax-mps` (macOS 14+).

## Tests

```bash
uv run --extra dev --extra jax pytest tests/tx/ -v
```

On Apple Silicon:

```bash
JAX_PLATFORMS=mps uv run --isolated --extra dev --extra jax --extra mps --extra tinker \
  pytest tests/backends/test_jax_backend.py -v
```
