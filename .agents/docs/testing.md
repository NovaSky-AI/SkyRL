# Testing


Mirrors `.github/workflows/cpu_skyrl.yaml` / `cpu_jax.yaml` -- keep in sync.

```bash
# Core library tests. A backend extra is required: `ray` ships in `skyrl-train`/`fsdp`, and
# without one, collection fails in tests/backends/skyrl_train/conftest.py with
# "No module named 'ray'". CI splits on the `vllm` marker since the halves need different extras.
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "not vllm"
uv run --isolated --extra fsdp --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "vllm"

# JAX / Tinker / Utils
uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tx tests/backends/test_jax_backend.py --ignore=tests/tx/gpu
uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tinker tests/utils --ignore=tests/tinker/skyrl_train
uv run --isolated --extra fsdp --extra tinker --extra dev pytest tests/tinker/skyrl_train/
```

**Running CPU tests on a machine with a live Ray cluster.** `tests/backends/skyrl_train/conftest.py`
calls bare `ray.init()`, which attaches to whatever cluster is already up -- including a training
cluster -- and its workers then die there, so the results are meaningless and the training job is
disturbed. Set `RAY_ADDRESS=local` to force an isolated instance.

## Tests for `patches/`

Tests that target code under `skyrl/backends/skyrl_train/patches/` mirror that layout so they can be
tracked and deleted with the patch: CPU tests under `tests/backends/skyrl_train/patches/`, GPU tests
under `tests/backends/skyrl_train/gpu/gpu_ci/patches/` (keep the `megatron` marker on Megatron
ones). See `skyrl/backends/skyrl_train/patches/megatron/README.md`.

## Opt-in hardware markers

`h100` marks tests needing hardware the default runners lack. `tests/backends/skyrl_train/gpu/conftest.py`
auto-skips them unless the marker is named explicitly, so `-m megatron_models` never picks them up:

```bash
uv run --isolated --extra dev --extra megatron pytest -m h100 tests/backends/skyrl_train/gpu/gpu_ci/megatron/
```

## GPU Tests

Always use `--isolated` for GPU tests:

```bash
# FSDP-based tests
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py -v

# Megatron tests
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py -v

# Specific test
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py -k "test_name" -v
```

## Ray Fixtures

- **`tests/backends/skyrl_train/gpu/conftest.py`** — function-scoped `ray_init_fixture` for GPU tests.
- **`tests/backends/skyrl_train/gpu/gpu_ci/conftest.py`** — function-scoped `ray_init_fixture` and class-scoped `class_scoped_ray_init_fixture`, builds Ray env vars.
- Test output from Ray workers appears in **stderr**, not stdout.
- For GPU-based tests, always use one of the ray init fixtures. If possible, use a shared init at the class/ module level to avoid repeated init/ teardown. When in doubt, use function-scoped `ray_init_fixture`. 


## Anti-patterns

- Manual `ray.init`, `ray.shutdown` and `ray.kill` in the tests.
