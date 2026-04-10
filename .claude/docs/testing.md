# Testing

## CPU Tests

```bash
# Core library tests
uv run --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu/

# JAX / Tinker / Utils
uv run --extra dev --extra jax pytest tests/tx/ tests/tinker/ tests/utils/
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
