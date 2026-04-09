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
- **`tests/backends/skyrl_train/gpu/gpu_ci/conftest.py`** — class/module-scoped fixtures, builds Ray env vars.
- Test output from Ray workers appears in **stderr**, not stdout.

## Known Quirks

- **`_SKYRL_USE_NEW_INFERENCE`**: Defaults to `False` (0). Controls old vs new inference codepath. Set to `1` to enable new inference. Tests that skip on this flag are marked with `@pytest.mark.skipif`.
- **Megatron tests**: Require `NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0` env vars.
