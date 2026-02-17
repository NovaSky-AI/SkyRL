# Pythonic Configs Migration — Summary & Remaining Work

## What's Done

### Commits landed
1. **`d1e1b29`** — Pythonic Configs 1/N: Introduce configuration dataclasses (`SkyRLConfig`, `TrainerConfig`, `GeneratorConfig`, `InferenceEngineConfig`, etc.) and migrate tests
2. **`bd462d3`** — Use new SkyRL config in main script; add `InferenceEngineConfig`; migrate DAPO
3. **`08bbcca`** — Pythonic Configs 2/N: Migrate all example shell scripts and their Python entrypoints to typed dataclass configs
4. **`ad93f7f`** — Pythonic Configs 3/N: Remove `DictConfig` from all type hints, migrate remaining entrypoints, clean up APIs

### What ad93f7f covered
- 12 remaining Python entrypoints migrated from `@hydra.main` + `DictConfig` → `SkyRLConfig.from_cli_overrides(sys.argv[1:])`
- All `DictConfig`/`ListConfig` type hints and `isinstance` branches removed from core library
- Unused `model_name` param removed from `SkyRLGymGenerator` and all callers
- `verifiers_generator.py` updated for nested `InferenceEngineConfig` field paths
- Docs updated; `legacy.py` added for backward-compatible YAML translation

### What's been fixed in current uncommitted work
- **`InferenceEngineClient` 5-arg signature**: 9 call sites across 7 files were still passing the old 3-arg `(engines, tokenizer, config)` form. Fixed to `(engines, tokenizer, config.trainer.policy.model.path, config.trainer.policy.lora, config.generator.inference_engine)`.
- **Flat field *assignments***: ~20 files had `cfg.generator.backend = ...` style assignments instead of `cfg.generator.inference_engine.backend = ...`. All assignment sites are fixed.
- **Scripts using `OmegaConf.create()`**: `scripts/multi_node_nccl_test.py` and `scripts/launch_multiple_remote_servers.py` replaced with `SkyRLConfig()`.

CPU tests pass: 396/396.

---

## What Still Needs Fixing

### Flat field *reads* on `GeneratorConfig` (not yet fixed)

These are places that **read** old-style flat fields like `cfg.generator.backend` instead of `cfg.generator.inference_engine.backend`. They compile and pass tests today only because `GeneratorConfig.__getattr__` or legacy translation hasn't been removed yet — but they're wrong and will break.

**Field mapping reference:**
| Old (flat on generator) | New (nested under inference_engine) |
|---|---|
| `cfg.generator.backend` | `cfg.generator.inference_engine.backend` |
| `cfg.generator.num_inference_engines` | `cfg.generator.inference_engine.num_engines` |
| `cfg.generator.inference_engine_tensor_parallel_size` | `cfg.generator.inference_engine.tensor_parallel_size` |
| `cfg.generator.inference_engine_pipeline_parallel_size` | `cfg.generator.inference_engine.pipeline_parallel_size` |
| `cfg.generator.inference_engine_data_parallel_size` | `cfg.generator.inference_engine.data_parallel_size` |
| `cfg.generator.inference_engine_expert_parallel_size` | `cfg.generator.inference_engine.expert_parallel_size` |
| `cfg.generator.gpu_memory_utilization` | `cfg.generator.inference_engine.gpu_memory_utilization` |
| `cfg.generator.async_engine` | `cfg.generator.inference_engine.async_engine` |
| `cfg.generator.run_engines_locally` | `cfg.generator.inference_engine.run_engines_locally` |
| `cfg.generator.weight_sync_backend` | `cfg.generator.inference_engine.weight_sync_backend` |
| `cfg.generator.model_dtype` | `cfg.generator.inference_engine.model_dtype` |
| `cfg.generator.enable_http_endpoint` | `cfg.generator.inference_engine.enable_http_endpoint` |
| `cfg.generator.http_endpoint_host` | `cfg.generator.inference_engine.http_endpoint_host` |
| `cfg.generator.http_endpoint_port` | `cfg.generator.inference_engine.http_endpoint_port` |

**Files with remaining flat field reads (~60 instances):**

1. **`tests/gpu/test_expert_parallel_inference.py`** — 6 reads: `cfg.generator.backend` (×2 in `get_sampling_params_for_backend`), `cfg.generator.inference_engine_tensor_parallel_size` (×2 in `init_inference_engines` calls), `cfg.generator.sampling_params` (OK — stays on generator)
2. **`tests/gpu/gpu_ci/test_inference_engine_client_http_endpoint.py`** — ~25 reads across 6 test functions: `weight_sync_backend`, `async_engine`, `inference_engine_tensor_parallel_size`, `num_inference_engines`
3. **`tests/gpu/gpu_ci/test_pause_and_continue_generation.py`** — ~15 reads across 3 test functions: same fields
4. **`tests/gpu/gpu_ci/test_lora.py`** — 4 reads: `async_engine`, `inference_engine_tensor_parallel_size`, `backend`
5. **`tests/gpu/gpu_ci/test_policy_local_engines_e2e.py`** — 4 reads: `async_engine`, `inference_engine_tensor_parallel_size`, `backend`
6. **`tests/gpu/gpu_ci/test_megatron_worker.py`** — 5 reads: `async_engine`, `inference_engine_tensor_parallel_size`, `backend`
7. **`tests/gpu/gpu_ci/test_engine_generation.py`** — 4 reads: `cfg.generator.backend` in `get_sampling_params_for_backend` calls
8. **`tests/gpu/gpu_ci/test_transfer_strategies_e2e.py`** — 1 read: `cfg.generator.num_inference_engines` in assert
9. **`tests/gpu/test_multi_node_pg.py`** — 1 assignment: `cfg.generator.weight_sync_backend`
10. **`examples/async/async_trainer.py`** — 1 read: `self.cfg.generator.backend` in `get_sampling_params_for_backend`

**Note:** `cfg.generator.sampling_params` is correct and should NOT be changed — `sampling_params` lives directly on `GeneratorConfig`.

**Note:** `skyrl_train/inference_engines/inference_engine_client_http_endpoint.py` lines 263, 284 have `config.generator.backend` in docstring comments only — should update for consistency.

### OmegaConf imports (acceptable, no action needed)
These are expected and should remain:
- `skyrl_train/config/config.py` — `DictConfig, OmegaConf` for `from_dict_config()` bridge
- `skyrl_train/workers/worker.py` — `OmegaConf` for loss config merging
- `skyrl_train/workers/megatron/megatron_worker.py` — `OmegaConf` for internal config handling
- `skyrl_train/workers/megatron/megatron_model_wrapper.py` — `OmegaConf` for Megatron config
- `tests/cpu/test_config.py` — `OmegaConf` for testing config system
- `fsdp_strategy.py`, `fsdp_worker.py` — PyTorch's `ShardedStateDictConfig` (unrelated to Hydra)
