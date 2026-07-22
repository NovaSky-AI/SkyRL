# Weight Sync

Training-to-inference weight transfer. Runs after every training step (or on the configured interval) to push updated policy weights from training workers (FSDP/Megatron) into the vLLM inference engines.

## Architecture

Two-sided protocol with sender (training) / receiver (inference):

```
skyrl/backends/skyrl_train/weight_sync/
├── base.py                 # WeightUpdateRequest, LoraLoadRequest, WeightChunk
├── transfer_strategy.py    # WeightSyncInitInfo / Sender / Strategy ABCs (sender-side only; receive is vLLM-native)
├── broadcast_strategy.py   # NCCL broadcast (non-colocated)
├── cuda_ipc_strategy.py    # CUDA IPC (colocated)
├── weight_extractor.py     # Sharded-param -> dense tensor extraction
└── weight_extractor_utils.py
```

vLLM worker-extension class (loaded via `--worker-extension-cls`):

- `skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py` — `NewInferenceWorkerWrap`. Three-phase chunked lifecycle.

The weight sync implementation relies on the native vLLM weight sync APIs - `WeightTransferEngine` abstractions as well as native RPC endpoints for weight updates.

## Transfer Strategies

- **Broadcast** (`BroadcastTransferStrategy`): NCCL collective. Used for **non-colocated** setups. Training and inference are on different GPUs; weights cross the wire over a dedicated process group.
- **CUDA IPC** (`CudaIpcTransferStrategy`): Per-chunk packed buffer + one IPC handle per rank. Used for **colocated** setups (`colocate_all=true`). Both sides live on the same GPU; the receiver maps the sender's CUDA allocation directly.

Strategy choice is decided by the sender (`get_transfer_strategy_cls`). The init info is expanded per server via `for_servers()` / `to_api_payload()` and pushed to the servers through the HTTP control plane (`init_weight_update_communicator` → vLLM's native `/init_weight_transfer_engine`); the receive side is vLLM's native weight-transfer engine, driven by `NewInferenceWorkerWrap`.

## Lifecycle (`NewInferenceWorkerWrap`)
1. `start_weight_update(is_checkpoint_format=True)` — initializes layerwise reload (moves layers to meta device, wraps loaders).
2. `update_weights_chunk(update_info)` — called repeatedly. Unpacks the SkyRL packed CUDA-IPC payload, slices the contiguous buffer per param, calls `model.load_weights(weights=...)` under `set_current_vllm_config`.
3. `finish_weight_update()` — runs `finalize_layerwise_reload` (quantization repacking, attention weight postprocessing).

## Sleep during non-colocated weight sync

Non-colocated normally keeps the engine fully awake and does `pause_generation → broadcast → resume_generation`. Two opt-in `generator.inference_engine.*` flags let the engine free its KV-cache memory *during* the sync so `gpu_memory_utilization` can be pushed higher (no need to keep KV cache resident alongside the weight-transfer scratch buffers). Both require `enable_sleep_mode` on the engine, which `inference_servers/utils.py` turns on when `sleep_engines_during_weight_sync` is set.

- **`sleep_engines_during_weight_sync`** (sync trainer): `sleep() → wake_up(["weights"]) → broadcast → wake_up(["kv_cache"])` — the same three-phase pattern colocated mode uses. The sleep discards the KV cache and aborts in-flight requests, which is correct for the synchronous trainer (generation is complete at sync time). Requires non-colocated + non-LoRA.
- **`preserve_inflight_requests_during_weight_sync`** (fully-async trainer): a **KV-preserving suspend/resume**. `pause_generation` (KEEP, freezes in-flight requests with KV intact) → offload weights+KV to CPU and free GPU → wake weights → broadcast → restore KV → `resume_generation`. Frozen requests resume with no abort and no prefill recompute, at the cost of a GPU↔CPU copy of the whole KV pool each sync. Requires `sleep_engines_during_weight_sync=True`, `fully_async.enabled`, and `clear_kv_cache_on_weight_sync=False`.

The preserve path is driven entirely from SkyRL — no vLLM patch. It deliberately avoids the `/sleep`+`/wake_up` HTTP endpoints (which route through `EngineCore.sleep`, force-clearing the prefix cache and preempting every running request at level ≥ 1). Instead it drives the per-worker `CuMemAllocator` directly via two `NewInferenceWorkerWrap` methods invoked over `/collective_rpc`:

- `skyrl_sleep_preserve_kv` — `allocator.sleep(offload_tags=("weights","kv_cache"))`: offloads both pools to CPU (model buffers live in the `weights` pool, so they survive too) and frees the GPU memory. The scheduler is untouched, so the KEEP-paused requests stay frozen with valid block tables.
- `skyrl_wake_preserved(tags)` — `allocator.wake_up(tags)` remaps to the **same virtual addresses** and copies CPU→GPU, so block tables remain valid; re-inits fp8 KV scales on the `kv_cache` wake. Does not resume the scheduler (the client does that via `/resume`).

Orchestrated in `WorkerDispatch.save_weights_for_sampler` (non-colocated single-tenant branch). Validated in `validate_inference_engine_cfg`. vLLM-version coupled (mirrors `GPUWorker.sleep`/`wake_up` and the `CuMemAllocator` API) — re-verify on vLLM bumps via the GPU weight-sync test.

## Convention: vLLM imports

`vllm` is a Linux-only optional dep. Import it **lazily inside methods**, not at module top. Match the existing pattern in `new_inference_worker_wrap.py`.

## Tests

```bash
# CPU — chunk packing, transfer strategy unit tests
uv run --extra dev pytest tests/backends/skyrl_train/weight_sync/ -v

# GPU — end-to-end weight sync (NCCL + CUDA IPC paths, TP=1 and TP=2)
uv run --isolated --extra dev --extra fsdp \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v
```

The CPU tests do **not** import `NewInferenceWorkerWrap`. Any change to the worker-extension class must be exercised by the GPU test above.

## When to touch what

| Change | Run |
|--------|-----|
| `WeightChunk` packing / size accounting | `tests/backends/skyrl_train/weight_sync/test_weight_chunk.py` |
| Broadcast or CUDA IPC sender | `test_transfer_strategies.py` (CPU) **and** GPU `test_weight_sync.py` |
| `NewInferenceWorkerWrap` | GPU `test_weight_sync.py` (CPU tests will not catch regressions) |

## vLLM version coupling

`vllm` is pinned in `pyproject.toml`. Weight-sync code paths are tightly coupled to vLLM internals (`model_runner.load_weights`, `initialize_layerwise_reload`, `SKIP_TENSORS`). When bumping the pin, re-verify the GPU weight-sync tests.

## Gotchas

- NemotronH / Mamba: vLLM's layerwise reload corrupts `conv1d.weight` via shared-storage view buffers. Workaround at the top of `new_inference_worker_wrap.py` adds `"conv_weights"` to `SKIP_TENSORS` at import time. Remove pending vLLM PR #42481 (vLLM 0.21.0).
- After `update_weights_chunk` runs, call `torch.accelerator.synchronize()` before returning so the sender doesn't drop its packed buffer mid-copy on the next barrier.
