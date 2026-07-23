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

## KV offload during non-colocated weight sync

Non-colocated normally keeps the engine fully awake and does `pause_generation → broadcast → resume_generation`. The opt-in `generator.inference_engine.offload_kv_for_weight_sync` flag sleeps the engine (freeing the KV cache from GPU) *during* the sync so `gpu_memory_utilization` can be pushed higher (no need to keep KV cache resident alongside the weight-transfer scratch buffers). It turns on `enable_sleep_mode` (via `inference_servers/utils.py`). Requires non-colocated and non-LoRA. Orchestrated in `WorkerDispatch.save_weights_for_sampler`; the flow depends on the trainer:

- **Synchronous trainer** (`fully_async.enabled=false`): generation is complete at sync time, so there are no in-flight requests. A plain `sleep() → wake_up(["weights"]) → broadcast → wake_up(["kv_cache"])` (the same three-phase pattern colocated uses) is enough — the standard `/sleep`+`/wake_up` endpoints discard the KV cache and free the memory.
- **Fully-async trainer** (`fully_async.enabled=true`): generation overlaps the sync, so `pause_generation` (KEEP) freezes in-flight requests, then the allocator is driven directly (see below) so the scheduler is **not** resumed on the weights wake. The KV cache is offloaded to CPU and restored so frozen requests resume with no abort or prefill recompute — **unless** `clear_kv_cache_on_weight_sync=true`, in which case the broadcast resets the prefix cache anyway, so the KV is discarded (skipping the CPU copy) rather than offloaded.

The fully-async path is driven entirely from SkyRL — no vLLM patch. It deliberately avoids the `/sleep`+`/wake_up` HTTP endpoints (which route through `EngineCore.sleep`, force-clearing the prefix cache and preempting every running request at level ≥ 1). Instead it drives the per-worker `CuMemAllocator` directly via two `NewInferenceWorkerWrap` methods invoked over `/collective_rpc`:

- `skyrl_sleep_for_weight_sync(offload_kv)` — `allocator.sleep(offload_tags=("kv_cache",) if offload_kv else ())`: **discards** the weights pool (the broadcast overwrites every parameter on wake) and either offloads the KV cache to CPU or discards it. Model buffers live in the weights pool and are NOT covered by the parameter broadcast (e.g. non-persistent rotary `inv_freq`), so they are saved to CPU here and restored on wake — mirroring `GPUWorker.sleep(level=2)`. All GPU memory is freed regardless. The scheduler is untouched, so KEEP-paused requests stay frozen with valid block tables.
- `skyrl_wake_for_weight_sync(tags)` — `torch.cuda.empty_cache()` (release the broadcast's transient buffers so cumem can remap the KV pool) then `allocator.wake_up(tags)`, which remaps to the **same virtual addresses** and copies CPU→GPU so block tables remain valid. On the `weights` wake it restores the saved buffers; on the `kv_cache` wake it re-inits fp8 KV scales. Does not resume the scheduler (the client does that via `/resume`).

Validated in `validate_inference_engine_cfg`. vLLM-version coupled (mirrors `GPUWorker.sleep`/`wake_up` and the `CuMemAllocator` API) — re-verify on vLLM bumps via the GPU weight-sync test.

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
