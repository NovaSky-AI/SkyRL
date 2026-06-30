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

## Transfer Strategies

- **Broadcast** (`BroadcastTransferStrategy`): NCCL collective. Used for **non-colocated** setups. Training and inference are on different GPUs; weights cross the wire over a dedicated process group.
- **CUDA IPC** (`CudaIpcTransferStrategy`): Per-chunk packed buffer + one IPC handle per rank. Used for **colocated** setups (`colocate_all=true`). Both sides live on the same GPU; the receiver maps the sender's CUDA allocation directly.

Strategy choice is decided by the sender (`get_transfer_strategy_cls`). The init info is expanded per server via `for_servers()` / `to_api_payload()` and pushed to the servers through the HTTP control plane (`init_weight_update_communicator` → vLLM's native `/init_weight_transfer_engine`); the receive side is vLLM's native weight-transfer engine, driven by `NewInferenceWorkerWrap`.

## Lifecycle (`NewInferenceWorkerWrap`)
1. `start_weight_update(is_checkpoint_format=True)` — initializes layerwise reload (moves layers to meta device, wraps loaders).
2. `update_weights_chunk(update_info)` — called repeatedly. Unpacks the SkyRL packed CUDA-IPC payload, slices the contiguous buffer per param, calls `model.load_weights(weights=...)` under `set_current_vllm_config`.
3. `finish_weight_update()` — runs `finalize_layerwise_reload` (quantization repacking, attention weight postprocessing).

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
