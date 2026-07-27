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
├── delta_strategy.py       # Checkpoint-delta sender + strategy (disk / gs:// / s3://)
├── delta_checkpoint.py     # DeltaCheckpointPublisher, LocalCheckpointStore, manifest + XOR payloads
├── delta_engine.py         # DeltaWeightTransferEngine (receive side, runs in the vLLM worker)
├── delta_payload.py        # zstd compress/decompress + uint8 tensor <-> bytes helpers
├── weight_extractor.py     # Sharded-param -> dense tensor extraction
└── weight_extractor_utils.py
```

vLLM worker-extension class (loaded via `--worker-extension-cls`):

- `skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py` — `NewInferenceWorkerWrap`. Three-phase chunked lifecycle.

The weight sync implementation relies on the native vLLM weight sync APIs - `WeightTransferEngine` abstractions as well as native RPC endpoints for weight updates.

## Transfer Strategies

- **Broadcast** (`BroadcastTransferStrategy`): NCCL collective. Used for **non-colocated** setups. Training and inference are on different GPUs; weights cross the wire over a dedicated process group.
- **CUDA IPC** (`CudaIpcTransferStrategy`): Per-chunk packed buffer + one IPC handle per rank. Used for **colocated** setups (`colocate_all=true`). Both sides live on the same GPU; the receiver maps the sender's CUDA allocation directly.
- **Delta** (`DeltaTransferStrategy`): Weights travel as compressed XOR deltas against the base checkpoint, through a shared filesystem or object store instead of the network fabric. Selected with `generator.inference_engine.weight_sync_backend=delta`; intended for **non-colocated** setups where the two sides are not NCCL-reachable (separate clusters, PD-disaggregated serving). Not supported with LoRA (`validate_cfg` rejects it).

Strategy choice is decided by the sender (`get_transfer_strategy_cls`). The init info is expanded per server via `for_servers()` / `to_api_payload()` and pushed to the servers through the HTTP control plane (`init_weight_update_communicator` → vLLM's native `/init_weight_transfer_engine`); the receive side is vLLM's native weight-transfer engine, driven by `NewInferenceWorkerWrap`.

## Delta backend

Unlike the other two strategies, delta sync does not push tensors to the receiver at all — it
publishes bytes to `sync_dir` and the receiver pulls them.

**Publish (trainer, rank 0).** `DeltaCheckpointPublisher` keeps a CPU `uint8` snapshot of the
full model, XORs it against the new weights to get a per-tensor patch, zstd-compresses each
patch, and writes them as safetensors payload files plus a `manifest.json` under
`<sync_dir>/delta-<version:08d>/`. Unchanged tensors are omitted. Payload files roll over at
`max_file_size_in_gb`.

**Fetch (receiver, before pause).** A control-plane operation the other strategies do not have:
`RemoteInferenceClient.fetch_weights` → `/fetch_weights` on every server, driven by
`DeltaWeightTransferSender._apply_receiver_update` *before* generation is paused, so the
download and patch-apply happen off the critical path. `LocalCheckpointStore` maintains a
mutable copy of the checkpoint under `local_checkpoint_dir/weights`, replaying every delta from
its current version up to the target (so a late-joining engine catches up), and applies each
patch by XOR-ing directly into the mmap'd safetensors files.

**Reload.** Only then is generation paused and the local checkpoint reloaded into vLLM via
`iter_tensors`. Note the delta shrinks the *transfer*, not the reload — the whole checkpoint is
re-read even for an empty delta.

`DeltaWeightSyncConfig.__post_init__` derives `local_checkpoint_dir` and `publish_staging_dir`
from `sync_dir` when unset, so consuming classes never invent their own defaults.

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

# GPU — end-to-end delta sync (sparse perturbation, fsdp and megatron)
uv run --isolated --extra dev --extra fsdp \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/test_delta_weight_sync_e2e.py -m "not megatron" -v
uv run --isolated --extra dev --extra megatron \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/test_delta_weight_sync_e2e.py -m megatron -v
```

The CPU tests do **not** import `NewInferenceWorkerWrap`. Any change to the worker-extension class must be exercised by the GPU test above.

The Megatron variants in `test_prefix_cache_reset.py` **skip silently** without megatron-core, so
run that file under `--extra megatron` too before trusting a green CPU run.

## When to touch what

| Change | Run |
|--------|-----|
| `WeightChunk` packing / size accounting | `tests/backends/skyrl_train/weight_sync/test_weight_chunk.py` |
| Broadcast or CUDA IPC sender | `test_transfer_strategies.py` (CPU) **and** GPU `test_weight_sync.py` |
| `NewInferenceWorkerWrap` | GPU `test_weight_sync.py` (CPU tests will not catch regressions) |
| Delta publish / manifest / payload format | `test_delta_checkpoint.py` **and** GPU `test_delta_weight_sync_e2e.py` |
| `LocalCheckpointStore` (fetch, replay, apply, cache keys) | `test_delta_checkpoint.py` |
| `DeltaWeightTransferEngine` | GPU `test_delta_weight_sync_e2e.py` only — it runs inside the vLLM worker |
| Who pauses / resets the prefix cache | `test_prefix_cache_reset.py` **and** `distributed/test_worker_dispatch.py` |
| `DeltaWeightSyncConfig` defaults or validation | `tests/train/test_config.py` |

## vLLM version coupling

`vllm` is pinned in `pyproject.toml`. Weight-sync code paths are tightly coupled to vLLM internals (`model_runner.load_weights`, `initialize_layerwise_reload`, `SKIP_TENSORS`). When bumping the pin, re-verify the GPU weight-sync tests.

## Gotchas

- NemotronH / Mamba: vLLM's layerwise reload corrupts `conv1d.weight` via shared-storage view buffers. Workaround at the top of `new_inference_worker_wrap.py` adds `"conv_weights"` to `SKIP_TENSORS` at import time. Remove pending vLLM PR #42481 (vLLM 0.21.0).
- After `update_weights_chunk` runs, call `torch.accelerator.synchronize()` before returning so the sender doesn't drop its packed buffer mid-copy on the next barrier.
- Delta: `DeltaWeightTransferEngine` is registered as an **import side effect** of `new_inference_worker_wrap.py`, which is the module vLLM loads via `--worker-extension-cls`. Registering anywhere else (e.g. while building CLI args in the driver) is a no-op — it has to happen in the process that owns the engine.
- Delta: the receive-side `delta checkpoint fetch:` / `receive reload-only:` log lines are emitted inside the nested vLLM worker process and do **not** reach the driver log, even with `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`. Find them with `grep -rhE "delta checkpoint (fetch|receive)" /tmp/ray/session_latest/logs/`. Filter by mtime — that directory accumulates lines from earlier runs.
- Delta: `_safe_path_name` appends a digest of the full value because sibling delta URIs differ only in their trailing `delta-<version>`; a plain length cap collapses every version onto one cache directory. Don't "simplify" it back to truncation.
- Delta: `s3://` needs the `s5cmd` CLI, which the `aws` extra installs into the run's venv (`--extra aws`). `gs://` needs the `gcloud` CLI as a *system* binary — the `gcp` extra only provides a Python library and will not satisfy it.
