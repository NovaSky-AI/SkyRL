# Weight Sync

Training-to-inference weight transfer. Runs after every training step (or on the configured interval) to push updated policy weights from training workers (FSDP/Megatron) into the vLLM inference engines.

## Architecture

Two-sided protocol with sender (training) / receiver (inference):

```
skyrl/backends/skyrl_train/weight_sync/
├── base.py                 # WeightUpdateRequest, LoraLoadRequest, WeightChunk
├── transfer_strategy.py    # WeightSyncInitInfo / Sender / Strategy ABCs (sender-side only; receive is vLLM-native)
├── broadcast_strategy.py   # NCCL broadcast (non-colocated)
├── nccl_trainer_send.py    # vendored: vLLM 0.26's NCCL trainer-send statics
├── cuda_ipc_strategy.py    # CUDA IPC (colocated)
├── delta_strategy.py       # Checkpoint-delta sender + strategy (disk / gs:// / s3://)
├── delta_checkpoint.py     # DeltaCheckpointPublisher, LocalCheckpointStore, manifest + XOR payloads
├── delta_engine.py         # DeltaWeightTransferEngine (receive side, runs in the vLLM worker)
├── delta_payload.py        # zstd compress/decompress + uint8 tensor <-> bytes helpers
├── weight_extractor.py     # Sharded-param -> dense tensor extraction
├── weight_extractor_utils.py
└── sharded_rdt/            # the sharded_rdt (NIXL pull) backend; __init__ is import-free
    ├── sharded_rdt_strategy.py # sharded_rdt as a WeightTransferStrategy (thin adapter)
    ├── rdt_send.py             # trainer-side driver: WeightSource impls + RdtWeightSyncSender
    ├── rdt_control_plane.py    # SyncRdtControlPlaneClient (blocking HTTP to /collective_rpc)
    ├── rdt_vllm_register.py    # registers the sharded_rdt engine into vLLM's factory
    ├── rdt_libfabric_shim.py   # LIBFABRIC provider shim for NIXL
    ├── sharded_rdt_base.py     # vendored: trainer-side ABCs (WeightSource, layerwise_groups)
    ├── sharded_rdt_trainer.py  # vendored: trainer engine + the _RDTProducerServer sidecar
    ├── sharded_rdt_engine.py   # vendored: consumer engine (runs in the vLLM worker)
    ├── sharded_rdt_common.py   # vendored: RdtRouter, op-chain allowlist, buffer sizing
    └── sharded_rdt_fake.py     # vendored: FakeRDTTensor placeholders for the bake
```

The subpackage's `__init__.py` deliberately imports nothing: `sharded_rdt_engine` and
`sharded_rdt_trainer` import `vllm` at module scope, so re-exporting from it would pull
vllm into every `weight_sync` import and break the CPU CI job that runs without the wheel.
Public names (`ShardedRdtTransferStrategy` and friends) are re-exported from
`weight_sync/__init__.py`, which only reaches the vllm-free `sharded_rdt_strategy`.

vLLM worker-extension class (loaded via `--worker-extension-cls`):

- `skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py` — `NewInferenceWorkerWrap`. Three-phase chunked lifecycle.

The weight sync implementation relies on the native vLLM weight sync APIs - `WeightTransferEngine` abstractions as well as native RPC endpoints for weight updates.

## Transfer Strategies

- **Broadcast** (`BroadcastTransferStrategy`): NCCL collective. Used for **non-colocated** setups. Training and inference are on different GPUs; weights cross the wire over a dedicated process group.
- **CUDA IPC** (`CudaIpcTransferStrategy`): Per-chunk packed buffer + one IPC handle per rank. Used for **colocated** setups (`colocate_all=true`). Both sides live on the same GPU; the receiver maps the sender's CUDA allocation directly.
- **Delta** (`DeltaTransferStrategy`): Weights travel as compressed XOR deltas against the base checkpoint, through a shared filesystem or object store instead of the network fabric. Selected with `generator.inference_engine.weight_sync_backend=delta`; intended for **non-colocated** setups where the two sides are not NCCL-reachable (separate clusters, PD-disaggregated serving). Not supported with LoRA, nor with serialized FP8 weight sync (`fp8_weight_sync_mode=blockwise`) whose marker names and scale tensors the delta checkpoint format cannot represent; `validate_cfg` rejects both.
- **Sharded RDT** (`sharded_rdt`): the inference workers **pull** the slices they consume from
  the trainer ranks over NIXL/RDMA, instead of the trainer pushing every tensor to every
  worker. Selected with `generator.inference_engine.weight_sync_backend=sharded_rdt`;
  non-colocated only (`placement.colocate_all=false`), Megatron or FSDP, and it forces
  `distributed_executor_backend=ray` because the workers dial named trainer actors. Not
  supported with serialized FP8 weight sync (`fp8_weight_sync_mode=blockwise`): the weight
  sources publish whole bridge tensors, never payload+scale pairs, so
  `validate_inference_engine_cfg` rejects the combination. See the dedicated section below
  for the capabilities it declares.

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

## Sharded RDT (`sharded_rdt`)

`sharded_rdt` is selected through `get_transfer_strategy_cls` like every other backend
(`ShardedRdtTransferStrategy`, in `weight_sync/sharded_rdt/sharded_rdt_strategy.py`), but underneath it
is vLLM's *trainer-send* model — a `WeightSource`, a `TrainerWeightTransferEngine` and a
`VLLMWeightSyncClient` — where the engine owns the whole round trip and the workers pull.
The strategy is a thin adapter over `RdtWeightSyncSender` (`weight_sync/sharded_rdt/rdt_send.py`),
which is where the real work lives.

Where the pull model does not fit the push backends' shape, declared as capabilities
so the workers hold no backend conditional:

| flag | why |
|---|---|
| `sender_initializes_receivers = True` | `trainer_init` opens the inference side itself (the bake needs the source metadata and must run under `set_current_vllm_config`), so worker rank 0 must **not** also call `init_weight_update_communicator`. |
| `force_disable_expandable_segments = True` | The sidecar shares gathered tensors over CUDA IPC on every run, not only under colocation. |
| `empty_cache_after_send = False` | Publish buffers are reused by the next training step; scrubbing them back to CUDA costs 0.25-0.53s/rank at 235B. The worker still empties under `colocate_all`. |

The sender overrides `WeightTransferSender.send` rather than implementing `send_chunks`:
there is no chunk stream to push, and the base `send()` would call
`get_weight_metadata`, which on the Megatron extractor is a whole-model
`export_hf_weights` pass — the exact gather the RDT weight source exists to avoid.
`send_chunks` raises here.

This is still intermediate: when the pinned vLLM ships trainer-send for NCCL/IPC too,
those backends collapse into this shape and the strategy layer goes away — the adapter is
what disappears, not `rdt_send.py`.

Three kinds of process: the **trainer ranks** (each builds a `WeightSource` and a
`ShardedRDTTrainerWeightTransferEngine`), one **producer sidecar** Ray actor per trainer
rank (pinned to that rank's GPU, sharing its memory over CUDA IPC — this is the NIXL serve
surface), and the **consumers** (a `ShardedRDTWeightTransferEngine` inside each vLLM
worker).

Init is **eager**, in `init_weight_sync_state`, not deferred to the first send: rank 0
would otherwise block in the inference-side init RPC while the other ranks spin in gather
collectives, and the sidecars could not finish NIXL agent creation because libfabric's
CUDA probe blocks behind the spinning kernels. Consumers **bake** a static pull plan at
init by driving `model.load_weights` against `FakeRDTTensor` placeholders, so per-sync
`update_info` is empty.

Per sync the trainer walks its owned gather groups (`layerwise_groups`, one group per
decoder layer); each group is gathered, packed into a CUDA-IPC ring slot and published to
the sidecar. Consumers pull packed chunks into a ring of receive buffers, scatter them into
the vLLM params, and signal `free_group` at every owner of the group. The producer counts
signals against the consumer total and releases the group, which is the trainer's credit to
gather the next one — so the loop self-paces to the consumers' pull rate with at most
`gather_lookahead + 1` groups resident.

Knobs are `SKYRL_RDT_*` env vars, forwarded to every Ray worker by
`prepare_runtime_environment`: `SKYRL_RDT_LOOKAHEAD` (gather credit depth, default 1),
`SKYRL_RDT_NUM_BUFFERS` (ring depth K, default 2), `SKYRL_RDT_STACKED_EXPERTS`,
`SKYRL_RDT_EXPORT_RING`, `SKYRL_RDT_GC_FREEZE`, `SKYRL_RDT_SHARE_SLOTS`,
`SKYRL_RDT_STALL_TIMEOUT_S`, `SKYRL_RDT_BUFFER_PRESIZE_GB`, and
`SKYRL_RDT_VERIFY_STACKED=1` (a one-off numeric check of the stacked source's
expert tensors against the bridge's per-expert export, on the first iteration).
The
sidecar does **not** inherit them (it is a Ray actor inheriting the raylet's environment),
which is why the ring / lookahead / timeout knobs ride the init info instead.

RDT receive buffers live **outside** `gpu_memory_utilization`: K x the largest chunk, per
rank. `use_expandable_segments` must stay off on the engine (NIXL cannot register
VMM-backed allocations) and is force-disabled around the sync on the trainer.

## Slot sharing across replicas (`sharded_rdt`)

With several inference deployments, consumers whose ids differ by a multiple of
`workers_per_replica` (= `num_consumers // num_replicas`) are the same worker of
different deployments: same parallel config, same baked plan, same chunk sequence,
byte-identical pack layout. They are served out of ONE registered serve slot on
each producer instead of one ring each, because NIXL reads are one-sided and
non-destructive. Two halves, both required:

```
[routing]  RdtRouter.producer_for carves the owner-set block over ONE deployment
             (consumer_id % workers_per_replica), so worker w of every deployment
             resolves the SAME producer -> the R copies meet somewhere to merge.
             One deployment => width is the fleet => the historical rule exactly.
[sender]   workers_per_replica -> ShardedRDTTrainerInitInfo -> the sidecar
             SKYRL_RDT_SHARE_SLOTS=0 sets it to 0, i.e. sharing off
[consumer] _preregister_at_init: reserve_serve_buffer(cid, max_bytes, plan_digest)
             one ring per GROUP, and the digest is where mismatched deployments fail
[producer] rdt_produce_weights_batched: the group's sharers rendezvous per chunk
             (keyed by `seq`), the LAST arrival packs, all return that blob
             and the slot is `seq % ring_depth`
```

The serve slot is chosen from the consumer's ISSUE index, never from a per-call
counter on the producer. The pipeline drains pull i before issuing i+K, so
`seq % K` is provably free; execution order is not, because Ray may start a
consumer's K concurrent produce calls in any order, and the slot of a pull that
is still being read then gets repacked. That bug was live and cost 2x on the
logprob gap.

Pulls and free signals still carry the fleet-GLOBAL consumer id; only the block
carve uses the intra-deployment index. The producer needs the global id to tell
the sharers of a slot apart and count their arrivals separately.

What makes the slot safe is the same inference the unshared path makes: a consumer
issues the pull that reuses its own ring slot only after draining the pull K
earlier, so a sharer's arrival proves it finished reading whatever it saw K
arrivals ago. A generation's slot returns to the group's free list only once every
sharer that arrived at it is K arrivals past it, so at most K generations hold a
slot and the K slots always suffice.

One deployment (`workers_per_replica == num_consumers`) makes every group a
singleton, which is the previous serve path exactly — same rotation over K slots,
and the request signature is not even computed. `begin_sync` takes the consumer
IDS alongside the count, because a rendezvous is a specific set of ids where the
free barrier only needs a total.

Producer-side cost this addresses (235B, 4 nodes, K=2, largest chunk 1.16 GiB):
5.0 GiB of serve rings per trainer GPU at one deployment, and without these
changes 14 at two, 28 at four, 54 at eight. Both together hold it at 5.0 for any
count, with the pack work flat too. Sharing alone would be 12/22/40 -- at R>1 most
of a producer's ring count comes from distinct worker indices, not from replicas
of one, which is what the overlay fixes.

`_RDTProducerServer` carries a **stall watchdog** (`stall_timeout_s`, default 300s,
`SKYRL_RDT_STALL_TIMEOUT_S`). Without it, a consumer that stops pulling mid-sync never
sends its `free_group` signals, the three waits block forever, and every trainer rank
wedges in NCCL with no exception anywhere. On no publish/serve/free progress for the
timeout, the producer fires the `set_gather_error` channel and the run fails with a real
error. It does not recover; it makes the failure diagnosable. The slot-sharing rendezvous
waits on the same channel, so a sharer that never arrives fails the same way.

The `sharded_rdt_*.py` files are vendored from the vLLM PR: github.com/vllm-project/vllm/pull/43375; see each file's header for the removal plan once the pinned vLLM ships the trainer-side ABCs natively.

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
# CPU — chunk packing, transfer strategies, and the sharded_rdt pull plan /
# producer sidecar / weight source / control plane
uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/ -v

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

`vllm` is pinned in `pyproject.toml` (currently `0.28.0`, in both the `fsdp` and `megatron` extras). Weight-sync code paths are tightly coupled to vLLM internals (`model_runner.load_weights`, `initialize_layerwise_reload`, `SKIP_LOAD_TENSORS`). When bumping the pin, re-verify the GPU weight-sync tests.

### `packed` is an init-time wire param (0.28.0+)

Through 0.26.0 the NCCL sender put `packed` on the **per-round** update info. 0.28.0 moved it
to the **init** info: `NCCLWeightTransferInitInfo.packed` is recorded once during
`/init_weight_transfer_engine`, `receive_weights` reads `self.packed`, and
`NCCLWeightTransferUpdateInfo` now rejects the key outright.

So `BroadcastInitInfo.packed` must be sent by `to_api_payload()` and must agree with the
`packed` handed to `nccl_trainer_send_weights`. Its default is `False` on the vLLM side, so
forgetting it does not raise — the trainer packs while the worker unpacks one-by-one, and the
transfer hangs inside NCCL. Only the GPU weight-sync test catches this.

### Vendored trainer-send (`weight_sync/nccl_trainer_send.py`)

0.28.0 deleted `NCCLWeightTransferEngine.trainer_send_weights` and the engine's `trainer_init`
staticmethod, replacing them with a trainer-side engine abstraction
(`NCCLTrainerWeightTransferEngine`, via `WeightTransferTrainerFactory`). SkyRL still drives the
send itself, so `nccl_trainer_send.py` keeps the 0.26.0 behaviour: it wraps
`packed_nccl_broadcast_producer` and re-exports `nccl_common.trainer_init`, both of which
survive in 0.28.0 unchanged. Delete that module when SkyRL's sender layer migrates onto vLLM's
trainer-send engines.

The `sharded_rdt` backend is unaffected: it vendors its own trainer-side ABCs in
`sharded_rdt_base.py`, which already match 0.28.0's `Generic[TTrainerInitInfo]` shape.

### DeepGEMM is unavailable under the torch override

vLLM 0.28.0's metadata pins `torch==2.13.0`; we override torch to `2.11.0` because the CUDA
extension wheels we build against (flash-attn, causal-conv1d, mamba-ssm, transformer-engine)
have no 2.13 builds. vLLM's main extensions are stable-libtorch-ABI and load fine, but
`vllm/third_party/deep_gemm/_C` is a version-specific `cpython-312` build linked against torch
2.13's `c10` and fails with an undefined-symbol `ImportError`. vLLM catches this and logs
`Module vllm.third_party.deep_gemm was found but failed to import`, then falls back, so the
engine runs — but the DeepGEMM fused-MoE and sparse-attention-indexer paths are gone. Revisit
when those wheels publish torch 2.13 builds.

## Gotchas

- After `update_weights_chunk` runs, call `torch.accelerator.synchronize()` before returning so the sender doesn't drop its packed buffer mid-copy on the next barrier.
- Delta: `DeltaWeightTransferEngine` is registered as an **import side effect** of `new_inference_worker_wrap.py`, which is the module vLLM loads via `--worker-extension-cls`. Registering anywhere else (e.g. while building CLI args in the driver) is a no-op — it has to happen in the process that owns the engine.
- Delta: the receive-side `delta checkpoint fetch:` / `receive reload-only:` log lines are emitted inside the nested vLLM worker process and do **not** reach the driver log, even with `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`. Find them with `grep -rhE "delta checkpoint (fetch|receive)" /tmp/ray/session_latest/logs/`. Filter by mtime — that directory accumulates lines from earlier runs.
- Delta: `_safe_path_name` appends a digest of the full value because sibling delta URIs differ only in their trailing `delta-<version>`; a plain length cap collapses every version onto one cache directory. Don't "simplify" it back to truncation.
- Delta: `s3://` needs the `s5cmd` CLI, which the `aws` extra installs into the run's venv (`--extra aws`). `gs://` needs the `gcloud` CLI as a *system* binary — the `gcp` extra only provides a Python library and will not satisfy it.
