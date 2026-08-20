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

## Degraded sync (engine fault tolerance)

When `generator.inference_engine.fault_tolerance.enabled=true` and an inference server
has died, the sync serves only the survivors. This works for `sharded_rdt` and nothing
else: NCCL and cuda_ipc broadcast over a communicator fixed at provision, so losing a
member hangs the broadcast rather than degrading it. `_live_server_urls` in
`worker_dispatch.py` is the runtime backstop that refuses a partial broadcast.

The mechanism is small because freeing is a per-group barrier: each consumer signals
`free_group(gi)` at every owner of the group exactly once per sync, and each producer
counts signals against one uniform target — the live consumer count passed at
`begin_sync`.

```
[driver]   save_weights_for_sampler: live = client.active_server_urls (None if whole)
             -> _broadcast_to_inference_engines(..., live_server_urls=live)   # one dispatch,
                                                                              # so all ranks agree
[worker]   RdtWeightSyncSender.send(extractor, live_server_urls)
             live_cids = urls -> PROVISIONED indices -> replica ranks -> consumer ids
             control_plane.set_live(live)          # start/update/finish only; init keeps the
                                                   # provisioned list (replica_rank is a position)
[producer] send_weights(live_consumer_ids) -> begin_sync(len(live)): every group's
             free barrier counts to the live total instead of the provisioned one
[consumer] unchanged — a consumer's plan depends only on its own consumer_id
```

Invariants worth not breaking: provisioned geometry (`num_consumers`, the router,
ownership, `served_names`) is frozen for the run and liveness is only ever a lower
barrier *target* over it; every rank must get the same live set; and the gather loop
still iterates every owned group on every rank, because the non-expert gather is a
collective.

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
[producer] rdt_produce_weights_batched: the group's live sharers rendezvous per
             chunk (keyed by `seq`), the LAST arrival packs, all return that blob
             and the slot is `seq % ring_depth`
```

The serve slot is chosen from the consumer's ISSUE index, never from a per-call
counter on the producer. The pipeline drains pull i before issuing i+K, so
`seq % K` is provably free; execution order is not, because Ray may start a
consumer's K concurrent produce calls in any order, and the slot of a pull that
is still being read then gets repacked. That bug was live and cost 2x on the
logprob gap -- see `~/default/RDT_WEIGHT_SYNC.md` §8.

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
and the request signature is not even computed. `begin_sync` now takes the live
consumer IDS alongside the count, because a rendezvous is a specific set of ids
where the free barrier only needs a total; a degraded sync narrows the groups.

Producer-side cost this addresses (235B, 4 nodes, K=2, largest chunk 1.16 GiB):
5.0 GiB of serve rings per trainer GPU at one deployment, and without these
changes 14 at two, 28 at four, 54 at eight. Both together hold it at 5.0 for any
count, with the pack work flat too. Sharing alone would be 12/22/40 -- at R>1 most
of a producer's ring count comes from distinct worker indices, not from replicas
of one, which is what the overlay fixes. Counters: `shared_serves`,
`share_wait_seconds`, `cfg_share_width`, `cfg_serve_rings` in
`get_produce_timing()`.

`_RDTProducerServer` also carries a **stall watchdog** (`stall_timeout_s`, default 300s).
Detection is generation-driven and no generation is in flight during a sync, so a death
inside the sync window has no detector — the dead consumer never sends its `free_group`
signals, the three waits block forever, and every trainer rank wedges in NCCL with no
exception anywhere. On no publish/serve/free progress for the timeout, the producer fires
the existing `set_gather_error` channel and the run fails with a real error. Part 1 does
not recover from this; it only makes it diagnosable.

Vendored files (`sharded_rdt_*.py`) mirror `~/default/vllm-rdt-weight-sync`; keep both in
sync. Design doc: `~/default/rdt_fault_tolerance.md`.

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
