# Bounded-memory colocated weight sync

> **Scope:** `trainer.placement.colocate_all=true`, Megatron trainer, CUDA-IPC weight sync
> (`skyrl_ipc`), full fine-tuning. vLLM pin: `0.33.0`.
> **Read first:** `.agents/docs/weight_sync.md`, `.agents/docs/backends/megatron.md`.
> **Results:** [`colocated_weight_sync_results.md`](colocated_weight_sync_results.md).

Every option described here defaults off. With all of them off, colocated weight sync runs
exactly as before.

## 1. Problem

Colocated sync puts the trainer and the vLLM engine on the same GPUs. Before a sync the
engine sleeps, and its weights are discarded (level 2) or moved to CPU (level 1). The sync
then runs these steps (`WorkerDispatch.save_weights_for_sampler`):

1. Reload the policy's BF16 params onto the GPUs (`_ensure_on_gpu("policy", need_model=True)`).
2. `wake_up(tags=["weights"])`: map all of the engine's weight memory again.
3. `send_weights()`: every trainer rank runs the Megatron-Bridge export, which gathers each
   param across TP / EP / PP into a full, unsharded HF tensor. Each rank packs those tensors
   into a reusable CUDA buffer (`weight_transfer_threshold_cuda_ipc_GB`, default 1 GB). Rank 0
   then sends one blocking `update_weights` request per chunk. The vLLM worker on each GPU
   maps that GPU's buffer, clones the tensors out, and loads them layer by layer.
4. Offload the policy, then `wake_up(tags=["kv_cache"])`.

During step 3 one GPU holds all of the following at once:

| Item | Size in the motivating case |
|---|---|
| The engine's woken weights | ~145 GiB per GPU |
| The trainer's reloaded BF16 shard | ~37 GiB per GPU |
| The export's gather transient: one unsharded HF tensor on every rank | up to ~16 GB (one layer's fused BF16 expert tensor) |
| The trainer's packed buffer, the receiver's clones, and one layer of buffered tensors on the receiver | a few GiB |

The motivating case is a ~600B MoE (~1.2 TB of BF16 weights) served by TP8 vLLM engines on
8xB200 nodes (178 GiB per GPU) and colocated with Megatron at TP2 / EP32. The first two rows
alone add up to ~182 GiB, more than the GPU has. Small models never hit this. It limits
colocation once the engine's weights fill most of the GPU.

There is a second cost. Every vLLM worker receives the **whole unsharded model** and slices
out its own shard in the loader. In the motivating case that is ~1.2 TB per worker per sync.
Each byte crosses the IPC buffer, is cloned on the receiver, and is mostly discarded.

## 2. Design

Four independent options. Each one removes one of the costs above.

```
Trainer rank on GPU g                                    vLLM worker on GPU g
---------------------                                    --------------------
params offloaded (pinned CPU)
  -> H2D copy of one param (offloaded-params export)
  -> bridge gather (TP / EP collectives)
  -> keep only the experts worker g serves (local shards)
  -> pack into slot k mod N (slot ring)  -- CUDA IPC -->  map slot (multi-slot importer),
                                                           clone, layerwise load
rank 0: update_weights for slot k on a helper thread,
        while all ranks pack k+1
```

### 2.1 Export from offloaded params

`trainer.policy.megatron_config.weight_sync_from_offloaded_params`.

The trainer's params stay offloaded during the sync. The export reads each param from its
pinned CPU copy, one at a time. The trainer then needs one param's worth of GPU memory for the
read, plus the gather transient, instead of its whole reloaded shard.

- **Dispatch.** `_prepare_for_weight_sync` sleeps the engines, then offloads every model and
  its optimizer (`_sync_from_offloaded_params()`). It does not reload the policy. The offload
  also refreshes the CPU copy from the just-updated GPU params. `_finish_weight_sync` has
  nothing to offload afterwards.
- **Export.** `megatron_utils.export_from_offloaded_params(models)` is a context manager.
  `offload_megatron_model_to_cpu` frees each DDP buffer's GPU storage and keeps the data in
  `buffer.param_data_cpu`. The params remain views of the empty GPU storage. Inside the
  context, Megatron-Bridge's per-param read (`model_bridge._gather_gtp_weight`) is replaced by
  a function that returns a GPU copy of the param's slice of `param_data_cpu` (located through
  `buffer.param_index_map`). It covers `buffers` and `expert_parallel_buffers`. Params whose GPU
  storage is not empty, and tensors it does not know, go to the original function.
- **Source.** `MegatronWeightSource._export()` (behind both `metadata` and `iter_full()`) always enters the context. It is a no-op while
  the params are on the GPU, so the default path is unchanged.
- **Cost.** One host-to-device copy of every param per sync. There is no prefetch: each copy
  is issued when the bridge reads the param.

The export still materializes each HF tensor unsharded on every trainer rank. This option
bounds what the trainer holds *between* tensors, not the largest single gathered tensor. At
full scale that gather transient is the limiting term (see section 5).

### 2.2 Slot ring and async control plane

`generator.inference_engine.weight_sync_num_ipc_buffers` (N).

`SkyrlIPCTrainerWeightTransferEngine._send_packed` routes to `colocated.slot_ring.SlotRingSender`
when any of N > 1, layer-aligned chunks or local shards is set. Otherwise vLLM's packed
producer runs unchanged.

Per chunk k, on every trainer rank:

1. Pack chunk k into slot `k % N`. The export (and its gathers) runs inside this step.
2. `all_gather_object` of this rank's slot handle and, for local shards, its layout.
3. Rank 0 joins send(k-1), then posts send(k) on a one-thread pool. With N = 1 it waits for
   send(k) at once, because the next pack overwrites the same slot.
4. Barrier. After it, every rank knows slot `(k-1) % N` is free.

So the blocking `update_weights` round trip for chunk k overlaps the export and pack of chunk
k+1. Sends stay strictly serialized, so each worker sees one update at a time and nothing
depends on server-side request ordering. N > 2 adds nothing over N = 2 under that rule.

Each slot is `weight_transfer_threshold_cuda_ipc_GB` (floored at the largest param, as
before). N slots cost N times that on every trainer rank.

### 2.3 Layer-aligned chunks

`generator.inference_engine.weight_sync_layer_aligned_chunks`.

A chunk never spans two `.layers.<N>.` prefixes. A layer larger than a slot is split across
several chunks. One round trip then carries about one layer, so the receiver can complete and
process a layer per chunk. A single tensor larger than a slot is an error naming
`weight_transfer_threshold_cuda_ipc_GB`.

### 2.4 Receiver-local shards

`generator.inference_engine.weight_sync_local_shard`.

Each trainer rank packs only what the vLLM worker on its own GPU serves. Today that means
routed experts under vLLM expert parallelism.

- **Plan.** Resolved once, at the first send, on every rank (`_skyrl_local_shard_plan`).
  Trainer ranks all-gather their GPU UUIDs. Rank 0 calls the worker RPC
  `skyrl_local_shard_info` on every server. Each worker returns its GPU UUID, its index in an
  `update_weights` list (`data_parallel_index * world_size + rank`), and its local experts
  (read from the MoE modules' `expert_map`). `build_local_shard_plan` pairs ranks and workers
  by UUID, and rank 0 broadcasts the result.
- **Filter.** A tensor whose name matches `.layers.<N>. ... .experts.<id>.` is packed only by
  ranks whose worker serves expert `<id>`. Everything else is packed by every rank, as today.
- **Agreed chunk cuts.** Every rank must yield the same number of chunks, because the export
  and the per-chunk barrier are collective. Each rank computes every rank's byte count from the
  shared plan and cuts when any rank's slot would overflow (or at a layer boundary). This is
  `iter_packed_chunks`.
- **Wire.** Rank 0 sends one payload per server through
  `SkyrlWeightSyncClient.update_weights_per_server`. Each payload is a list indexed by global
  worker rank, which `GPUWorker.update_weights` already routes. Each entry carries only that
  worker's names and its own GPU's handle.
- **Fallback (logged, full tensors).** No worker has an `expert_map` (no vLLM EP). Ranks and
  workers do not pair one-to-one by GPU. Two ranks share a GPU. MoE layers disagree on
  placement. The draft (MTP) session always sends full tensors.

Fused 3D expert tensors (`[E, ...]` under one name) are not sliced. They still go whole to
every worker.

### 2.5 Receiver: multi-slot importer

vLLM's `PackedBufferImporter` caches one IPC mapping and releases it when a different buffer
arrives. With a ring of slots it would rebuild and release a mapping on every chunk, which
drives the producer-side refcount negative. `SkyrlIPCWeightTransferEngine` therefore always
uses `colocated.slot_ring.MultiSlotImporter`. It opens each slot once per transfer and releases
all of them at finish (`skyrl_release_receive_buffers`). With one slot it behaves exactly like
vLLM's importer.

The receive engine also counts chunks, bytes and tensors per transfer
(`skyrl_receive_stats`), reset at `start_weight_update`.

## 3. Configuration

| Option | Default | Effect | Constraints |
|---|---|---|---|
| `trainer.policy.megatron_config.weight_sync_from_offloaded_params` | `false` | Keep the policy offloaded during the sync and export each param from its pinned CPU copy (2.1) | Megatron; full fine-tuning (no LoRA); no `ddp_config.overlap_param_gather`; no serialized FP8 |
| `generator.inference_engine.weight_sync_num_ipc_buffers` | `1` | Number of IPC slots in the ring. `2` pipelines the round trip behind the next pack (2.2) | `>= 1`. More than 2 adds nothing |
| `generator.inference_engine.weight_sync_layer_aligned_chunks` | `false` | Cut chunks at decoder-layer boundaries (2.3) | — |
| `generator.inference_engine.weight_sync_local_shard` | `false` | Each trainer rank packs only the experts its GPU's worker serves (2.4) | vLLM expert parallelism (`enable_expert_parallel`), else falls back to full tensors; no serialized FP8 |
| `generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB` | `1.0` | Existing. Size of each slot | Floored at the largest param |

### Validation

`_validate_colocated_weight_sync_cfg` in `skyrl/train/utils/utils.py`, called from
`validate_inference_engine_cfg`. If any option above is on, it requires:

- `trainer.placement.colocate_all=true`;
- the push backend (`weight_sync_backend` `nccl` or `ipc`). `sharded_rdt` and `delta` are
  rejected;
- `weight_sync_num_ipc_buffers >= 1`;
- no serialized FP8 (`fp8_weight_sync_mode`) with `weight_sync_local_shard` or
  `weight_sync_from_offloaded_params`.

`weight_sync_from_offloaded_params` additionally requires `trainer.strategy=megatron`,
`lora.rank == 0`, and `overlap_param_gather=false`:

- **LoRA.** The adapters' DDP buffers stay resident and the frozen base weights are offloaded
  outside those buffers. The export does not read those from CPU, so a merged sync would read
  freed params. An adapter-only sync never reloads the base anyway.
- **`overlap_param_gather`.** `sync_params_for_export` needs the params on the GPU to publish
  the optimizer's shards, and this path never reloads them.

Not rejected, and not tested: the ring, layer-aligned and local-shard options under FSDP, and
with the in-memory LoRA adapter sync (`lora.sync_mode=memory`).

### Debug and test hooks

- `SKYRL_COLOCATED_SYNC_DEBUG=1`: each trainer rank records the send's time, torch allocator
  peaks, peak device use (a `mem_get_info` poller, which includes the colocated vLLM worker),
  time and bytes inside the export generator, and round-trip count and time. Read with
  `Worker.get_last_weight_sync_stats()`. It adds device synchronizations; debug only.
- `SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB`: test only. Caps each trainer process's CUDA
  allocations during the send (`torch.cuda.set_per_process_memory_fraction`), to reproduce on
  a small model the OOM of a large one.
- vLLM worker RPCs (over `/collective_rpc`): `skyrl_debug_weight_checksums` (an integer
  checksum of every parameter and buffer),
  `skyrl_debug_memory_stats` (device, CuMem by tag, and the receive counters), and
  `skyrl_local_shard_info`.
- `Worker.reconfigure_for_test(cfg)`: test only. Swaps the trainer config and drops the
  transfer engine, so a test can compare two sync paths on the same trained weights.

Code: `skyrl/backends/skyrl_train/weight_sync/colocated/` (`slot_ring.py`,
`instrumentation.py`), `megatron_utils.export_from_offloaded_params`, and small hooks in
`weight_senders.py`, `weight_receivers.py`, `control_plane.py`, `sources.py`,
`worker_dispatch.py` and `new_inference_worker_wrap.py`.

## 4. Status

**Built:** sections 2.1 to 2.5, the validation, the debug hooks, and the tests:

- CPU: `tests/train/test_colocated_weight_sync_config.py` (validation),
  `tests/backends/skyrl_train/weight_sync/test_colocated_slot_ring.py` (chunk cuts, layer
  alignment, local-shard agreement across ranks, plan building),
  `tests/backends/skyrl_train/distributed/test_export_from_offloaded_params.py` (the
  offloaded-params read against fake DDP buffers; needs the `megatron` extra).
- GPU, opt-in: `tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_colocated_weight_sync.py`.
  It boots colocated vLLM from the checkpoint, records a checksum of every vLLM tensor, sleeps
  at level 2, runs the production `save_weights_for_sampler`, and requires every tensor on every
  worker to be byte-identical to the fresh load. A second test takes a training step and
  requires the configured path to write the same vLLM bytes as the default path. It is skipped
  unless `SKYRL_CWS_MODEL` is set. See the results doc for how to run it.

**Follow-up PRs:**

- **Selective weight sleep and staged wake-on-write.** Record which weight memory the reload
  rewrites, discard only that at sleep, and map each layer just before it is written. Built as
  a prototype. It passed byte parity on Qwen3-30B-A3B but did not shorten the sync there
  (2.0-2.1 s without wake-on-write, 2.2 s with it) and was not used at full scale. Held back.
- **Always-on colocated changes.** Release trainer garbage, cuBLAS workspaces and cached
  blocks before and after a colocated sync; LoRA adapter host staging; an adapter-only buffer
  floor; FP32 wire dtype only for FP32-sourced tensors.
- **Model hooks.** A clone-free receive for models that load in place, and a local-expert hook
  for models that shard experts over TP themselves instead of using vLLM EP.
- **Not built:** a per-GPU trainer budget that sizes the slots; slicing fused 3D expert tensors
  per worker; TP-slice local shards (a baked per-worker slice plan, as `sharded_rdt` does).

## 5. Findings for maintainers

1. **Megatron-Bridge per-instance state is silently lost.** `AutoBridge._model_bridge` is a
   property that builds a new bridge instance on every access, and the export builds another.
   A flag set on `bridge._model_bridge` never reaches the instance that exports. On a ~600B MoE
   a per-instance "export experts one at a time" flag was lost this way. Every sync stacked a
   whole layer's experts (~15 GiB), ran out of memory next to the woken engine, and fell
   back to stacking on the CPU. That made the sync take ~18 minutes. With the flag set at class
   level (and findings 4 and 5 fixed), one 1 GB slot plus local shards synced in ~23.5 s. The
   number of round trips was not the cause. Set such state at class level.
2. **The trainer's gather transient is the limiting term at full scale.** With the engine woken
   (~143-145 GiB per GPU) and the trainer at ~32-35 GiB with its params offloaded, ~9 GiB was
   left free after a sync. A second 1 GB slot, or one 2 GB slot, ran the trainer out of memory
   in the EP/TP gather. So the ring ran with one slot there.
3. **The ring gains nothing when the receiver is the bottleneck.** On Qwen3-30B-A3B rank 0
   waited ~3.8 s on sends against a 0.8 s export: the receiver's clone and layerwise load set
   the pace. The ring helps when the trainer's gather is slow. Local shards help in both cases,
   because they cut what the receiver clones and loads.
4. **The receiver's clone can run out of memory near capacity.** vLLM's packed IPC consumer
   clones each chunk before loading: ~1.73 GiB for an embedding-sized chunk at full scale,
   which ran the engine out of memory with ~9 GiB free. The clone-free receive is a follow-up.
5. **Trainer garbage holds GPU memory across the sync.** After a training step, 3-8 GiB of
   activations survived in reference cycles, plus cuBLAS workspaces and cached blocks. That was
   enough to fail vLLM's KV-cache wake after the sync. The cleanup is a follow-up.
6. **IPC importer per-chunk rebuild.** vLLM's one-entry `PackedBufferImporter` cannot serve a
   ring (section 2.5).
7. **Byte parity, not greedy equality, is the correctness gate.** With CUDA graphs on, a fresh
   engine with no sync already diverges from itself at near-tie tokens, and the default path
   diverges the same way after a sync. The harness asserts byte parity of every vLLM tensor and
   only checks that greedy output does not diverge before token 8.
8. **FlashInfer autotune cache is per rank.** vLLM 0.33 caches autotune results per rank. A run
   killed mid-autotune leaves some ranks cached, and the next boot hangs in `profile_run`. The
   harness uses a fresh `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` per run.

## 6. Open risks

| Risk | Impact | Mitigation |
|---|---|---|
| Gather transient: the export holds each unsharded HF tensor on every rank | Can fail next to the woken engine; limits slots to one at full scale | An owner-to-consumer reshard (all-to-all of each worker's slice) instead of all-gather, or exporting fused experts in EP-local pieces |
| Fused 3D expert tensors are not filtered by local shards | Bytes per worker stay high for models that name experts that way | Slice along dim 0, after checking whether the loader expects global or local expert order |
| A failed `update_weights` on rank 0 | Rank 0 raises before the per-chunk barrier; the other ranks wait in the barrier until the collective times out | Abort the sync on all ranks |
| Private extension points: Megatron-Bridge `model_bridge._gather_gtp_weight`; megatron-core `param_data_cpu` / `param_index_map`; vLLM `IPCTrainerWeightTransferEngine._send_packed` and `_packed_importer` | Silent breakage on a dependency bump | The CPU test for the export and the GPU harness are the bump gates |
| Host memory and H2D bandwidth | Pinned trainer copies, optimizer state and the engine's level-1 backup share the host | Check per-node totals at scale |
| Ring, layer-aligned and local-shard options are untested under FSDP and with in-memory LoRA sync | Unknown | Add a rejection or a test |

## 7. Upstream asks

None is required; each would remove a private hook.

- **vLLM:** a `num_buffers` option and per-worker layouts in the packed IPC producer, and a
  multi-slot `PackedBufferImporter`.
- **Megatron-Bridge:** a public hook to read a param's data during export, instead of
  replacing `_gather_gtp_weight`.
