# Weight Sync

Training-to-inference weight transfer. Runs after every training step (or on the configured interval) to push updated policy weights from training workers (FSDP/Megatron) into the vLLM inference engines.

## Architecture

Weight sync runs on vLLM's **trainer-send** abstraction. Each training worker builds
a `WeightSource` over its live model and hands it to a `TrainerWeightTransferEngine`,
whose `send_weights()` owns the whole round trip (`start_weight_update` →
`update_weights` → `finish_weight_update`, plus barriers and the non-sender collective
replay). SkyRL owns three things and nothing else: the source, the control-plane
client, and choosing an init info.

**Chunking is vLLM's responsibility.** SkyRL passes a source and stops — it does not
decide, express, or bound how the stream is cut on the wire. The packed NCCL and IPC
producers already chunk out of a fixed reusable buffer and consume the source lazily,
so a source's only obligation is to be a lazy generator that does not retain.

```
skyrl/backends/skyrl_train/weight_sync/
├── __init__.py             # backend selection: get_transfer_strategy / get_vllm_receive_backend
├── base.py                 # LoraLoadRequest (not a weight transfer -- an adapter path)
├── sources.py              # FsdpWeightSource / MegatronWeightSource (vLLM's metadata()+__iter__)
├── trainer_engines.py      # build_trainer_engine: init info + client -> trainer_init
├── control_plane.py        # SkyrlWeightSyncClient (blocking HTTP) + per-server init rewrites
├── skyrl_engines.py        # receive side: skyrl_nccl / skyrl_ipc (+ the drafter-reload proxy)
├── delta_trainer.py        # DeltaTrainerWeightTransferEngine (send side)
├── delta_engine.py         # DeltaWeightTransferEngine (receive side, in the vLLM worker)
├── delta_checkpoint.py     # DeltaCheckpointPublisher, LocalCheckpointStore, manifest + XOR payloads
├── delta_payload.py        # zstd compress/decompress + uint8 tensor <-> bytes helpers
└── sharded_rdt/            # the sharded_rdt (NIXL pull) backend; __init__ is import-free
    ├── rdt_send.py             # its WeightSources + build_rdt_trainer_init_info
    ├── sharded_rdt_base.py     # GroupedWeightSource + layerwise_groups (the two RDT-only channels)
    ├── rdt_vllm_register.py    # registers the sharded_rdt engine into vLLM's factory
    ├── rdt_libfabric_shim.py   # LIBFABRIC provider shim for NIXL
    ├── sharded_rdt_trainer.py  # vendored: trainer engine + the _RDTProducerServer sidecar
    ├── sharded_rdt_engine.py   # vendored: consumer engine (runs in the vLLM worker)
    ├── sharded_rdt_common.py   # vendored: RdtRouter, op-chain allowlist, buffer sizing
    └── sharded_rdt_fake.py     # vendored: FakeRDTTensor placeholders for the bake
```

Neither `__init__.py` imports anything: `sources`, `trainer_engines`, `delta_trainer`,
`sharded_rdt_engine` and `sharded_rdt_trainer` import `vllm` at module scope, so a
re-export would pull vllm into every `weight_sync` import and break the CPU CI job that
runs without the wheel. Import those modules at their call sites.

vLLM worker-extension class (loaded via `--worker-extension-cls`):

- `inference_servers/new_inference_worker_wrap.py` — `NewInferenceWorkerWrap`. Holds
  exactly **two** weight-sync things, both limits of *dispatch* rather than of the
  engine abstraction: `fetch_weights` (`/collective_rpc` reaches worker methods by name
  only, and no native route can invoke an engine method) and the sleep/wake pair
  (`EngineCore.sleep` hardcodes the prefix-cache clear). It is also where the
  receive-side factory registrations and the model-runner recorder patch are installed,
  because vLLM imports it in every worker process before model init.

### Trainer side, per backend

| logical backend | trainer engine | receive engine (`WeightTransferConfig.backend`) |
|---|---|---|
| `nccl` | vLLM's `NCCLTrainerWeightTransferEngine` | `skyrl_nccl` |
| `ipc` | vLLM's `IPCTrainerWeightTransferEngine` | `skyrl_ipc` |
| `delta` | `DeltaTrainerWeightTransferEngine` | `delta` |
| `sharded_rdt` | `ShardedRDTTrainerWeightTransferEngine` | `sharded_rdt` |

The receive side takes new names for NCCL and IPC because SkyRL subclasses vLLM's
engines (to reload the spec-decode drafter) and `register_engine` raises on an
already-registered name. The trainer-side factory has its own registry, so the send side
keeps vLLM's names.

## Transfer backends

- **Broadcast** (`nccl`): NCCL collective, packed. Used for **non-colocated** setups. Training and inference are on different GPUs; weights cross the wire over a dedicated process group.
- **CUDA IPC** (`ipc`): packed IPC handles out of one reusable buffer. Used for **colocated** setups (`colocate_all=true`). Both sides live on the same GPU; the receiver maps the sender's CUDA allocation directly and clones out of it.
- **Delta** (`delta`): Weights travel as compressed XOR deltas against the base checkpoint, through a shared filesystem or object store instead of the network fabric. Selected with `generator.inference_engine.weight_sync_backend=delta`; intended for **non-colocated** setups where the two sides are not NCCL-reachable (separate clusters, PD-disaggregated serving). Not supported with LoRA (`validate_cfg` rejects it).
- **Sharded RDT** (`sharded_rdt`): the inference workers **pull** the slices they consume from
  the trainer ranks over NIXL/RDMA, instead of the trainer pushing every tensor to every
  worker. Selected with `generator.inference_engine.weight_sync_backend=sharded_rdt`;
  non-colocated only (`placement.colocate_all=false`), Megatron or FSDP, and it forces
  `distributed_executor_backend=ray` because the workers dial named trainer actors. See
  the dedicated section below.

Selection is `get_transfer_strategy(weight_sync_backend, colocate_all)`, called from two
places that must agree: `inference_servers/utils.build_vllm_cli_args` on the driver (via
`get_vllm_receive_backend`, to build the servers' `WeightTransferConfig`, and to force
`distributed_executor_backend=ray` for `sharded_rdt`), and `build_trainer_engine` on each
worker. Both read the same two config values, so the trainer and receive engines cannot
disagree.

### Weight sources

`sources.py` implements exactly vLLM's two-channel contract, and the two channels must
agree element for element — `metadata()` declares what iteration will yield, and the
engine sizes the worker's receive buffers and cuts its packed chunk boundaries from it.
`NCCLTrainerWeightTransferEngine._checked_iter` enforces that at runtime and names the
first divergent parameter; the Megatron GPU test
(`gpu_ci/megatron/test_megatron_weight_source.py`) is the regression check that gets
there first.

- `FsdpWeightSource` — `state_dict()` for metadata (FSDP2 `DTensor.shape` is already the
  global shape, so declaring costs no collective); iteration all-gathers via
  `materialize_full_tensor`. `weight_prefix` handles syncing a CausalLM backbone into a
  vLLM multimodal namespace.
- `MegatronWeightSource` — `bridge.export_hf_weights(conversion_tasks=None)`, a lazy
  generator in HF-canonical order that gathers TP/PP/EP internally. `metadata()` must
  materialize once to learn shapes, so it runs a dry export and caches.

**There is no bucketing, and that is deliberate.** Bucketing does not bound memory — it
*accumulates* a whole bucket before handing it on, where the unbucketed export yields one
parameter at a time. What it was for, IPC handle count and Flash-RL fused-loader grouping,
`packed_ipc_producer`'s single reusable buffer subsumes. And one whole-model
`export_hf_weights` call satisfies `_accumulate_grouped_export`'s "every task of a
`group_key` in one call" requirement by construction, where bucketing has to special-case
it (splitting them means expert weights are silently never yielded).

### Control plane

`control_plane.SkyrlWeightSyncClient` is a **blocking** HTTP client over vLLM's native
RLHF routes, because `VLLMWeightSyncClient` is a synchronous protocol and the engine runs
off the event loop (`asyncio.to_thread(engine.send_weights)`). Four properties are
load-bearing: `Connection: close` (a full training step elapses between syncs, so any
keep-alive is stale), concurrent fan-out (required for *correctness* on the RDT path — a
serial `update_weights` deadlocks the producer's ref-counted group free), body-aware error
messages, and no timeout / no retry.

Two per-server init rewrites live there, because each engine builds only **one**
worker-side init dict:

- `nccl_init_payloads` — cumulative `rank_offset`, advancing one deployment's worth per
  deployment and staying put across the DP servers *within* one (vLLM's
  `data_parallel_index` already separates those). This is the highest-risk arithmetic in
  weight sync: a wrong offset mis-maps ranks and **hangs in the NCCL rendezvous** rather
  than erroring, so it cross-checks `world_size` against what the offsets imply.
- `rdt_init_payloads` — the deployment ordinal as `replica_rank` plus `num_replicas`, so
  the engine can offset its consumer ids into globally distinct ranges.

### Capability probes

Three things the trainer engine cannot do for itself are decided by `getattr` probes on
the engine in `Worker._sync_weights_to_inference_engines`:

| probe | default | who sets it |
|---|---|---|
| `skyrl_handles_prefix_cache_reset` | False | delta (it resets inside its own pause bracket) |
| `skyrl_force_disable_expandable_segments` | False | sharded_rdt (CUDA-IPC shares on every run, not only under colocation) |
| `skyrl_empty_cache_after_send` | True | sharded_rdt sets False (buffers are reused next step) |

They are probes and not declared attributes because two of the four engines are vLLM's
own classes and cannot carry SkyRL attributes at all — so the *absence* of a flag is the
common case and must mean the default. `skyrl_set_reset_prefix_cache(bool)` is the same
idea for a per-round value, since `send_weights()` takes no arguments.

## Delta backend

Unlike the push backends, delta sync does not send tensors to the receiver at all — it
publishes bytes to `sync_dir` and the receiver pulls them.

**Publish (trainer, rank 0).** `DeltaCheckpointPublisher` keeps a CPU `uint8` snapshot of the
full model, XORs it against the new weights to get a per-tensor patch, zstd-compresses each
patch, and writes them as safetensors payload files plus a `manifest.json` under
`<sync_dir>/delta-<version:08d>/`. Unchanged tensors are omitted. Payload files roll over at
`max_file_size_in_gb`.

**Fetch (receiver, before pause).** A control-plane operation the other backends do not have:
`SkyrlWeightSyncClient.fetch_weights` → `/fetch_weights` on every server, driven by
`DeltaTrainerWeightTransferEngine._apply_receiver_update` *before* generation is paused, so
the download and patch-apply happen off the critical path. It is also why the worker
extension still exists (see Architecture): `/collective_rpc` reaches worker methods only.

`LocalCheckpointStore` maintains a
mutable copy of the checkpoint under `local_checkpoint_dir/weights`, replaying every delta from
its current version up to the target (so a late-joining engine catches up), and applies each
patch by XOR-ing directly into the mmap'd safetensors files.

**Reload.** Only then is generation paused and the local checkpoint reloaded into vLLM via
`iter_tensors`. Note the delta shrinks the *transfer*, not the reload — the whole checkpoint is
re-read even for an empty delta. The pause/resume bracket and the prefix-cache reset ride the
engine, not the four-method client protocol: this is the only backend that reloads a whole
checkpoint in place, so it is the only one that needs generation stopped.

The engine owns its layerwise-reload lifecycle like every other one
(`start_weight_update` → `initialize_layerwise_reload`, `finish_weight_update` →
`finalize_layerwise_reload`), which is what runs `process_weights_after_loading`. Its
drafter reload takes a **second** `iter_tensors` pass rather than the push backends'
proxy: materializing a whole checkpoint into a list so the drafter can re-read it would
put the entire model in memory at once.

`DeltaWeightSyncConfig.__post_init__` derives `local_checkpoint_dir` and `publish_staging_dir`
from `sync_dir` when unset, so consuming classes never invent their own defaults.

## Sharded RDT (`sharded_rdt`)

`sharded_rdt` is selected through `get_transfer_strategy` and built by the same
`build_trainer_engine` call as every other backend; nothing about its trainer side is
special. What *is* special is that it **pulls**, and pulling needs two
channels vLLM's `WeightSource` has no concept of. Those live on
`sharded_rdt_base.GroupedWeightSource` and stay inside this package
(`weight_sync/sources.py` is vLLM's contract verbatim, so a future upstream chunking
change arrives for free):

| channel | what it really is |
|---|---|
| `held_names()` | **ownership** under PP / EP. Feeds `_resolve_ownership` → `_spawn_server(held)`, and consumers route pulls to producers by name. Not a chunking concern, and not optional: the default (hold everything) is correct at pp=1/ep=1 and wrong above it. |
| `groups()` | the coordination index the producer's free barrier counts (`_inflight` is keyed by group). |
| `iter_groups()` | batching — driving the gather per group instead of per tensor turns ~37k generator resumes into ~95 on a per-expert MoE model (~0.9s/sync at 235B). |

Three source flavors, chosen by `make_megatron_weight_source` / `make_fsdp_weight_source`:

- `RdtFsdpWeightSource` — the shared FSDP source re-ordered **group-major**, so
  `layerwise_groups` partitions `metadata()` exactly. The push backends do not care about
  order (any permutation transfers identically as long as the two channels agree), which
  is why only RDT pays for the reorder.
- `MegatronStackedWeightSource` — PP-local and EP-local, gathering experts at stack
  granularity. `_pp_local_export_ctx` patches the bridge's PP broadcast so the owning
  stage gets its tensor and every other stage gets `None` (which `megatron_to_hf` already
  skips). Without it every rank materializes the whole model under PP; this is the actual
  OOM fix, measured on Qwen3-32B at tp4/pp2 (70.56 GiB) and tp8/pp2 (73.06 GiB) of 79.18.
- `RdtMegatronWeightSource` — the whole-model fallback, taken on four conditions
  (`_demoted`: a gather group spans PP stages, i.e. tied embeddings or MTP; grouped-export
  archs; `etp != 1`; dense at `pp == 1`).

**PP/EP locality cannot be shared with the push backends.** It needs per-rank ownership,
so only a pull backend can consume a PP-local source: NCCL broadcasts from rank 0, IPC
shares whole-tensor handles, and delta publishes a whole checkpoint — all three require
the cross-stage export. Conversely, RDT's whole-model residency is *the point* for a pull
backend (each producer must be able to serve its bound consumer the complete model), which
is why the Qwen3-32B OOM does not transfer to the push backends: they stream and drop.

Two capability probes (see Architecture): `skyrl_force_disable_expandable_segments = True`
(the sidecar shares gathered tensors over CUDA IPC on every run, and VMM memory makes
export/rebuild 5-10x slower per storage) and `skyrl_empty_cache_after_send = False`
(publish buffers are reused by the next training step).

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

## Receive-side lifecycle

vLLM's native routes, driven by the trainer engine through `SkyrlWeightSyncClient`:

1. `/init_weight_transfer_engine` → `engine.init_transfer_engine(...)`. The **one**
   lifecycle method `GPUWorker` does not wrap in `set_current_vllm_config`, which is why
   the RDT engine opens that context itself (its bake drives `model.load_weights` against
   meta params, and `process_weights_after_loading` on MoE models reads
   `get_current_vllm_config()` to build kernels).
2. `/start_weight_update` → `engine.start_weight_update()` → `initialize_layerwise_reload`
   (moves layers to meta device, wraps loaders).
3. `/update_weights` → `engine.update_weights(...)`, once or many times. The actual
   receive; wrapped in `disable_mtp_completeness_check()` because MTP architectures raise
   on incomplete layer coverage.
4. `/finish_weight_update` → `engine.finish_weight_update()` → `finalize_layerwise_reload`
   (quantization repacking, attention weight postprocessing), then
   `model_runner.reset_lora_state()` on the worker.

`GPUWorker` opens `set_current_vllm_config` around steps 2-4 itself, which is why the
receive path needs no SkyRL wrapper.

### Spec-decode drafter reload

vLLM's engines call `self.model.load_weights(...)` directly and there is still no
`load_weights` callback, so `skyrl_engines.SkyrlDrafterReloadMixin` swaps `self.model` for
a `_LoadWeightsProxy` for the duration of `receive_weights` — the drafter
(`model_runner.drafter.model`, a separate module the main load never touches) is then
reloaded from exactly the weights the main model just received. The proxy is only
installed when this process actually *has* a drafter, so a non-MTP deployment runs vLLM's
path verbatim.

Delta is the exception: it re-streams `iter_tensors` a second time instead, because the
proxy has to materialize the weight list so the drafter can re-read it, and for a whole
checkpoint that is the entire model resident at once.

An engine has no route to its worker — it is constructed inside `Worker.load_model`, and
the worker-extension class is appended to `Worker.__bases__` *after* `Worker` — so
`patches/vllm/patch_model_runner_registry.py` wraps `GPUModelRunner.load_model` to record
the runner in a process-global weakref.

## KV offload during non-colocated weight sync

Non-colocated normally keeps the engine fully awake and does `pause_generation → broadcast → resume_generation`. The opt-in `generator.inference_engine.offload_kv_for_weight_sync` flag sleeps the engine (freeing the KV cache from GPU) *during* the sync so `gpu_memory_utilization` can be pushed higher (no need to keep KV cache resident alongside the weight-transfer scratch buffers). It turns on `enable_sleep_mode` (via `inference_servers/utils.py`). Requires non-colocated and non-LoRA. Orchestrated in `WorkerDispatch.save_weights_for_sampler`; the flow depends on the trainer:

- **Synchronous trainer** (`fully_async.enabled=false`): generation is complete at sync time, so there are no in-flight requests. A plain `sleep() → wake_up(["weights"]) → broadcast → wake_up(["kv_cache"])` (the same three-phase pattern colocated uses) is enough — the standard `/sleep`+`/wake_up` endpoints discard the KV cache and free the memory.
- **Fully-async trainer** (`fully_async.enabled=true`): generation overlaps the sync, so `pause_generation` (KEEP) freezes in-flight requests, then the allocator is driven directly (see below) so the scheduler is **not** resumed on the weights wake. The KV cache is offloaded to CPU and restored so frozen requests resume with no abort or prefill recompute — **unless** `clear_kv_cache_on_weight_sync=true`, in which case the broadcast resets the prefix cache anyway, so the KV is discarded (skipping the CPU copy) rather than offloaded.

The fully-async path is driven entirely from SkyRL — no vLLM patch. It deliberately avoids the `/sleep`+`/wake_up` HTTP endpoints (which route through `EngineCore.sleep`, force-clearing the prefix cache and preempting every running request at level ≥ 1). Instead it drives the per-worker `CuMemAllocator` directly via two `NewInferenceWorkerWrap` methods invoked over `/collective_rpc`:

- `skyrl_sleep_for_weight_sync(offload_kv)` — `allocator.sleep(offload_tags=("kv_cache",) if offload_kv else ())`: **discards** the weights pool (the broadcast overwrites every parameter on wake) and either offloads the KV cache to CPU or discards it. Model buffers live in the weights pool and are NOT covered by the parameter broadcast (e.g. non-persistent rotary `inv_freq`), so they are saved to CPU here and restored on wake — mirroring `GPUWorker.sleep(level=2)`. All GPU memory is freed regardless. The scheduler is untouched, so KEEP-paused requests stay frozen with valid block tables.
- `skyrl_wake_for_weight_sync(tags)` — `torch.cuda.empty_cache()` (release the broadcast's transient buffers so cumem can remap the KV pool) then `allocator.wake_up(tags)`, which remaps to the **same virtual addresses** and copies CPU→GPU so block tables remain valid. On the `weights` wake it restores the saved buffers; on the `kv_cache` wake it re-inits fp8 KV scales. Does not resume the scheduler (the client does that via `/resume`).

Validated in `validate_inference_engine_cfg`. vLLM-version coupled (mirrors `GPUWorker.sleep`/`wake_up` and the `CuMemAllocator` API) — re-verify on vLLM bumps via the GPU weight-sync test.

## Convention: vLLM imports

`vllm` is a Linux-only optional dep, and half the CPU suite runs without the wheel. Two
rules, not one:

- A module that **anything** vllm-free imports must import `vllm` **lazily inside
  methods**: `weight_sync/__init__.py`, `base.py`, `delta_checkpoint.py`,
  `delta_engine.py`, `control_plane.py` and `new_inference_worker_wrap.py` all follow this.
- A module that is *itself* vllm-only may import at module top — `sources.py`,
  `trainer_engines.py`, `delta_trainer.py`, `sharded_rdt_base.py`,
  `sharded_rdt_{engine,trainer}.py`. The rule then moves up: **nothing vllm-free may
  import them at module scope.** The workers import `sources` / `trainer_engines` inside
  `_build_weight_source` / `init_weight_sync_state` for exactly this reason, and
  `weight_sync/__init__.py` re-exports neither.

Tests for the second group carry `pytest.importorskip("vllm")` plus
`pytestmark = pytest.mark.vllm`, which is what puts them in the `-m "vllm"` CPU half.

## Tests

```bash
# CPU — the control plane's per-server init rewrites, the delta publisher, and the
# sharded_rdt pull plan / producer sidecar / grouped source contract
uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/ -v

# GPU — end-to-end weight sync: all four backends through build_trainer_engine
uv run --isolated --extra dev --extra fsdp \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v

# GPU — the Megatron source's two channels must agree (what _checked_iter enforces at runtime)
uv run --isolated --extra dev --extra megatron \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_weight_source.py -v

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
| A `WeightSource` (`sources.py` or `rdt_send.py`) | `test_sources.py` + `test_sharded_rdt_source.py` (CPU), GPU `test_weight_sync.py`, and — for Megatron — GPU `test_megatron_weight_source.py` |
| `control_plane.py`, especially the init rewrites | `test_control_plane.py` (CPU) **and** GPU `test_weight_sync.py` |
| `trainer_engines.py` / a trainer engine | `test_trainer_engines.py` (CPU) **and** GPU `test_weight_sync.py` |
| `skyrl_engines.py` (receive side) | GPU `test_weight_sync.py` only — it runs inside the vLLM worker |
| `NewInferenceWorkerWrap` | GPU `test_weight_sync.py` (CPU tests will not catch regressions) |
| Delta publish / manifest / payload format | `test_delta_checkpoint.py` **and** GPU `test_delta_weight_sync_e2e.py` |
| `LocalCheckpointStore` (fetch, replay, apply, cache keys) | `test_delta_checkpoint.py` |
| `DeltaWeightTransferEngine` (receive side) | GPU `test_delta_weight_sync_e2e.py` only — it runs inside the vLLM worker |
| Who pauses / resets the prefix cache | `test_prefix_cache_reset.py` **and** `distributed/test_worker_dispatch.py` |
| `DeltaWeightSyncConfig` defaults or validation | `tests/train/test_config.py` |

## vLLM version coupling

`vllm` is pinned in `pyproject.toml` (currently `0.28.0`, in both the `fsdp` and `megatron` extras). Weight-sync code paths are tightly coupled to vLLM internals. When bumping the pin, re-verify the GPU weight-sync tests and the extension points listed below.

### `packed` is chosen by SkyRL, agreed by vLLM

`packed` rides the **init** info, and the trainer engine propagates it to the worker at
`trainer_init`, so the two sides structurally cannot disagree. What SkyRL chooses is
`packed=True` on both push backends:

- NCCL: packed broadcasts out of a fixed reusable buffer instead of one NCCL call per
  parameter. vLLM's unpacked path is only reached by its own tests.
- IPC: `packed=True` is **not** vLLM's default but is required here. The unpacked path
  holds a strong ref to a contiguous copy of *every* parameter until past
  `finish_weight_update` (so the consumer's IPC views stay valid) — i.e. the whole model
  resident on the trainer. Packed streams through one reusable buffer, and the consumer
  clones out of it.

### Extension points relied on

- `WeightTransferEngineFactory` / `WeightTransferTrainerFactory` — `register_engine`
  raises `ValueError` on a duplicate name, and `WeightTransferConfig.backend` is typed
  `Literal[...] | str` and validated against the registry at engine creation.
- `TrainerInitInfo` — `backend` is a `ClassVar` the factory dispatches on; `rank` is
  keyword-only.
- `VLLMWeightSyncClient` — a **structural** (PEP 544) four-method protocol, so
  `SkyrlWeightSyncClient` needs no import or subclassing. Note 0.28 declares
  `finish_weight_update(weight_version: str | None = None)`.
- `set_weight_update_target` / `reset_weight_update_target` on `WeightTransferEngine` —
  the draft-session hook. SkyRL's proxy swap composes with it by restoring the exact
  object it found.
- `GPUModelRunner.load_model` — wrapped by `patch_model_runner_registry`.
- `CuMemAllocator` via `vllm.device_allocator.get_mem_allocator_instance` — the KV-offload
  path drives it directly because `EngineCore.sleep` hardcodes
  `clear_prefix_cache = level >= 1` and `CuMemBackend.suspend` cannot express "discard
  weights, offload kv_cache". Registering a custom `SleepModeBackend` does not help: the
  problem is on the dispatch path, not in the suspend mechanism.

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

- Every receive-side engine is registered as an **import side effect** of `new_inference_worker_wrap.py`, the module vLLM loads via `--worker-extension-cls`. Registering anywhere else (e.g. while building CLI args in the driver) is a no-op for the workers — it has to happen in the process that owns the engine. The driver registers too, separately, because it validates `WeightTransferConfig.backend` against the registry.
- `send_weights()` must be called on **every** trainer rank, and every rank must drain the source: iterating it is what drives the gather collectives, so a rank that skipped it deadlocks its peers. Only rank 0 touches the wire (the engines resolve `is_sender` from `init_info.rank`, never from a global process group, which is ambiguous once FSDP/TP/PP/EP groups exist).
- Delta: the receive-side `delta checkpoint fetch:` / `receive reload-only:` log lines are emitted inside the nested vLLM worker process and do **not** reach the driver log, even with `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`. Find them with `grep -rhE "delta checkpoint (fetch|receive)" /tmp/ray/session_latest/logs/`. Filter by mtime — that directory accumulates lines from earlier runs.
- Delta: `_safe_path_name` appends a digest of the full value because sibling delta URIs differ only in their trailing `delta-<version>`; a plain length cap collapses every version onto one cache directory. Don't "simplify" it back to truncation.
- Delta: `s3://` needs the `s5cmd` CLI, which the `aws` extra installs into the run's venv (`--extra aws`). `gs://` needs the `gcloud` CLI as a *system* binary — the `gcp` extra only provides a Python library and will not satisfy it.
