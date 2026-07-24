"""Trainer-side driver for the sharded-RDT (NIXL pull) weight-sync backend.

RDT is deliberately kept OUT of SkyRL's ``WeightTransferStrategy`` /
``WeightTransferSender`` abstraction. That abstraction encodes trainer-side send
logic for the legacy *push* backends (NCCL broadcast, CUDA IPC), which extract
weights on the worker and hand chunks to a sender. RDT instead already matches
vLLM's *new* trainer-send model — a ``WeightSource`` + a
``TrainerWeightTransferEngine`` + a ``VLLMWeightSyncClient`` — where the engine
owns the whole round trip and the inference workers pull. Forcing that through
``send_chunks(chunks, metadata)`` meant ignoring both args and bolting
RDT-only hooks onto the shared base classes.

So we bypass the abstraction: ``Worker.init_weight_sync_state`` builds a
``RdtWeightSyncSender`` for the RDT backend, and ``broadcast_to_inference_engines``
calls ``.send()`` directly (skipping extraction + ``send_chunks`` entirely). This
is an intermediate state: when vLLM ships trainer-send for NCCL/IPC too
(see ``vllm-trainer-send-pr3``), those backends collapse into this same shape and
the ``WeightTransferStrategy`` layer is deleted — at which point this file's
``_FsdpWeightSource`` + the ``SyncRdtControlPlaneClient`` glue become the canonical
path and the vendored ``sharded_rdt_*`` files are dropped in favor of
``vllm.distributed.weight_transfer``.

Two ``WeightSource`` flavors: ``_FsdpWeightSource`` (FSDP ``FSDPWeightExtractor``,
all-gather via ``full_tensor()``) and ``MegatronWeightSource`` (Megatron
``MegatronWeightExtractor``, gather via the Megatron-Bridge ``export_hf_weights``).
``_trainer_init_blocking`` selects between them by the extractor flavor.
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

import torch

from skyrl.backends.skyrl_train.weight_sync.rdt_control_plane import (
    SyncRdtControlPlaneClient,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    WeightSource,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
    layerwise_groups,
)
from skyrl.train.utils.utils import str_to_torch_dtype

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig

logger = logging.getLogger(__name__)

# Defaults for the must-agree wire knobs (mirror the vLLM sharded_rdt defaults).
_DEFAULT_NUM_RDT_BUFFERS = 2
_DEFAULT_LAYERWISE_SPLIT = 1
_DEFAULT_ARENA_PRESIZE_GB = 0.0
_DEFAULT_PACK_CHECK = False
# Max gathered-but-not-yet-freed groups the producer holds at once (backpressure).
_DEFAULT_GATHER_LOOKAHEAD = 2


class _FsdpWeightSource(WeightSource):
    """``WeightSource`` over the FSDP policy model for the sidecar trainer engine.

    Yields ``(name, full tensor)`` pairs in group-contiguous (pre / per-layer /
    post) order, cast to the inference dtype, using the worker's
    ``FSDPWeightExtractor`` so the names (incl. ``weight_prefix``) and the
    all-gather match exactly what the consumer engine baked its plan over.

    ``metadata()`` reads state_dict shapes only (no gather); iteration all-gathers
    each parameter (``full_tensor()``) and is therefore a collective every trainer
    rank must run in lockstep — which the vendored engine's ``send_weights``
    guarantees (all ranks iterate the source).
    """

    def __init__(self, weight_extractor: Any, dtype: torch.dtype) -> None:
        self._extractor = weight_extractor
        self._dtype = dtype
        meta = weight_extractor.get_weight_metadata(dtype)
        names = list(meta["names"])
        shapes = [list(s) for s in meta["shapes"]]
        # Reorder into group-major order so layerwise_groups(names) partitions the
        # list exactly and each model.layers.<N>.* block is contiguous (the order
        # the trainer engine validates + the gather loop drives).
        idx = {n: i for i, n in enumerate(names)}
        order = [idx[n] for g in layerwise_groups(names) for n in g]
        self._names = [names[i] for i in order]
        self._shapes = [shapes[i] for i in order]

    def metadata(self) -> List[ParamMeta]:
        return [ParamMeta(name, self._dtype, tuple(shape)) for name, shape in zip(self._names, self._shapes)]

    def __iter__(self) -> Iterator[tuple]:
        # A worker thread does not inherit the main thread's current CUDA device;
        # set it so the gather collectives + casts land on this rank's GPU.
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
        prefix = getattr(self._extractor, "weight_prefix", "") or ""
        # Use the SAME model handle the metadata came from: get_weight_metadata
        # reads ``weight_extractor.model.state_dict()`` (the inner HF model),
        # whereas the worker's ``self.model`` may be a wrapper prefixed otherwise.
        sd = self._extractor.model.state_dict()
        for name in self._names:
            raw = name[len(prefix) :] if prefix and name.startswith(prefix) else name
            full = self._extractor._gather_tensor(sd[raw]).to(self._dtype).detach().contiguous()
            yield name, full


class MegatronWeightSource(WeightSource):
    """``WeightSource`` over the Megatron policy model for the sidecar trainer engine.

    Wraps ``MegatronWeightExtractor`` and streams ``(HF name, full tensor)`` pairs
    via the Megatron-Bridge (``bridge.export_hf_weights``), which gathers each
    parameter across TP / PP / EP internally and — per its contract — "All ranks
    get full tensors". That is exactly the FSDP ``full_tensor()`` semantics the RDT
    producer model needs: every trainer rank materializes the whole model (one
    parameter at a time, so peak memory is bounded), so each producer can serve
    its bound consumer the complete model.

    We call the bridge **directly** with ``conversion_tasks=None`` rather than the
    extractor's ``extract_weights`` because the extractor is built with
    ``enable_bucketing=True``, and bucketing hoists the grouped MoE-expert tasks
    into dedicated leading buckets — which would break the group-major
    (pre / per-decoder-layer / post) contiguity the trainer engine's
    ``trainer_init`` validates. The non-bucketed export yields HF-canonical order,
    which ``layerwise_groups`` partitions cleanly. ``metadata()`` and iteration run
    the same export, so their order always agrees.

    Megatron only: this reads ``weight_extractor.bridge`` / ``.actor_module``.
    ``rdt_send`` selects it (vs ``_FsdpWeightSource``) by the presence of
    ``bridge`` on the extractor.
    """

    def __init__(self, weight_extractor: Any, dtype: torch.dtype) -> None:
        self._bridge = weight_extractor.bridge
        self._module = weight_extractor.actor_module
        self._dtype = dtype
        self._meta: Optional[List[ParamMeta]] = None

    def _export(self) -> Iterator[tuple]:
        # conversion_tasks=None -> full model in HF-canonical (group-contiguous)
        # order; the bridge gathers TP/PP/EP and yields full tensors on every rank.
        return self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=None)

    def metadata(self) -> List[ParamMeta]:
        # One collective dry-run export to learn names/shapes; tensors are
        # discarded. Runs once (trainer_init caches the result), so the extra
        # gather is a one-time init cost, not per-sync.
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._export():
                meta.append(ParamMeta(name, self._dtype, tuple(tensor.shape)))
                del tensor
            self._meta = meta
        return self._meta

    def __iter__(self) -> Iterator[tuple]:
        # A worker thread does not inherit the main thread's current CUDA device;
        # set it so the bridge's gather collectives + casts land on this rank's GPU.
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
            device = torch.cuda.current_device()
        else:
            device = None
        for name, tensor in self._export():
            full = tensor.to(device=device, dtype=self._dtype).detach().contiguous()
            yield name, full


class MegatronStackedWeightSource(WeightSource):
    """Megatron ``WeightSource`` with STACKED expert gathers (per-tensor-overhead fix).

    The plain ``MegatronWeightSource`` streams every HF tensor through the
    bridge, which runs its PP-broadcast + EP all-gather machinery **per
    exported tensor**. For per-expert MoE archs (qwen3_moe: 128 experts x 3
    projections per layer) that is ~400 small latency-bound collectives per
    layer and dominates sync time (~13ms/tensor measured; 235B => ~470s/sync).

    This source keeps the bridge for everything EXCEPT the per-expert weights
    (attention, norms, router, embeddings — via ``conversion_tasks`` filtering,
    the same subset-export idiom ``MegatronWeightExtractor`` uses for
    bucketing), and gathers the experts itself at STACK granularity — two big
    collectives per MoE layer instead of ~400:

      fc1 stack [n_local, 2F, H] --ep all_gather--> [E, 2F, H] --pp broadcast-->
      per-expert HF views: gate_proj = fc1[e, :F], up_proj = fc1[e, F:],
      down_proj = fc2[e]  (Megatron-Bridge qwen3_moe mapping: dim-0 [gate; up]
      chunk, fc2 verbatim, no transpose, dtype pass-through; global expert
      index e = n_local * ep_rank + i, matching all_gather rank order).

    Ordering contract: expert names are emitted immediately after the bridge's
    non-expert names of the same layer, so each ``model.layers.N.*`` block stays
    contiguous and ``layerwise_groups`` partitions the combined list cleanly.
    ``metadata()`` and ``__iter__`` derive from the same plan, so they always
    agree. Dense models have no expert tasks and degenerate to the plain
    filtered==full export.

    Falls back (see ``make_weight_source``) for grouped-export archs (qwen3.5
    style ``is_grouped_export`` mappings emit fused HF names — different
    contract) and can be disabled with ``SKYRL_RDT_STACKED_EXPERTS=0``.

    Set ``SKYRL_RDT_VERIFY_STACKED=1`` to numerically compare this source's
    expert tensors against the bridge's per-expert export for sampled layers on
    the first iteration (raises on mismatch; one-time cost of a few seconds).
    """

    _EXPERT_PRED = ".experts.linear_fc"  # model_bridge.py uses the same predicate

    def __init__(self, weight_extractor: Any, dtype: torch.dtype) -> None:
        self._bridge = weight_extractor.bridge
        self._module = weight_extractor.actor_module
        self._dtype = dtype
        self._meta: Optional[List[ParamMeta]] = None
        # {global_layer: {"fc1": [tasks by local expert idx], "fc2": [...],
        #                 "pp_rank": int, "n_local": int, "F": int, "H": int}}
        self._expert_layers: Optional[dict] = None
        self._verified = False

    # ---------------- task partitioning ----------------

    @staticmethod
    def _global_layer(global_param_name: str) -> int:
        # global_param_name like "decoder.layers.<N>.mlp.experts.linear_fc1.weight<E>"
        return int(global_param_name.split("layers.")[1].split(".")[0])

    @staticmethod
    def _local_expert(global_param_name: str) -> int:
        return int(global_param_name.rsplit("weight", 1)[1])

    def _partition_tasks(self, tasks: list) -> tuple:
        """Split conversion tasks into (non_expert_tasks, expert_layers dict)."""
        non_expert = []
        layers: dict = {}
        for t in tasks:
            if self._EXPERT_PRED not in t.param_name:
                non_expert.append(t)
                continue
            layer = self._global_layer(t.global_param_name)
            fc = "fc1" if ".linear_fc1." in t.param_name else "fc2"
            layers.setdefault(layer, {"fc1": [], "fc2": []})[fc].append(t)
        for layer, d in layers.items():
            d["fc1"].sort(key=lambda t: self._local_expert(t.global_param_name))
            d["fc2"].sort(key=lambda t: self._local_expert(t.global_param_name))
        return non_expert, layers

    def _expert_geometry(self, layers: dict) -> dict:
        """Exchange per-layer expert geometry across the PP group so EVERY rank
        knows the full set of MoE layers, their shapes, and the owning PP stage —
        regardless of whether get_conversion_tasks lists remote-stage params.
        Returns the merged {layer: {n_local,F,H,owner_pp_idx,ep_size,owned,...}}."""
        from megatron.core import parallel_state

        ep_size = torch.distributed.get_world_size(parallel_state.get_expert_model_parallel_group())
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_size = torch.distributed.get_world_size(pp_group)
        my_pp_idx = torch.distributed.get_rank(pp_group)

        local_geo: dict = {}
        for layer, d in layers.items():
            if d["fc1"] and d["fc1"][0].param_weight is not None:
                two_f, h = d["fc1"][0].param_weight.shape
                local_geo[layer] = (len(d["fc1"]), two_f // 2, h, my_pp_idx)

        if pp_size > 1:
            gathered: List[Optional[dict]] = [None] * pp_size
            torch.distributed.all_gather_object(gathered, local_geo, group=pp_group)
            merged_geo: dict = {}
            for g in gathered:
                merged_geo.update(g or {})
        else:
            merged_geo = local_geo

        merged: dict = {}
        for layer, (n_local, F, H, owner_pp_idx) in merged_geo.items():
            d = layers.get(layer, {"fc1": [], "fc2": []})
            owned = layer in local_geo
            merged[layer] = {
                **d,
                "owned": owned,
                "n_local": n_local,
                "F": F,
                "H": H,
                "owner_pp_idx": owner_pp_idx,
                "ep_size": ep_size,
            }
        return merged

    # ---------------- gather primitives ----------------

    def _gather_layer_stacks(self, d: dict) -> tuple:
        """EP all_gather + PP broadcast one layer's expert stacks.
        Returns (fc1_stack [E,2F,H], fc2_stack [E,H,F]) on EVERY rank."""
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
        device = torch.cuda.current_device()
        n_local, F, H = d["n_local"], d["F"], d["H"]
        E = n_local * d["ep_size"]

        fc1_stack = torch.empty((E, 2 * F, H), dtype=self._dtype, device=device)
        fc2_stack = torch.empty((E, H, F), dtype=self._dtype, device=device)
        if d["owned"]:
            ep_group = parallel_state.get_expert_model_parallel_group()
            fc1_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in d["fc1"]])
            fc2_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in d["fc2"]])
            torch.distributed.all_gather_into_tensor(
                fc1_stack.view(d["ep_size"], -1), fc1_local.reshape(-1), group=ep_group
            )
            torch.distributed.all_gather_into_tensor(
                fc2_stack.view(d["ep_size"], -1), fc2_local.reshape(-1), group=ep_group
            )
            del fc1_local, fc2_local

        if len(pp_ranks) > 1:
            src = pp_ranks[d["owner_pp_idx"]]
            torch.distributed.broadcast(fc1_stack, src=src, group=pp_group)
            torch.distributed.broadcast(fc2_stack, src=src, group=pp_group)
        return fc1_stack, fc2_stack

    def _yield_layer_experts(self, layer: int, d: dict) -> Iterator[tuple]:
        fc1_stack, fc2_stack = self._gather_layer_stacks(d)
        F = fc1_stack.shape[1] // 2
        E = fc1_stack.shape[0]
        prefix = f"model.layers.{layer}.mlp.experts"
        for e in range(E):
            yield f"{prefix}.{e}.gate_proj.weight", fc1_stack[e, :F].contiguous()
            yield f"{prefix}.{e}.up_proj.weight", fc1_stack[e, F:].contiguous()
            yield f"{prefix}.{e}.down_proj.weight", fc2_stack[e].contiguous()
        del fc1_stack, fc2_stack

    @staticmethod
    def _layer_of_hf_name(name: str) -> Optional[int]:
        if ".layers." not in name:
            return None
        try:
            return int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return None

    # ---------------- WeightSource interface ----------------

    def _iter_impl(self, collect_meta: bool) -> Iterator[tuple]:
        """Single generator driving both metadata() and __iter__ so their
        order agrees by construction. Interleaves stacked expert yields at the
        layer boundaries of the bridge's (filtered) non-expert stream."""
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
        # Fresh tasks each pass: mapping objects need clean PP-collective caches
        # (same reason MegatronWeightExtractor rebuilds them per sync).
        tasks = self._bridge.get_conversion_tasks(self._module)
        non_expert, layers = self._partition_tasks(tasks)
        layers = self._expert_geometry(layers)
        pending = dict(layers)  # layers whose experts are not yet emitted

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        prev_layer: Optional[int] = None
        for name, tensor in self._bridge.export_hf_weights(
            self._module, show_progress=False, conversion_tasks=non_expert
        ):
            layer = self._layer_of_hf_name(name)
            if prev_layer is not None and layer != prev_layer and prev_layer in pending:
                yield from self._yield_layer_experts(prev_layer, pending.pop(prev_layer))
            prev_layer = layer
            yield name, tensor.to(device=device, dtype=self._dtype).detach().contiguous()
        if prev_layer is not None and prev_layer in pending:
            yield from self._yield_layer_experts(prev_layer, pending.pop(prev_layer))
        # Layers whose non-expert tensors were all emitted before a boundary was
        # seen (or post-block orderings): emit any stragglers in layer order.
        for layer in sorted(pending):
            yield from self._yield_layer_experts(layer, pending.pop(layer))

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._iter_impl(collect_meta=True):
                meta.append(ParamMeta(name, self._dtype, tuple(tensor.shape)))
                del tensor
            self._meta = meta
        return self._meta

    def __iter__(self) -> Iterator[tuple]:
        if os.environ.get("SKYRL_RDT_VERIFY_STACKED") == "1" and not self._verified:
            self._verify_against_bridge()
            self._verified = True
        return self._iter_impl(collect_meta=False)

    # ---------------- one-time numeric verification ----------------

    def _verify_against_bridge(self) -> None:
        """Compare this source's expert tensors against the bridge's per-expert
        export for sampled layers. Collective: every rank must run it."""
        from megatron.core import parallel_state

        if torch.distributed.get_world_size(parallel_state.get_pipeline_model_parallel_group()) > 1:
            logger.warning("[stacked-verify] skipped: bridge re-export comparison requires pp=1")
            return
        tasks = self._bridge.get_conversion_tasks(self._module)
        _, layers = self._partition_tasks(tasks)
        if not layers:
            return
        layers = self._expert_geometry(layers)
        sample = sorted(layers)[:: max(1, (len(layers) - 1) // 2)][:3]  # first/mid/last
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for layer in sample:
            mine = dict(self._yield_layer_experts(layer, layers[layer]))
            expert_tasks = layers[layer]["fc1"] + layers[layer]["fc2"]
            for name, tensor in self._bridge.export_hf_weights(
                self._module, show_progress=False, conversion_tasks=expert_tasks
            ):
                ref = tensor.to(dtype=self._dtype)
                got = mine.get(name)
                if got is None:
                    raise RuntimeError(f"[stacked-verify] layer {layer}: missing {name}")
                if got.shape != ref.shape or not torch.equal(got, ref.to(got.device)):
                    raise RuntimeError(
                        f"[stacked-verify] layer {layer}: MISMATCH for {name} "
                        f"(shape {tuple(got.shape)} vs {tuple(ref.shape)})"
                    )
                del ref, tensor
            del mine
            if rank == 0:
                logger.info(f"[stacked-verify] layer {layer}: all expert tensors match bridge export")


def make_weight_source(weight_extractor: Any, dtype: torch.dtype) -> WeightSource:
    """Pick the RDT ``WeightSource`` for the trainer's weight extractor.

    The Megatron extractor exposes a Megatron-Bridge (``bridge`` /
    ``export_hf_weights``); the FSDP extractor exposes an all-gatherable
    ``state_dict`` (``model`` / ``_gather_tensor``). Selection is by the presence
    of ``bridge`` so neither backend module has to be imported here.

    For Megatron, prefer ``MegatronStackedWeightSource`` (stack-granularity
    expert gathers; ~20x fewer collectives on per-expert MoE archs) unless the
    arch uses grouped-export mappings (fused HF expert names — different
    contract) or ``SKYRL_RDT_STACKED_EXPERTS=0``.
    """
    if hasattr(weight_extractor, "bridge"):
        if os.environ.get("SKYRL_RDT_STACKED_EXPERTS", "1") != "0":
            try:
                tasks = weight_extractor.bridge.get_conversion_tasks(weight_extractor.actor_module)
                expert_tasks = [t for t in tasks if MegatronStackedWeightSource._EXPERT_PRED in t.param_name]
                grouped = any(getattr(t.mapping, "is_grouped_export", False) for t in expert_tasks)
                wrapped = any(".to_wrap." in t.global_param_name for t in expert_tasks)
                if expert_tasks and not grouped and not wrapped:
                    return MegatronStackedWeightSource(weight_extractor, dtype)
                if wrapped:
                    logger.info(
                        "[rdt] LoRA-wrapped experts: stacked source would serve UNMERGED "
                        "weights; using plain MegatronWeightSource (bridge merges adapters)"
                    )
                elif grouped:
                    logger.info("[rdt] grouped-export arch; using plain MegatronWeightSource")
                else:
                    logger.info("[rdt] no per-expert tasks (dense model); using plain MegatronWeightSource")
            except Exception:  # noqa: BLE001
                logger.warning("[rdt] stacked-source probe failed; using plain MegatronWeightSource", exc_info=True)
        return MegatronWeightSource(weight_extractor, dtype)
    return _FsdpWeightSource(weight_extractor, dtype)


class RdtWeightSyncSender:
    """Drives sharded-RDT weight sync through the vendored vLLM trainer-send
    engine, bypassing SkyRL's ``WeightTransferStrategy`` / ``WeightTransferSender``.

    Built once per worker in ``init_weight_sync_state``; ``send()`` is called from
    ``broadcast_to_inference_engines`` each RL step. The heavy ``trainer_init``
    (spawn the per-rank ``_RDTProducerServer`` sidecar + rank-0 bake) and every
    ``send_weights`` block on Ray + ``torch.distributed`` collectives, so both run
    off the worker's event loop via ``asyncio.to_thread``. The engine drives the
    inference side through the synchronous ``SyncRdtControlPlaneClient`` (blocking
    HTTP to the servers' ``/collective_rpc``), so the whole engine runs
    sync-to-sync in that worker thread with no event-loop involvement — see
    ``rdt_control_plane`` for why sidestepping the async ``RemoteInferenceClient``
    here is safe.
    """

    def __init__(
        self,
        inference_client: "RemoteInferenceClient",
        ie_cfg: "InferenceEngineConfig",
        inference_world_size: int,
        trainer_actor_namespace: Optional[str],
    ) -> None:
        if not inference_world_size:
            raise ValueError(
                f"sharded_rdt requires the inference world size (consumer count); got {inference_world_size!r}."
            )
        self._ie_cfg = ie_cfg
        self._world_size = int(inference_world_size)
        self._namespace = trainer_actor_namespace
        # Snapshot only what the sync control plane needs (server URLs + DP size);
        # the async client, its aiohttp session, and the event loop are NOT used.
        self._server_urls = list(inference_client.server_urls)
        self._data_parallel_size = int(inference_client.data_parallel_size)
        self._engine: Any = None
        self._control_plane: Optional[SyncRdtControlPlaneClient] = None

    def initialize(self, weight_extractor: Any) -> None:
        """Eagerly rendezvous (spawn sidecar servers + rank-0 bake) at
        ``init_weight_sync_state`` time, BEFORE any weight-gather collective can
        be in flight. Deferring this to the first ``send()`` deadlocks: rank 0
        blocks in the inference-side init RPC while the other ranks spin in the
        gather NCCL collectives waiting for it, and the producer servers (which
        share those ranks' GPUs) then can't finish NIXL agent creation —
        libfabric's fi_getinfo CUDA probe blocks behind the spinning kernels.
        Called with every rank inside ``init_weight_sync_state`` (followed by a
        barrier), ranks that finish early sit idle, so the window is empty.
        Blocking is fine here: ``init_weight_sync_state`` already blocks on a
        ``torch.distributed.barrier()``."""
        if weight_extractor is None:
            raise RuntimeError(
                "sharded_rdt weight sync requires the worker's weight_extractor " "(built in init_weight_sync_state)."
            )
        if self._engine is None:
            self._engine = self._trainer_init_blocking(weight_extractor)

    async def send(self, weight_extractor: Any) -> None:
        """Sync weights once; every rank must call it (the gather is a
        collective). ``initialize()`` should already have run; the lazy fallback
        here only covers callers that skipped it (and reintroduces the
        first-send deadlock risk described there — do not rely on it)."""
        if weight_extractor is None:
            raise RuntimeError(
                "sharded_rdt weight sync requires the worker's weight_extractor " "(built in init_weight_sync_state)."
            )
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._trainer_init_blocking, weight_extractor)
        await asyncio.to_thread(self._engine.send_weights)

    def _trainer_init_blocking(self, weight_extractor: Any) -> Any:
        """Build the WeightSource + control-plane client + trainer init info and
        rendezvous. Runs in a worker thread (blocks on the Ray spawn, an
        all-gather collective, and — on rank 0 — the inference-side bake)."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
            ShardedRDTTrainerInitInfo,
            ShardedRDTTrainerWeightTransferEngine,
        )

        # Constructed on every rank (the engine holds a client on all ranks), but
        # only the sender rank actually issues control-plane calls.
        self._control_plane = SyncRdtControlPlaneClient(self._server_urls, self._data_parallel_size)
        dtype = str_to_torch_dtype(self._ie_cfg.model_dtype)
        source = make_weight_source(weight_extractor, dtype)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        init_info = ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=self._world_size,
            trainer_actor_namespace=self._namespace,
            num_rdt_buffers=int(getattr(self._ie_cfg, "rdt_num_rdt_buffers", _DEFAULT_NUM_RDT_BUFFERS)),
            layerwise_split=int(getattr(self._ie_cfg, "rdt_layerwise_split", _DEFAULT_LAYERWISE_SPLIT)),
            arena_presize_gb=float(getattr(self._ie_cfg, "rdt_arena_presize_gb", _DEFAULT_ARENA_PRESIZE_GB)),
            pack_check=bool(getattr(self._ie_cfg, "rdt_pack_check", _DEFAULT_PACK_CHECK)),
            gather_lookahead=int(getattr(self._ie_cfg, "rdt_gather_lookahead", _DEFAULT_GATHER_LOOKAHEAD)),
        )
        return ShardedRDTTrainerWeightTransferEngine.trainer_init(
            init_info,
            client=self._control_plane,
            source=source,
        )

    def teardown(self) -> None:
        engine = self._engine
        control_plane = self._control_plane
        self._engine = None
        self._control_plane = None
        if engine is not None:
            engine.shutdown()
        if control_plane is not None:
            control_plane.close()
