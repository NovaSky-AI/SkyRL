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
import sys
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

import torch
from loguru import logger as _loguru

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


def _install_shared_pp_spec_cache() -> None:
    """Make Megatron-Bridge mapping PP-broadcast caches PROCESS-WIDE.

    ``MegatronParamMapping.broadcast_from_pp_rank`` learns each tensor's
    (shape, dtype, owner stage) via an ``all_gather_object`` over the PP group,
    cached per MAPPING INSTANCE. The RDT source rebuilds conversion tasks (and
    thus mappings) fresh every sync to keep live param references, so every
    non-expert tensor and every non-expert LoRA adapter paid that cross-node
    object-gather (~2-3 ms) EVERY sync (~2 s/sync at 235B). The cached spec is
    a static property of the global param name (shapes/dtypes/stage ownership
    never change across syncs; the actual WEIGHT broadcast still runs every
    sync, so trained values always flow) — so share ONE cache dict across all
    mapping instances, keyed as before by the global param name. Idempotent;
    disable with ``SKYRL_RDT_SHARED_SPEC_CACHE=0``."""
    # DEFAULT OFF (measured no benefit): the registry's mapping instances
    # persist across syncs, so their per-instance spec caches already survive —
    # the per-sync all_gather_object this was built to remove only ever ran
    # once per process. Sharing also caused a real bug (degenerate "None"
    # cache-key collision across adapters, fixed by namespacing). Kept only as
    # an experiment flag.
    if os.environ.get("SKYRL_RDT_SHARED_SPEC_CACHE", "0") == "0":
        return
    from megatron.bridge.models.conversion import param_mapping as _pm

    cls = _pm.MegatronParamMapping
    if getattr(cls, "_skyrl_shared_spec_cache", False):
        return
    shared_spec: dict = {}
    orig_init = cls.__init__
    orig_bcast_t = cls.broadcast_from_pp_rank

    def _init(self, *a, **k):
        orig_init(self, *a, **k)
        # Only the TENSOR spec cache is shared: specs are static per param.
        # Object broadcasts stay per-instance (unknown value lifetimes).
        self._tensor_spec_output_cache = shared_spec

    # Namespace cache keys by the mapping's (unique, stable) megatron param
    # name: with per-instance caches a degenerate key was harmless (e.g.
    # ``str(self.hf_param)`` is the string "None" for adapters whose HF name
    # resolution returns None), but in a SHARED cache it collides across
    # mappings and hands one tensor's spec (shape / tensor-parallel metadata)
    # to another — observed as a skipped TP gather in the LoRA merge.
    def _bcast_t(self, tensor, cache_key=None):
        if cache_key is not None:
            cache_key = f"{self.megatron_param}::{cache_key}"
        return orig_bcast_t(self, tensor, cache_key=cache_key)

    cls.__init__ = _init
    cls.broadcast_from_pp_rank = _bcast_t
    cls._skyrl_shared_spec_cache = True
    logger.info("[rdt] shared PP spec cache installed on MegatronParamMapping")


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
        _install_shared_pp_spec_cache()
        # One-time sliced-PP-gather topology self-check (see _gather_layer_stacks).
        self._pp_gather_checked = False
        # Batched non-expert export state (learned on the first, unbatched pass).
        self._nonexpert_layout: Optional[dict] = None
        self._nonexpert_owner: Optional[dict] = None
        self._nonexpert_checked = 0
        self._seen_first_layer = False
        self._cur_adapter_ctx: Optional[tuple] = None

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
            if ".adapter." in t.param_name:
                # LoRA adapter params (linear_in/linear_out). Never export them:
                # expert adapters are merged into the stacks; non-expert adapters
                # are merged by the bridge inside the filtered export.
                continue
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

    def _gather_layer_stacks(self, layer: int, d: dict) -> tuple:
        """Materialize one layer's full expert stacks (fc1 [E,2F,H], fc2 [E,H,F])
        on EVERY rank, with LoRA already merged when the model is wrapped.

        Sliced path (default, ``SKYRL_RDT_SLICED_GATHER=1``): the owner stage
        merges LoRA into its LOCAL expert shards (zero adapter collectives),
        PP-broadcasts only the local shards (1/ep_size of the stack — the old
        full-stack broadcast pushed ep_size identical copies of the same bytes
        across the PP link, ~38 GB/layer aggregate at 235B tp4/pp2/ep8), and
        then EVERY stage runs the intra-stage EP all_gather to reconstruct the
        full stacks. Correct because PP peers share (tp, dp) coordinates, so
        the pp-peer of expert-rank k IS expert-rank k of the owner stage —
        verified at runtime by a one-time self-check against the broadcast
        path (raises on mismatch; writes /tmp/rdt_profile/pp_gather_check.txt).

        Legacy path (``SKYRL_RDT_SLICED_GATHER=0``): owner-stage EP gather +
        full-stack PP broadcast + full-stack LoRA merge on all ranks.
        """
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)

        if os.environ.get("SKYRL_RDT_SLICED_GATHER", "1") == "0":
            fc1_stack, fc2_stack = self._gather_layer_stacks_broadcast(d)
            if d.get("adapter_ctx") is not None:
                mb, tasks_by_base = d["adapter_ctx"]
                self._merge_lora_into_stacks(layer, fc1_stack, fc2_stack, mb, tasks_by_base)
            return fc1_stack, fc2_stack

        if not self._pp_gather_checked and len(pp_ranks) > 1:
            self._pp_gather_checked = True
            self._selfcheck_sliced_gather(d)

        return self._finish_layer_gather(self._start_layer_gather(layer, d))

    def _start_layer_gather(self, layer: int, d: dict) -> tuple:
        """Issue one layer's shard broadcast + all-stage EP all_gather with
        ``async_op=True``. ``Work.wait()`` on NCCL work objects only inserts a
        stream dependency (no host block), so issuing layer L+1 here while
        layer L's tensors are being exported overlaps the expert collectives
        with the bridge's non-expert export. Single-thread issue in a
        deterministic layer order on every rank — no cross-rank reordering.
        The local shards must stay referenced until the gather completes, so
        they ride in the returned state."""
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
        ep_group = parallel_state.get_expert_model_parallel_group()
        device = torch.cuda.current_device()
        n_local, F, H = d["n_local"], d["F"], d["H"]
        E = n_local * d["ep_size"]

        fc1_local = torch.empty((n_local, 2 * F, H), dtype=self._dtype, device=device)
        fc2_local = torch.empty((n_local, H, F), dtype=self._dtype, device=device)
        if d["owned"]:
            fc1_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in d["fc1"]])
            fc2_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in d["fc2"]])
            if d.get("adapter_ctx") is not None:
                _mb, tasks_by_base = d["adapter_ctx"]
                self._merge_lora_into_local_shards(layer, fc1_local, fc2_local, tasks_by_base)
        if len(pp_ranks) > 1:
            src = pp_ranks[d["owner_pp_idx"]]
            # wait() = stream dependency so the EP gather reads the broadcast
            # result; host does not block.
            torch.distributed.broadcast(fc1_local, src=src, group=pp_group, async_op=True).wait()
            torch.distributed.broadcast(fc2_local, src=src, group=pp_group, async_op=True).wait()

        fc1_stack = torch.empty((E, 2 * F, H), dtype=self._dtype, device=device)
        fc2_stack = torch.empty((E, H, F), dtype=self._dtype, device=device)
        w1 = torch.distributed.all_gather_into_tensor(
            fc1_stack.view(d["ep_size"], -1), fc1_local.reshape(-1), group=ep_group, async_op=True
        )
        w2 = torch.distributed.all_gather_into_tensor(
            fc2_stack.view(d["ep_size"], -1), fc2_local.reshape(-1), group=ep_group, async_op=True
        )
        return (fc1_stack, fc2_stack, (w1, w2), (fc1_local, fc2_local))

    @staticmethod
    def _finish_layer_gather(st: tuple) -> tuple:
        fc1_stack, fc2_stack, works, _locals = st
        for w in works:
            w.wait()  # stream dependency into the current stream
        return fc1_stack, fc2_stack

    def _gather_layer_stacks_broadcast(self, d: dict) -> tuple:
        """Legacy gather: owner-stage EP all_gather + FULL-stack PP broadcast.
        Kept as the ``SKYRL_RDT_SLICED_GATHER=0`` fallback and as the reference
        for the sliced path's one-time self-check."""
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

    def _selfcheck_sliced_gather(self, d: dict) -> None:
        """One-time (first MoE layer, pp>1) runtime proof that the sliced
        shard-broadcast + all-stage EP gather reconstructs byte-identical
        stacks to the legacy full-stack broadcast — i.e. that pp-peer(ep-rank
        k) == ep-rank k of the owner stage on THIS topology. Raw weights only
        (no LoRA merge) so byte equality is exact. Collective on every rank.
        Raises on mismatch; success is recorded on the trainer node."""
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
        ep_group = parallel_state.get_expert_model_parallel_group()
        device = torch.cuda.current_device()
        n_local, F, H = d["n_local"], d["F"], d["H"]
        E = n_local * d["ep_size"]

        ref1, ref2 = self._gather_layer_stacks_broadcast(d)

        fc1_local = torch.empty((n_local, 2 * F, H), dtype=self._dtype, device=device)
        fc2_local = torch.empty((n_local, H, F), dtype=self._dtype, device=device)
        if d["owned"]:
            fc1_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in d["fc1"]])
            fc2_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in d["fc2"]])
        src = pp_ranks[d["owner_pp_idx"]]
        torch.distributed.broadcast(fc1_local, src=src, group=pp_group)
        torch.distributed.broadcast(fc2_local, src=src, group=pp_group)
        got1 = torch.empty((E, 2 * F, H), dtype=self._dtype, device=device)
        got2 = torch.empty((E, H, F), dtype=self._dtype, device=device)
        torch.distributed.all_gather_into_tensor(got1.view(d["ep_size"], -1), fc1_local.reshape(-1), group=ep_group)
        torch.distributed.all_gather_into_tensor(got2.view(d["ep_size"], -1), fc2_local.reshape(-1), group=ep_group)
        ok = bool(torch.equal(got1, ref1) and torch.equal(got2, ref2))
        try:
            os.makedirs("/tmp/rdt_profile", exist_ok=True)
            with open("/tmp/rdt_profile/pp_gather_check.txt", "a") as f:
                f.write(f"pid={os.getpid()} sliced_gather_matches_broadcast={ok}\n")
        except OSError:
            pass
        del ref1, ref2, got1, got2, fc1_local, fc2_local
        if not ok:
            raise RuntimeError(
                "[stacked-source] sliced PP gather self-check FAILED: pp-peer/ep-rank "
                "mapping does not hold on this topology; rerun with SKYRL_RDT_SLICED_GATHER=0"
            )

    def _merge_lora_into_local_shards(
        self, layer: int, fc1_local: torch.Tensor, fc2_local: torch.Tensor, tasks_by_base: dict
    ) -> None:
        """Merge ``base += (alpha/dim) * B @ A`` into this rank's LOCAL expert
        shards, BEFORE the shard broadcast/gather — so the merged bytes ride
        the same collectives as the base weights and the old per-layer adapter
        collectives (materialize PP bcast + EP all_gather, ~5 s/sync at 235B)
        disappear. Reads the adapter tasks' local ``param_weight`` directly:
        with etp==1 (guaranteed by make_weight_source) the bridge's
        materialize would return exactly these tensors on the owner stage.
        Same fp32-accumulate-then-cast rounding as the full-stack merge.
        Owner stage only (non-owner ranks receive merged shards)."""
        n_local = fc1_local.shape[0]
        for proj, local in (("linear_fc1", fc1_local), ("linear_fc2", fc2_local)):
            tasks = tasks_by_base.get(f"decoder.layers.{layer}.mlp.experts.{proj}")
            if not tasks:
                continue
            t = tasks[0]
            A = t.linear_in_task.param_weight
            B = t.linear_out_task.param_weight
            if A is None or B is None:
                raise RuntimeError(
                    f"[stacked-source] layer {layer} {proj}: owner stage is missing local "
                    "adapter weights; cannot local-merge (set SKYRL_RDT_SLICED_GATHER=0)"
                )
            # 3D = per-LOCAL-expert [n_local, ...]; 2D = shared across this
            # rank's local experts (share_expert_adapters) -> expand.
            A_loc = A.detach() if A.ndim > 2 else A.detach().unsqueeze(0).expand(n_local, *A.shape)
            B_loc = B.detach() if B.ndim > 2 else B.detach().unsqueeze(0).expand(n_local, *B.shape)
            delta = torch.bmm(B_loc.float(), A_loc.float()).mul_(t.alpha / t.dim)
            merged = local.float().add_(delta)
            local.copy_(merged.to(local.dtype))
            del A_loc, B_loc, delta, merged

    # ---------------- LoRA stack-merge (phase 2) ----------------

    def _adapter_tasks_by_base(self) -> tuple:
        """(model_bridge, {global_base_prefix: [AdapterWeightConversionTask]}).
        Built fresh per pass (clean PP-collective caches); contains an
        all_gather_object over the PP group, so every rank must call it at the
        same point. Empty dict when the model has no adapters."""
        from megatron.bridge.models.conversion.utils import unwrap_model

        mb = self._bridge._model_bridge
        return mb, mb.build_adapter_conversion_tasks(unwrap_model(self._module))

    @staticmethod
    def _per_expert_adapter(w: torch.Tensor, gathered: Optional[list], E: int) -> torch.Tensor:
        """Expand a materialized adapter weight to per-expert form [E, ...].
        3D weights are per-LOCAL-expert (concat ep-rank-major == global order);
        2D weights are shared across each rank's local experts (repeat)."""
        parts = gathered if gathered is not None else [w]
        if w.ndim > 2:
            return torch.cat(list(parts), dim=0)
        n_local = E // len(parts)
        return torch.stack(list(parts)).repeat_interleave(n_local, dim=0)

    def _merge_lora_into_stacks(
        self, layer: int, fc1_stack: torch.Tensor, fc2_stack: torch.Tensor, mb: Any, tasks_by_base: dict
    ) -> None:
        """Apply merged = base + (alpha/dim) * B @ A per expert, batched per
        projection (one bmm per layer/proj instead of the bridge's per-tensor
        merges). Semantics mirror LoRAMerge.merge: fp32 accumulate, cast back.
        Collective (materialize: PP bcast + ETP gather; EP all_gather): every
        rank runs it for every MoE layer in the same order."""
        for proj, stack in (("linear_fc1", fc1_stack), ("linear_fc2", fc2_stack)):
            tasks = tasks_by_base.get(f"decoder.layers.{layer}.mlp.experts.{proj}")
            if not tasks:
                continue
            aw = mb.materialize_adapter_weights(tasks)[0]
            A = aw.linear_in_weight.weight
            B = aw.linear_out_weight.weight
            gA = mb._gather_expert_adapter_weight(A)
            gB = mb._gather_expert_adapter_weight(B)
            E = stack.shape[0]
            A_all = self._per_expert_adapter(A, gA, E).to(stack.device)
            B_all = self._per_expert_adapter(B, gB, E).to(stack.device)
            delta = torch.bmm(B_all.float(), A_all.float()).mul_(aw.alpha / aw.dim)
            merged = stack.float().add_(delta)
            stack.copy_(merged.to(stack.dtype))
            del A_all, B_all, delta, merged

    def _yield_layer_experts(self, layer: int, d: dict, pending: Optional[dict] = None) -> Iterator[tuple]:
        fc1_stack, fc2_stack = self._gather_layer_stacks(layer, d)
        F = fc1_stack.shape[1] // 2
        E = fc1_stack.shape[0]
        prefix = f"model.layers.{layer}.mlp.experts"
        for e in range(E):
            yield f"{prefix}.{e}.gate_proj.weight", fc1_stack[e, :F].contiguous()
            yield f"{prefix}.{e}.up_proj.weight", fc1_stack[e, F:].contiguous()
            yield f"{prefix}.{e}.down_proj.weight", fc2_stack[e].contiguous()
        del fc1_stack, fc2_stack

    # ---------------- batched non-expert export ----------------

    @staticmethod
    @__import__("contextlib").contextmanager
    def _suppress_pp_broadcast():
        """Run bridge mapping code with PP broadcasts as identity (owner-stage
        ranks compute HF tensors LOCALLY; the packed per-group broadcast below
        is the only cross-stage traffic). Process-wide but the gather loop is
        the only exporter in this process."""
        from megatron.bridge.models.conversion import param_mapping as _pm

        cls = _pm.MegatronParamMapping
        orig_t = cls.broadcast_from_pp_rank
        orig_o = cls.broadcast_obj_from_pp_rank
        cls.broadcast_from_pp_rank = lambda self, tensor, cache_key=None: tensor
        cls.broadcast_obj_from_pp_rank = lambda self, obj, cache_key=None: obj
        try:
            yield
        finally:
            cls.broadcast_from_pp_rank = orig_t
            cls.broadcast_obj_from_pp_rank = orig_o

    def _nonexpert_group_key(self, global_param_name: str) -> str:
        if "layers." in global_param_name:
            try:
                return f"layer.{int(global_param_name.split('layers.')[1].split('.')[0])}"
            except (IndexError, ValueError):
                pass
        return "pre" if not self._seen_first_layer else "post"

    def _group_nonexpert_tasks(self, non_expert: list) -> list:
        """Consecutive (group_key, [tasks]) runs in task (HF-canonical) order."""
        groups: list = []
        self._seen_first_layer = False
        for t in non_expert:
            gk = self._nonexpert_group_key(t.global_param_name)
            if gk.startswith("layer."):
                self._seen_first_layer = True
            if groups and groups[-1][0] == gk:
                groups[-1][1].append(t)
            else:
                groups.append((gk, [t]))
        return groups

    def _record_nonexpert_stream(self, stream, non_expert: list):
        """First (unbatched) pass: pass tensors through while recording each
        group's yield layout [(hf_name, shape)], then compute per-group
        batchability + owner stage with ONE pp all_gather_object."""

        layout: dict = {}
        seen_layer = False
        for name, tensor in stream:
            lyr = self._layer_of_hf_name(name)
            if lyr is not None:
                seen_layer = True
                gk = f"layer.{lyr}"
            else:
                gk = "post" if seen_layer else "pre"
            layout.setdefault(gk, []).append((name, list(tensor.shape)))
            yield name, tensor

        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        groups = self._group_nonexpert_tasks(non_expert)
        mine = {gk: all(t.param_weight is not None for t in tasks) for gk, tasks in groups}
        gathered: list = [None] * torch.distributed.get_world_size(pp_group)
        torch.distributed.all_gather_object(gathered, mine, group=pp_group)
        owner: dict = {}
        for gk, _tasks in groups:
            owner[gk] = next((i for i, g in enumerate(gathered) if g and g.get(gk)), None)
        self._nonexpert_layout = layout
        self._nonexpert_owner = owner
        n_batchable = sum(1 for gk in owner if owner[gk] is not None and gk in layout)
        logger.info("[rdt] non-expert batching ready: %d/%d groups batchable", n_batchable, len(owner))

    def _batched_nonexpert_stream(self, non_expert: list):
        """Warm-sync non-expert export: the owner stage runs the bridge's
        standard export (TP gathers + transforms + non-expert LoRA merge)
        LOCALLY with PP broadcasts suppressed, packs each group's HF tensors
        into ONE buffer, and a single PP broadcast per group replaces the
        bridge's per-tensor broadcast + per-tensor spec object-gather.
        Groups with split ownership (e.g. tied embeddings) or missing layout
        fall back to the plain per-task export on every rank."""
        import math

        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
        my_pp_idx = torch.distributed.get_rank(pp_group)
        device = torch.cuda.current_device()

        for gk, tasks in self._group_nonexpert_tasks(non_expert):
            layout = self._nonexpert_layout.get(gk)
            owner = self._nonexpert_owner.get(gk)
            if layout is None or owner is None:
                # split/unknown ownership: plain export, all ranks lockstep
                yield from self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=tasks)
                continue
            numels = [math.prod(shape) if shape else 1 for _n, shape in layout]
            buf = torch.empty(sum(numels), dtype=self._dtype, device=device)
            if my_pp_idx == owner:
                with self._suppress_pp_broadcast():
                    out = self._owner_export_group(tasks)
                # merge_adapter_weights=False flips the persistent registry
                # mappings into HF-PEFT "unmerged" naming from their SECOND
                # export on: LoRA-wrapped bases come back as
                # ``X.base_layer.weight`` instead of ``X.weight``. Our explicit
                # merge already applied the adapters (keys unchanged), so fold
                # the alias back onto the layout's merged names.
                for alias in [n for n in out if ".base_layer.weight" in n]:
                    canonical = alias.replace(".base_layer.weight", ".weight")
                    if canonical not in out:
                        out[canonical] = out.pop(alias)
                missing = [n for n, _shape in layout if n not in out]
                if missing:
                    diag = {
                        "group": gk,
                        "missing": missing[:8],
                        "n_missing": len(missing),
                        "out_keys_sample": sorted(out)[:8],
                        "my_pp_idx": my_pp_idx,
                        "owner": owner,
                        "task_pw_none": {t.global_param_name: (t.param_weight is None) for t in tasks},
                        "export_trace": getattr(self, "_owner_export_trace", None),
                    }
                    try:
                        os.makedirs("/tmp/rdt_profile", exist_ok=True)
                        with open("/tmp/rdt_profile/nonexpert_fail.txt", "a") as f:
                            f.write(repr(diag) + "\n")
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"[stacked-source] batched non-expert export missed names for "
                        f"{gk}: {missing[:4]}... (diagnostics in /tmp/rdt_profile/"
                        "nonexpert_fail.txt; disable with SKYRL_RDT_BATCHED_NONEXPERT=0)"
                    )
                off = 0
                for (name, shape), n_el in zip(layout, numels):
                    buf[off : off + n_el].copy_(out[name].to(self._dtype).flatten())
                    off += n_el
                del out
            torch.distributed.broadcast(buf, src=pp_ranks[owner], group=pp_group)
            if self._nonexpert_checked < 2:
                self._nonexpert_checked += 1
                self._selfcheck_batched_nonexpert(gk, tasks, layout, numels, buf)
            off = 0
            for (name, shape), n_el in zip(layout, numels):
                yield name, buf[off : off + n_el].view(shape)
                off += n_el

    def _owner_export_group(self, tasks: list) -> dict:
        """Owner-stage, collective-free export of one group's tasks.

        Per task (mirroring the bridge's standard-path loop):
        ``export_hf_weights(conversion_tasks=[task], merge_adapter_weights=
        False)`` — the internal adapter build runs a PP ``all_gather_object``
        that would deadlock an owner-only call, so LoRA is merged explicitly
        from the per-pass adapter ctx (built lockstep in ``_iter_impl``).
        ``_merge_lora_adapter_weights`` merges into EVERY key of the dict it is
        given, so it must see one task's converted dict at a time. Caller holds
        the PP-broadcast suppression."""
        mm = self._module if isinstance(self._module, list) else [self._module]
        ctx = self._cur_adapter_ctx
        mb, tasks_by_base = ctx if ctx is not None else (None, {})
        matz_cache: dict = {}
        out: dict = {}
        self._owner_export_trace = []  # [(global_name, mapping_cls, module_none, pre_keys, post_keys, detected)]
        tasks = [self._clone_task_fresh_mapping(t) for t in tasks]
        for task in tasks:
            tdict = {
                n: t
                for n, t in self._bridge.export_hf_weights(
                    self._module,
                    show_progress=False,
                    conversion_tasks=[task],
                    merge_adapter_weights=False,
                )
            }
            _pre_keys = sorted(tdict)
            if ctx is not None and ".to_wrap.weight" in task.global_param_name:
                prefix = task.global_param_name.partition(".to_wrap.weight")[0]
                atasks = tasks_by_base.get(prefix)
                if atasks:
                    aws = matz_cache.get(prefix)
                    if aws is None:
                        aws = mb.materialize_adapter_weights(atasks)
                        matz_cache[prefix] = aws
                    tdict = mb._merge_lora_adapter_weights(mm, tdict, aws)
            self._owner_export_trace.append(
                (
                    task.global_param_name,
                    type(task.mapping).__name__,
                    task.megatron_module is None,
                    _pre_keys,
                    sorted(tdict),
                    repr(getattr(task.mapping, "_detected_type", "n/a")),
                    getattr(task.mapping, "_mapping", "n/a") is None,
                )
            )
            out.update(tdict)
        return out

    @staticmethod
    def _clone_task_fresh_mapping(task):
        """Clone a conversion task with a FRESH, stateless copy of its mapping.

        The registry's mapping instances are persistent, stateful (spec/obj
        caches, AutoMapping's detected type, lazily rewritten HF names) and
        collective-bearing. Driving them owner-only (suppressed) gives the two
        pipeline stages different call histories on the SAME objects, which
        desynced later all-rank exports three different ways (base_layer
        renames, crossed object broadcasts, gloo timeouts). Owner-local exports
        therefore run on throwaway clones; the persistent instances only ever
        see all-rank lockstep calls."""
        import copy
        import dataclasses

        m = copy.copy(task.mapping)
        m._tensor_spec_output_cache = {}
        m._broadcast_obj_cache = {}
        if hasattr(m, "_mapping"):
            m._mapping = None
        if hasattr(m, "_detected_type"):
            m._detected_type = None
        try:
            return dataclasses.replace(task, mapping=m)
        except TypeError:
            t2 = copy.copy(task)
            object.__setattr__(t2, "mapping", m)
            return t2

    def _selfcheck_batched_nonexpert(self, gk, tasks, layout, numels, buf) -> None:
        """One-time proof (first batched group) that the owner-computed packed
        broadcast is byte-identical to the bridge's plain per-tensor export.
        Collective on every rank; raises on mismatch."""
        import os as _os

        plain = {
            n: t for n, t in self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=tasks)
        }
        ok = True
        off = 0
        for (name, shape), n_el in zip(layout, numels):
            got = buf[off : off + n_el].view(shape)
            ref_t = plain.get(name)
            if ref_t is None:  # persistent-mapping rename (see alias fold above)
                ref_t = plain[name.replace(".weight", ".base_layer.weight")]
            ref = ref_t.to(dtype=self._dtype, device=got.device).reshape(shape)
            if not torch.equal(got, ref):
                ok = False
            off += n_el
        try:
            _os.makedirs("/tmp/rdt_profile", exist_ok=True)
            with open("/tmp/rdt_profile/pp_gather_check.txt", "a") as f:
                f.write(f"pid={_os.getpid()} batched_nonexpert[{gk}]_matches_plain={ok}\n")
        except OSError:
            pass
        if not ok:
            raise RuntimeError(
                f"[stacked-source] batched non-expert self-check FAILED for {gk}; "
                "rerun with SKYRL_RDT_BATCHED_NONEXPERT=0"
            )

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
        # LoRA-wrapped experts: raw to_wrap weights need the adapter delta
        # merged in (the bridge would have merged during its per-tensor export).
        wrapped = any(".to_wrap." in t.global_param_name for d in layers.values() for t in d.get("fc1", []))
        adapter_ctx = self._adapter_tasks_by_base() if wrapped else None
        # Also used by the batched non-expert export (built HERE, once per pass,
        # on EVERY rank in lockstep — the bridge's internal
        # build_adapter_conversion_tasks contains a PP all_gather_object, which
        # an owner-only export call must never trigger).
        self._cur_adapter_ctx = adapter_ctx
        for d in layers.values():
            d["adapter_ctx"] = adapter_ctx
        pending = dict(layers)  # layers whose experts are not yet emitted

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        from megatron.core import parallel_state as _ps

        _pp_size = torch.distributed.get_world_size(_ps.get_pipeline_model_parallel_group())
        # DEFAULT OFF: four distinct in-bridge state leaks (base_layer renames,
        # crossed object broadcasts, gloo collective timeouts, un-gathered qkv
        # reshape under cloned mappings) showed owner-only export cannot be
        # bolted onto the bridge's stateful, collective-bearing mappings from
        # outside. Measured upside when it worked: warm syncs 2.8-3.1s vs ~7s.
        # The clean path to re-enable is an owner-local export mode INSIDE
        # Megatron-Bridge (see multi_node_rdt.md X.10) — keep this as the
        # prototype behind the flag.
        _batch_ok = os.environ.get("SKYRL_RDT_BATCHED_NONEXPERT", "0") == "1" and _pp_size > 1 and not collect_meta
        if _batch_ok and self._nonexpert_layout is not None:
            _stream = self._batched_nonexpert_stream(non_expert)
        else:
            _stream = self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=non_expert)
            if _batch_ok:
                _stream = self._record_nonexpert_stream(_stream, non_expert)
        prev_layer: Optional[int] = None
        for name, tensor in _stream:
            layer = self._layer_of_hf_name(name)
            if prev_layer is not None and layer != prev_layer and prev_layer in pending:
                yield from self._yield_layer_experts(prev_layer, pending.pop(prev_layer), pending)
            prev_layer = layer
            yield name, tensor.to(device=device, dtype=self._dtype).detach().contiguous()
        if prev_layer is not None and prev_layer in pending:
            yield from self._yield_layer_experts(prev_layer, pending.pop(prev_layer), pending)
        # Layers whose non-expert tensors were all emitted before a boundary was
        # seen (or post-block orderings): emit any stragglers in layer order.
        for layer in sorted(pending):
            yield from self._yield_layer_experts(layer, pending.pop(layer), pending)

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._iter_impl(collect_meta=True):
                meta.append(ParamMeta(name, self._dtype, tuple(tensor.shape)))
                del tensor
            self._meta = meta
        return self._meta

    def __iter__(self) -> Iterator[tuple]:
        if os.environ.get("SKYRL_RDT_PROBE_ITER") == "1":
            raise RuntimeError("[stacked-source] PROBE: __iter__ executed (tripwire)")
        if not self._verified:
            armed = os.environ.get("SKYRL_RDT_VERIFY_STACKED") == "1"
            print(f"[stacked-source] iterate: verify={'ARMED' if armed else 'off'}", file=sys.stderr, flush=True)
            if armed:
                self._verify_against_bridge()
                print("[stacked-verify] PASSED: all sampled layers match bridge export", file=sys.stderr, flush=True)
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
        wrapped = any(".to_wrap." in t.global_param_name for d in layers.values() for t in d.get("fc1", []))
        adapter_ctx = self._adapter_tasks_by_base() if wrapped else None
        for d in layers.values():
            d["adapter_ctx"] = adapter_ctx
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
                ref_dev = ref.to(got.device)
                if got.shape != ref.shape:
                    raise RuntimeError(
                        f"[stacked-verify] layer {layer}: SHAPE MISMATCH for {name} "
                        f"({tuple(got.shape)} vs {tuple(ref.shape)})"
                    )
                if not torch.equal(got, ref_dev):
                    # Batched bmm (LoRA merge) can differ from the bridge's
                    # per-expert mm by reduction order; allow last-ulp noise.
                    max_diff = (got.float() - ref_dev.float()).abs().max().item()
                    if max_diff > 1e-2:
                        raise RuntimeError(
                            f"[stacked-verify] layer {layer}: MISMATCH for {name} (max|diff|={max_diff:.3e})"
                        )
                    if rank == 0:
                        logger.info(f"[stacked-verify] {name}: non-bitwise but close (max|diff|={max_diff:.3e})")
                del ref, tensor
            del mine
            if rank == 0:
                print(
                    f"[stacked-verify] layer {layer}: expert tensors match bridge export", file=sys.stderr, flush=True
                )


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
                etp_ok = True
                if wrapped:
                    # The batched stack merge assumes ETP==1 (the bridge's fused
                    # fc1 gate/up adapter split also forces tp_size=1 there).
                    try:
                        from megatron.core import parallel_state

                        etp_ok = (
                            torch.distributed.get_world_size(parallel_state.get_expert_tensor_parallel_group()) == 1
                        )
                    except Exception:  # noqa: BLE001
                        etp_ok = False
                if expert_tasks and not grouped and etp_ok:
                    if wrapped:
                        logger.info("[rdt] stacked expert source with LoRA stack-merge (etp=1)")
                    return MegatronStackedWeightSource(weight_extractor, dtype)
                if wrapped and not etp_ok:
                    logger.info(
                        "[rdt] LoRA-wrapped experts with ETP>1: adapter B-split ordering "
                        "unsupported; using plain MegatronWeightSource"
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
            _loguru.info("[rdt-profile] channel-check: loguru forwarding OK (rank init)")
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
        try:
            import json as _json
            import os as _os

            _prof = {"pid": _os.getpid()}
            try:
                _prof["rank"] = torch.distributed.get_rank()
            except Exception:  # noqa: BLE001
                pass
            if hasattr(self._engine, "get_sync_timing"):
                _prof["sync"] = self._engine.get_sync_timing()
            if hasattr(self._engine, "get_produce_timing"):
                _prof["produce"] = await asyncio.to_thread(self._engine.get_produce_timing)
            _os.makedirs("/tmp/rdt_profile", exist_ok=True)
            with open("/tmp/rdt_profile/trainer.jsonl", "a") as _f:
                _f.write(_json.dumps(_prof) + "\n")
        except Exception:  # noqa: BLE001 - profiling must never break the sync
            pass
        # Profiling: surface the vendored engine's per-round timing decomposition
        # (trainer-side sync buckets + producer-server counters) in the worker
        # log. Fail-soft — never let profiling break a sync.
        try:
            if hasattr(self._engine, "get_sync_timing"):
                sync_timing = self._engine.get_sync_timing()
                _loguru.info("[rdt-sync-timing] sync_timing=%s", sync_timing)
                print(f"[rdt-sync-timing] sync_timing={sync_timing}", file=sys.stderr, flush=True)
            if hasattr(self._engine, "get_produce_timing"):
                produce_timing = await asyncio.to_thread(self._engine.get_produce_timing)
                _loguru.info("[rdt-sync-timing] produce_timing=%s", produce_timing)
                print(f"[rdt-sync-timing] produce_timing={produce_timing}", file=sys.stderr, flush=True)
        except Exception:
            logger.exception("[rdt-sync-timing] failed to collect timing (ignored)")

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

        # Pipeline-depth knobs: env first (SKYRL_* env vars are forwarded into
        # every Ray worker by prepare_runtime_environment), then the ie_cfg
        # attribute if present, then the vLLM defaults. The env override exists
        # because the sync is a credit-limited latency pipeline: its steady-state
        # period is (publish->serve->pull->free RTT) / credits, so deepening
        # K/lookahead hides per-link latency without touching any link.
        def _knob(env: str, attr: str, default):
            v = os.environ.get(env)
            return v if v is not None else getattr(self._ie_cfg, attr, default)

        init_info = ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=self._world_size,
            trainer_actor_namespace=self._namespace,
            num_rdt_buffers=int(_knob("SKYRL_RDT_NUM_BUFFERS", "rdt_num_rdt_buffers", _DEFAULT_NUM_RDT_BUFFERS)),
            layerwise_split=int(_knob("SKYRL_RDT_LAYERWISE_SPLIT", "rdt_layerwise_split", _DEFAULT_LAYERWISE_SPLIT)),
            arena_presize_gb=float(
                _knob("SKYRL_RDT_ARENA_PRESIZE_GB", "rdt_arena_presize_gb", _DEFAULT_ARENA_PRESIZE_GB)
            ),
            pack_check=bool(getattr(self._ie_cfg, "rdt_pack_check", _DEFAULT_PACK_CHECK)),
            nosync=os.environ.get("SKYRL_RDT_NOSYNC") == "1",
            gather_lookahead=int(_knob("SKYRL_RDT_LOOKAHEAD", "rdt_gather_lookahead", _DEFAULT_GATHER_LOOKAHEAD)),
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
