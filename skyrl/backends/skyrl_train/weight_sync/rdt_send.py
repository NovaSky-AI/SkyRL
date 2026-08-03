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
import time
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
        # Per-layer expert HF name lists, built once (see _expert_names_for).
        self._expert_names: dict = {}
        # Cached PP-exchanged expert geometry (see _expert_geometry).
        self._geo_cache: Optional[dict] = None
        self._geo_sig: Optional[tuple] = None
        # Per-sync source phase timing (expert_gather / ne_* buckets), drained
        # into trainer.jsonl by RdtWeightSyncSender after each send.
        self._phase: dict = {}
        self._phase_prefix = ""

    def _phase_add(self, key: str, dt: float) -> None:
        # The metadata pass (trainer_init) runs the same walk as a real sync;
        # prefixing keeps its cost out of the warm buckets (it otherwise lands
        # in sync 0's record and makes ne_bridge look doubled).
        key = self._phase_prefix + key
        self._phase[key] = self._phase.get(key, 0.0) + dt

    def pop_phase_timing(self) -> dict:
        p, self._phase = self._phase, {}
        return p

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

        # Expert geometry is a property of the model + parallel layout, so it
        # cannot change between syncs — but the exchange is a PP
        # all_gather_object on the sync critical path (0.15s mean / 0.32s max
        # per sync at 235B). Cache it, keyed on this rank's local geometry so a
        # layout change still forces a rebuild. Every rank's local geometry is
        # stable, so all ranks take the same branch on every sync (a split
        # decision would deadlock in the collective below).
        sig = tuple(sorted(local_geo.items()))
        if self._geo_cache is not None and self._geo_sig == sig:
            merged_geo: dict = self._geo_cache
        elif pp_size > 1:
            gathered: List[Optional[dict]] = [None] * pp_size
            torch.distributed.all_gather_object(gathered, local_geo, group=pp_group)
            merged_geo = {}
            for g in gathered:
                merged_geo.update(g or {})
            self._geo_sig, self._geo_cache = sig, merged_geo
        else:
            merged_geo = local_geo
            self._geo_sig, self._geo_cache = sig, merged_geo

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

        The owner stage merges LoRA into its LOCAL expert shards (zero adapter
        collectives), PP-broadcasts only the local shards (1/ep_size of the
        stack — broadcasting the assembled stack instead would push ep_size
        identical copies of the same bytes across the PP link, ~38 GB/layer
        aggregate at 235B tp4/pp2/ep8), and then EVERY stage runs the
        intra-stage EP all_gather to reconstruct the full stacks.

        This is correct because PP peers share (tp, dp) coordinates, so the
        pp-peer of expert-rank k IS expert-rank k of the owner stage — a
        topology property, so it is verified at runtime by a one-time
        self-check against the assemble-then-broadcast reference
        (raises on mismatch; writes /tmp/rdt_profile/pp_gather_check.txt).
        """
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)

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
        """Reference gather: owner-stage EP all_gather + FULL-stack PP broadcast.

        Not used in the sync path — the sliced gather in ``_gather_layer_stacks``
        replaced it (same stacks, 1/ep_size of the PP traffic, no adapter
        collectives). Kept solely as the independent reference the sliced
        path's one-time self-check compares against, so a topology where
        pp-peer(ep-rank k) != ep-rank k is caught rather than silently
        producing wrong weights."""
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
                "mapping does not hold on this topology; run with "
                "SKYRL_RDT_STACKED_EXPERTS=0 to fall back to the bridge's "
                "per-expert export (much slower) and report this topology"
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
                    "adapter weights; cannot local-merge (set SKYRL_RDT_STACKED_EXPERTS=0)"
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

    def _expert_names_for(self, layer: int, E: int) -> list:
        """Per-layer expert HF names in yield order (gate/up/down per expert),
        built once and cached — otherwise ~36k f-strings per sync at 235B."""
        names = self._expert_names.get(layer)
        if names is None:
            prefix = f"model.layers.{layer}.mlp.experts"
            names = []
            for e in range(E):
                names.append(f"{prefix}.{e}.gate_proj.weight")
                names.append(f"{prefix}.{e}.up_proj.weight")
                names.append(f"{prefix}.{e}.down_proj.weight")
            self._expert_names[layer] = names
        return names

    def _extend_layer_experts(self, layer: int, d: dict, names: list, tensors: list) -> None:
        """Gather one layer's expert stacks and append per-expert entries.

        The per-expert tensors are views into the contiguous [E, 2F, H] /
        [E, H, F] stacks — each a contiguous slab (identical storage/offset/
        stride to the old per-expert ``fc1_stack[e, :F].contiguous()``, which
        was already a no-copy self-return). ``unbind`` batches the view
        creation: 3 dispatcher calls per layer instead of 3E ``__getitem__``
        + 3E no-op ``.contiguous()`` — the old path's main per-tensor cost."""
        _t0 = time.perf_counter()
        fc1_stack, fc2_stack = self._gather_layer_stacks(layer, d)
        self._phase_add("expert_gather", time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        F = fc1_stack.shape[1] // 2
        E = fc1_stack.shape[0]
        gates = fc1_stack[:, :F].unbind(0)
        ups = fc1_stack[:, F:].unbind(0)
        downs = fc2_stack.unbind(0)
        names.extend(self._expert_names_for(layer, E))
        for e in range(E):
            tensors.append(gates[e])
            tensors.append(ups[e])
            tensors.append(downs[e])
        self._phase_add("src_expert", time.perf_counter() - _t0)

    def _yield_layer_experts(self, layer: int, d: dict) -> Iterator[tuple]:
        """Per-(name, tensor) view of one layer's experts (verification path)."""
        names: list = []
        tensors: list = []
        self._extend_layer_experts(layer, d, names, tensors)
        return zip(names, tensors)

    @staticmethod
    def _layer_of_hf_name(name: str) -> Optional[int]:
        if ".layers." not in name:
            return None
        try:
            return int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return None

    # ---------------- WeightSource interface ----------------

    def _iter_groups_impl(self, collect_meta: bool) -> Iterator[tuple]:
        """Single group-batched walk driving metadata(), __iter__ AND
        iter_groups() so their order agrees by construction. Yields
        ``(names, tensors)`` parallel lists per layerwise group — pre /
        model.layers.N / post, the same contiguous partition
        ``layerwise_groups`` derives from metadata() — with each layer's
        stacked expert views appended at its boundary. Batching by group
        exists because the per-tensor handoff (~37k generator yields + per-name
        gather-loop bookkeeping per sync at 235B) cost ~0.9s of pure Python on
        the sync critical path."""
        self._phase_prefix = "meta_" if collect_meta else ""
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
        # Per-sync setup, timed separately: it runs once per walk but is NOT
        # per-yield work, so it would otherwise hide in the source_next
        # residual (measured 0.74s/sync at 235B — bigger than every per-tensor
        # cost combined).
        # Fresh tasks each pass: mapping objects need clean PP-collective caches
        # (same reason MegatronWeightExtractor rebuilds them per sync).
        _t0 = time.perf_counter()
        tasks = self._bridge.get_conversion_tasks(self._module)
        self._phase_add("src_setup_tasks", time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        non_expert, layers = self._partition_tasks(tasks)
        self._phase_add("src_setup_partition", time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        layers = self._expert_geometry(layers)
        self._phase_add("src_setup_geometry", time.perf_counter() - _t0)
        # LoRA-wrapped experts: raw to_wrap weights need the adapter delta
        # merged in (the bridge would have merged during its per-tensor export).
        # Task lists are global (identical on every rank), so this condition —
        # and the lockstep build below — cannot desync ranks.
        _t0 = time.perf_counter()
        wrapped = any(".to_wrap." in t.global_param_name for d in layers.values() for t in d.get("fc1", [])) or any(
            ".to_wrap." in t.global_param_name for t in non_expert
        )
        adapter_ctx = self._adapter_tasks_by_base() if wrapped else None
        self._phase_add("src_setup_adapter", time.perf_counter() - _t0)
        for d in layers.values():
            d["adapter_ctx"] = adapter_ctx
        pending = dict(layers)  # layers whose experts are not yet emitted

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        target_dev = torch.device("cuda", device) if device is not None else None
        _stream = self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=non_expert)
        # Manual next() so time INSIDE the bridge export ("ne_bridge": TP/PP
        # gathers, transforms, adapter merge — not our code) is split from our
        # dtype/device conversion ("ne_convert"). CPU-side wall time: kernel
        # launches are async, but the loop's pacing is CPU-bound, which is what
        # these buckets attribute.
        _it = iter(_stream)
        names: list = []
        tensors: list = []
        prev_layer: Optional[int] = None
        while True:
            _t0 = time.perf_counter()
            item = next(_it, None)
            self._phase_add("ne_bridge", time.perf_counter() - _t0)
            if item is None:
                break
            name, tensor = item
            layer = self._layer_of_hf_name(name)
            if names and layer != prev_layer:
                # Group boundary (pre -> layer 0, layer N -> N+1, last layer
                # -> post): append the closing layer's experts, then flush.
                if prev_layer is not None and prev_layer in pending:
                    self._extend_layer_experts(prev_layer, pending.pop(prev_layer), names, tensors)
                yield names, tensors
                names, tensors = [], []
            prev_layer = layer
            _t0 = time.perf_counter()
            # Warm steady state: bridge already yields target dtype on-device —
            # skip the no-op .to()/.contiguous() dispatches.
            if tensor.dtype != self._dtype or (target_dev is not None and tensor.device != target_dev):
                tensor = tensor.to(device=target_dev, dtype=self._dtype)
            tensor = tensor.detach()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self._phase_add("ne_convert", time.perf_counter() - _t0)
            names.append(name)
            tensors.append(tensor)
        if prev_layer is not None and prev_layer in pending:
            self._extend_layer_experts(prev_layer, pending.pop(prev_layer), names, tensors)
        if names:
            yield names, tensors
        # Layers whose non-expert tensors were all emitted before a boundary was
        # seen (or post-block orderings): emit any stragglers in layer order.
        for layer in sorted(pending):
            names, tensors = [], []
            self._extend_layer_experts(layer, pending.pop(layer), names, tensors)
            yield names, tensors

    def _iter_impl(self, collect_meta: bool) -> Iterator[tuple]:
        """Flattened per-(name, tensor) view of the group walk — used by
        metadata() and the fallback per-name __iter__."""
        for names, tensors in self._iter_groups_impl(collect_meta):
            yield from zip(names, tensors)

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._iter_impl(collect_meta=True):
                meta.append(ParamMeta(name, self._dtype, tuple(tensor.shape)))
                del tensor
            self._meta = meta
        return self._meta

    def _maybe_verify(self) -> None:
        if not self._verified:
            armed = os.environ.get("SKYRL_RDT_VERIFY_STACKED") == "1"
            print(f"[stacked-source] iterate: verify={'ARMED' if armed else 'off'}", file=sys.stderr, flush=True)
            if armed:
                self._verify_against_bridge()
                print("[stacked-verify] PASSED: all sampled layers match bridge export", file=sys.stderr, flush=True)
            self._verified = True

    def iter_groups(self) -> Iterator[tuple]:
        """Group-batched iteration for the gather loop: ``(names, tensors)``
        parallel lists per layerwise group, one generator handoff per group
        instead of per tensor. Optional WeightSource extension — the vendored
        gather loop uses it when present and falls back to __iter__."""
        self._maybe_verify()
        return self._iter_groups_impl(collect_meta=False)

    def __iter__(self) -> Iterator[tuple]:
        if os.environ.get("SKYRL_RDT_PROBE_ITER") == "1":
            raise RuntimeError("[stacked-source] PROBE: __iter__ executed (tripwire)")
        self._maybe_verify()
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
            _src = getattr(self._engine, "source", None)
            if _src is not None and hasattr(_src, "pop_phase_timing"):
                _prof["src_phase"] = _src.pop_phase_timing()
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
