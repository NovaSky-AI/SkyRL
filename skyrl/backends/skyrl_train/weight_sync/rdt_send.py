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
import contextlib
import cProfile
import logging
import os
import sys
import time
from dataclasses import dataclass
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
# Producer stall watchdog (seconds). Mirrors sharded_rdt_trainer.DEFAULT_STALL_TIMEOUT_S
# and InferenceFaultToleranceConfig.stall_timeout_s; duplicated rather than imported so
# resolving the knob does not pull the vendored trainer module into every worker.
_DEFAULT_STALL_TIMEOUT_S = 300.0


def _pp_local_requested() -> bool:
    """PP-local gather: each pipeline stage gathers and serves only its OWN layers
    instead of every stage gathering the whole model (see
    ``MegatronStackedWeightSource.owned_groups``).

    ON by default. It only engages at pp > 1, and it is self-healing: if any gather
    group turns out to be produced by more than one stage (tied embeddings, MTP),
    ``owned_groups`` logs and reverts the whole sync to gather-to-all. So the
    failure mode of defaulting it on is "no change", not a broken sync.

    Worth 15-25% of the 235B wall (multi_node_rdt.md X.20.4: warm mean 4.89s with
    it, against a 5.5-5.9s gather-to-all baseline on the same topology). It halves
    how many layers each stage gathers; it does NOT reduce peak memory, since a
    gathered layer's stack is still full-E and the resident set is still
    lookahead x per-layer.

    Set ``SKYRL_RDT_PP_LOCAL=0`` to force the gather-to-all path.
    """
    return os.environ.get("SKYRL_RDT_PP_LOCAL", "1") == "1"


def _qkv_device_fix_enabled() -> bool:
    """Keep the QKV split's index tensors on the weight's device (default on)."""
    return os.environ.get("SKYRL_RDT_QKV_DEVICE_FIX", "1") == "1"


@contextlib.contextmanager
def _qkv_index_device_ctx():
    """Build the QKV split's index tensors on the weight's device, not the host.

    ``split_qkv_weights`` and friends extract Q/K/V from Megatron's interleaved
    layout by indexing the (CUDA) packed weight with ``torch.arange`` index
    tensors, which default to CPU. Indexing a CUDA tensor with CPU indices forces
    a blocking H2D copy plus a stream sync per gather, so the host stalls for
    however much GPU work happens to be queued — measured at ~0.65 s per sync per
    rank of ``ne_bridge`` at 235B, and the producer is the sync's pacer, so it
    lands in the wall roughly 1:1.

    Rather than vendor the three functions (257 lines of interleave math that would
    then have to track upstream, and whose divergence would silently produce the
    WRONG Q/K/V split), this wraps them and defaults ``torch.arange``'s device to
    the weight's for the duration of one call. No upstream logic is copied, so an
    upstream change cannot desync us; and if upstream adopts the same fix, the
    ``setdefault`` simply stops mattering. The window is one function call in a
    single-threaded export, and the four ``arange`` calls inside each function are
    exactly the index tensors this is for.

    The in-tree fix is four characters — ``device=qkv.device`` on the aranges —
    and belongs upstream as a PR against megatron-bridge, since it taxes every
    megatron->hf export of every GQA model. Until that lands, reaching it from
    outside gets the same result with no forked copy to carry.
    """
    if not _qkv_device_fix_enabled():
        yield
        return

    import functools

    from megatron.bridge.models.conversion import param_mapping as _pm

    try:
        from megatron.bridge.models.conversion import peft_bridge as _peft
    except ImportError:  # pragma: no cover - peft is optional
        _peft = None

    names = ("split_qkv_weights", "split_qkv_biases", "split_qkv_weights_scale")
    targets = [m for m in (_pm, _peft) if m is not None]

    def _wrap(orig):
        @functools.wraps(orig)
        def wrapped(config, qkv, *args, **kwargs):
            dev = getattr(qkv, "device", None)
            if dev is None or dev.type == "cpu":
                return orig(config, qkv, *args, **kwargs)
            real_arange = torch.arange

            def _arange(*a, **kw):
                kw.setdefault("device", dev)
                return real_arange(*a, **kw)

            torch.arange = _arange
            try:
                return orig(config, qkv, *args, **kwargs)
            finally:
                torch.arange = real_arange

        return wrapped

    saved = []
    for mod in targets:
        for name in names:
            fn = getattr(mod, name, None)
            if fn is not None and not getattr(fn, "_skyrl_qkv_wrapped", False):
                new = _wrap(fn)
                new._skyrl_qkv_wrapped = True
                saved.append((mod, name, fn))
                setattr(mod, name, new)
    try:
        yield
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)


@contextlib.contextmanager
def _pp_local_export_ctx():
    """Export each stage's own parameters, with no cross-stage communication.

    ``megatron_to_hf`` normally starts by broadcasting the parameter from the
    pipeline stage that holds it to every other stage, so every rank ends up with
    every tensor. A stage that exports only its OWN tasks would then mismatch its
    peers' collectives and hang. Inside this context the two PP primitives return
    the local value instead: the owning stage gets its own tensor, every other
    stage gets ``None``, which every ``megatron_to_hf`` already treats as "not
    mine" and skips. TP and EP gathers are untouched — they run within a stage.

    Patched on the CLASS rather than on the tasks' mapping instances, for two
    reasons. Nothing in megatron-bridge overrides these two methods (they are
    defined once on ``MegatronParamMapping``), so one wrap covers every mapping
    including the ``_HFNameSuffixMapping`` wrapper, which reaches them through
    ``__getattr__``. And ``AutoMapping`` builds its delegate lazily INSIDE
    ``megatron_to_hf``, so an instance-level change — clearing ``pp_group`` to
    reach the upstream ``pp_size == 1`` fast path, say — would miss the delegate
    that actually performs the broadcast.

    Works against the OFFICIAL megatron-bridge; no fork required.
    """
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    saved = (
        MegatronParamMapping.broadcast_from_pp_rank,
        MegatronParamMapping.broadcast_obj_from_pp_rank,
    )

    def _local_tensor(self, tensor, cache_key=None):
        return tensor

    def _local_obj(self, obj, cache_key=None):
        return obj

    MegatronParamMapping.broadcast_from_pp_rank = _local_tensor
    MegatronParamMapping.broadcast_obj_from_pp_rank = _local_obj
    try:
        yield
    finally:
        (
            MegatronParamMapping.broadcast_from_pp_rank,
            MegatronParamMapping.broadcast_obj_from_pp_rank,
        ) = saved


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


@dataclass(frozen=True)
class _ExpertLayer:
    """One MoE layer's expert-stack plan, known on EVERY rank.

    Geometry is PP-exchanged (see ``_expert_geometry``) so a stage that does
    not hold this layer still knows the shapes and can take part in the
    collectives — every rank must issue every layer's gather in the same order.
    ``fc1``/``fc2`` are this rank's local expert conversion tasks, sorted by
    local expert index, and are empty on non-owning stages.
    """

    layer: int
    fc1: list
    fc2: list
    owned: bool
    """This rank's PP stage holds this layer's expert weights."""
    n_local: int
    """Experts per EP rank."""
    F: int
    """MoE FFN hidden size (fc1 packs [gate; up] as 2F rows)."""
    H: int
    """Model hidden size."""
    owner_pp_idx: int
    ep_size: int

    @property
    def E(self) -> int:
        """Global expert count."""
        return self.n_local * self.ep_size


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

    PP-local gather is ON by default (``SKYRL_RDT_PP_LOCAL=0`` disables it) and
    engages at pp>1: the source stops gathering across pipeline stages, so each
    stage yields only its own layers and declares them through ``owned_groups()``,
    and the RDT consumers route each layer's pull to a stage that holds it. See
    ``owned_groups`` for how ownership is discovered and when the mode falls back.
    """

    _EXPERT_PRED = ".experts.linear_fc"  # model_bridge.py uses the same predicate

    def __init__(self, weight_extractor: Any, dtype: torch.dtype) -> None:
        self._bridge = weight_extractor.bridge
        self._module = weight_extractor.actor_module
        self._dtype = dtype
        self._meta: Optional[List[ParamMeta]] = None
        self._verified = False
        # One-time sliced-PP-gather topology self-check (see _gather_layer_stacks).
        self._pp_gather_checked = False
        # Per-layer expert HF name lists, built once (see _expert_names_for).
        self._expert_names: dict = {}
        # Cached PP-exchanged expert geometry (see _expert_geometry).
        self._geo_cache: Optional[dict] = None
        self._geo_sig: Optional[tuple] = None
        # PP-local gather (SKYRL_RDT_PP_LOCAL=1, pp>1): this stage exports only
        # its own parameters. Resolved from the metadata pass — see owned_groups.
        self._pp_local = _pp_local_requested() and self._pp_geometry()[0] > 1
        self._group_stages: List[set] = []  # group idx -> stages that produce it
        self._owned_group_idx: List[int] = []
        self._group_index_of_name: dict = {}
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

    # ---------------- allocator probe ----------------
    #
    # Why: ~4.2s of ne_bridge is unaccounted for and is NOT collective latency
    # (NCCL calls do not block the host). The surviving hypothesis is real
    # cudaMalloc inside the sync. Weight sync runs with expandable_segments
    # forced OFF (VMM memory makes CUDA-IPC export 5-10x slower), so every
    # buffer the export allocates must come from a classic segment, and
    # worker.py only calls empty_cache ONCE per process on the assumption that
    # "later syncs re-use the classic blocks the first sync created". If the
    # intervening training step (which runs with expandable_segments back ON)
    # causes those blocks to be released, every sync re-creates them.
    #
    # segment.*.allocated is a cumulative count of cudaMalloc calls, so its
    # per-sync delta is a direct count of new segments. memory_reserved() is a
    # single cheap C call, so it can bracket the per-layer expert gathers
    # without perturbing their timing.

    _MEM_COUNTERS = (
        ("segment.all.allocated", "mem_segs_created"),
        ("segment.large_pool.allocated", "mem_segs_large"),
        ("allocation.all.allocated", "mem_allocs"),
        ("num_alloc_retries", "mem_retries"),
        ("num_ooms", "mem_ooms"),
    )

    @staticmethod
    def _mem_counters() -> dict:
        if not torch.cuda.is_available():
            return {}
        stats = torch.cuda.memory_stats()
        snap = {dst: float(stats.get(src, 0)) for src, dst in MegatronStackedWeightSource._MEM_COUNTERS}
        snap["mem_reserved_gb"] = float(stats.get("reserved_bytes.all.current", 0)) / 2**30
        return snap

    @staticmethod
    def _expandable_segment_counts() -> tuple:
        """(expandable, classic) live segment counts, from the allocator itself.

        Do NOT infer this from ``PYTORCH_CUDA_ALLOC_CONF``: worker.py toggles the
        setting through ``torch._C._accelerator_setAllocatorSettings`` and never
        touches the environment, so the env var says nothing about the live
        state. ``memory_snapshot`` reports each segment's ``is_expandable``,
        which is the only authoritative source. Called twice per sync, so its
        cost (proportional to live block count) stays off the per-tensor path.
        """
        try:
            segments = torch.cuda.memory_snapshot()
        except Exception:  # noqa: BLE001 - diagnostics must never break a sync
            return (-1.0, -1.0)
        expandable = sum(1 for s in segments if s.get("is_expandable"))
        return (float(expandable), float(len(segments) - expandable))

    def _mem_probe_start(self) -> dict:
        # Opt-in: the probe answered its question (allocation is NOT the ne_bridge
        # cost) and its two memory_snapshot() calls add ~1s+/sync of wall.
        if os.environ.get("SKYRL_RDT_MEM_PROBE") != "1":
            return {}
        snap = self._mem_counters()
        if snap:
            # Reserved bytes can only grow via new cudaMalloc segments UNLESS the
            # growth expands existing VMM segments. Recording both counts is what
            # distinguishes those two worlds, and the sync is supposed to run with
            # expandable segments OFF (they make CUDA-IPC export 5-10x slower).
            expandable, classic = self._expandable_segment_counts()
            self._mem_seg_at_start = (expandable, classic)
            self._phase_add("mem_seg_expandable_at_start", expandable)
            self._phase_add("mem_seg_classic_at_start", classic)
            self._phase_add("mem_reserved_gb_at_start", snap["mem_reserved_gb"])
        return snap

    def _mem_probe_end(self, start: dict) -> None:
        if not start:
            return
        end = self._mem_counters()
        expandable, classic = self._expandable_segment_counts()
        exp0, cls0 = getattr(self, "_mem_seg_at_start", (expandable, classic))
        self._phase_add("mem_seg_expandable_delta", expandable - exp0)
        self._phase_add("mem_seg_classic_delta", classic - cls0)
        for key, before in start.items():
            if key == "mem_reserved_gb":
                self._phase_add("mem_reserved_growth_gb", end[key] - before)
            else:
                self._phase_add(key, end[key] - before)

    def _dump_bridge_profile(self, prof: cProfile.Profile) -> None:
        """Persist this walk's accumulated ne_bridge cProfile.

        One file per rank per walk under /tmp/rdt_profile (fetched off the
        nodes by analyze_nebridge_prof.py); the ``meta_`` prefix marks the
        trainer-init metadata pass so warm syncs aggregate separately.
        """
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else os.getpid()
            walk = getattr(self, "_cprof_walk", 0)
            self._cprof_walk = walk + 1
            os.makedirs("/tmp/rdt_profile", exist_ok=True)
            prof.dump_stats(f"/tmp/rdt_profile/nebridge_{self._phase_prefix}rank{rank:03d}_walk{walk:02d}.prof")
        except Exception:  # noqa: BLE001 - diagnostics must never break a sync
            logging.getLogger(__name__).exception("ne_bridge profile dump failed")

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

    @staticmethod
    def _pp_geometry() -> tuple:
        """``(pp_size, pp_rank)`` for this rank, or ``(1, 0)`` without Megatron."""
        try:
            from megatron.core import parallel_state

            pp_group = parallel_state.get_pipeline_model_parallel_group()
        except Exception:  # noqa: BLE001 - no mpu (CPU tests, decentralized PGs)
            return 1, 0
        return torch.distributed.get_world_size(pp_group), torch.distributed.get_rank(pp_group)

    def _expert_geometry(self, layers: dict) -> dict:
        """Exchange per-layer expert geometry across the PP group so EVERY rank
        knows the full set of MoE layers, their shapes, and the owning PP stage —
        regardless of whether get_conversion_tasks lists remote-stage params.
        Returns ``{layer: _ExpertLayer}``.

        PP-local mode skips the exchange: a stage never gathers another stage's
        layers, so foreign geometry is not just unnecessary, it must not appear in
        the walk (the layers dict IS what the walk iterates)."""
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
        elif pp_size > 1 and not self._pp_local:
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
            merged[layer] = _ExpertLayer(
                layer=layer,
                fc1=d["fc1"],
                fc2=d["fc2"],
                owned=layer in local_geo,
                n_local=n_local,
                F=F,
                H=H,
                owner_pp_idx=owner_pp_idx,
                ep_size=ep_size,
            )
        return merged

    # ---------------- gather primitives ----------------

    def _gather_layer_stacks(self, lay: _ExpertLayer, adapter_ctx: Optional[tuple]) -> tuple:
        """Materialize one layer's full expert stacks (fc1 [E,2F,H], fc2 [E,H,F])
        with LoRA already merged when the model is wrapped.

        The owner stage merges LoRA into its LOCAL expert shards (zero adapter
        collectives), PP-broadcasts only the local shards (1/ep_size of the
        stack — broadcasting the assembled stack instead would push ep_size
        identical copies of the same bytes across the PP link, ~38 GB/layer
        aggregate at 235B tp4/pp2/ep8), and then EVERY stage runs the
        intra-stage EP all_gather to reconstruct the full stacks.

        PP-local mode drops the broadcast entirely: only the owning stage walks
        the layer, so the stacks are rebuilt from its own shards by the EP
        all_gather alone (expert-parallel groups never span stages — they vary
        only the ``ep`` axis of the rank layout). That also retires the topology
        assumption below, which exists solely because of the broadcast.

        This is correct because PP peers share (tp, dp) coordinates, so the
        pp-peer of expert-rank k IS expert-rank k of the owner stage — a
        topology property, so it is verified at runtime by a one-time
        self-check against the assemble-then-broadcast reference
        (raises on mismatch; writes /tmp/rdt_profile/pp_gather_check.txt).

        Every collective is issued with ``async_op=True`` and waited before
        return; for NCCL that wait only inserts a stream dependency (the host
        does not block), which is why the CPU-side cost of this method is
        ~0.08s/sync at 235B while the transfers themselves overlap later work.
        Layers are gathered one at a time in the same deterministic order on
        every rank — there is no cross-layer prefetch.
        """
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
        ep_group = parallel_state.get_expert_model_parallel_group()
        device = torch.cuda.current_device()
        n_local, F, H, E = lay.n_local, lay.F, lay.H, lay.E

        if not self._pp_gather_checked and len(pp_ranks) > 1 and not self._pp_local:
            self._pp_gather_checked = True
            self._selfcheck_sliced_gather(lay)

        fc1_local = torch.empty((n_local, 2 * F, H), dtype=self._dtype, device=device)
        fc2_local = torch.empty((n_local, H, F), dtype=self._dtype, device=device)
        if lay.owned:
            fc1_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in lay.fc1])
            fc2_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in lay.fc2])
            if adapter_ctx is not None:
                _mb, tasks_by_base = adapter_ctx
                self._merge_lora_into_local_shards(lay.layer, fc1_local, fc2_local, tasks_by_base)
        if len(pp_ranks) > 1 and not self._pp_local:
            src = pp_ranks[lay.owner_pp_idx]
            torch.distributed.broadcast(fc1_local, src=src, group=pp_group, async_op=True).wait()
            torch.distributed.broadcast(fc2_local, src=src, group=pp_group, async_op=True).wait()

        fc1_stack = torch.empty((E, 2 * F, H), dtype=self._dtype, device=device)
        fc2_stack = torch.empty((E, H, F), dtype=self._dtype, device=device)
        works = (
            torch.distributed.all_gather_into_tensor(
                fc1_stack.view(lay.ep_size, -1), fc1_local.reshape(-1), group=ep_group, async_op=True
            ),
            torch.distributed.all_gather_into_tensor(
                fc2_stack.view(lay.ep_size, -1), fc2_local.reshape(-1), group=ep_group, async_op=True
            ),
        )
        for w in works:
            w.wait()
        # fc1_local / fc2_local stay referenced until here, so the gathers
        # cannot read freed shards.
        return fc1_stack, fc2_stack

    def _gather_layer_stacks_broadcast(self, lay: _ExpertLayer) -> tuple:
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
        F, H, E = lay.F, lay.H, lay.E

        fc1_stack = torch.empty((E, 2 * F, H), dtype=self._dtype, device=device)
        fc2_stack = torch.empty((E, H, F), dtype=self._dtype, device=device)
        if lay.owned:
            ep_group = parallel_state.get_expert_model_parallel_group()
            fc1_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in lay.fc1])
            fc2_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in lay.fc2])
            torch.distributed.all_gather_into_tensor(
                fc1_stack.view(lay.ep_size, -1), fc1_local.reshape(-1), group=ep_group
            )
            torch.distributed.all_gather_into_tensor(
                fc2_stack.view(lay.ep_size, -1), fc2_local.reshape(-1), group=ep_group
            )
            del fc1_local, fc2_local

        if len(pp_ranks) > 1:
            src = pp_ranks[lay.owner_pp_idx]
            torch.distributed.broadcast(fc1_stack, src=src, group=pp_group)
            torch.distributed.broadcast(fc2_stack, src=src, group=pp_group)
        return fc1_stack, fc2_stack

    def _selfcheck_sliced_gather(self, lay: _ExpertLayer) -> None:
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
        n_local, F, H, E = lay.n_local, lay.F, lay.H, lay.E

        ref1, ref2 = self._gather_layer_stacks_broadcast(lay)

        fc1_local = torch.empty((n_local, 2 * F, H), dtype=self._dtype, device=device)
        fc2_local = torch.empty((n_local, H, F), dtype=self._dtype, device=device)
        if lay.owned:
            fc1_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in lay.fc1])
            fc2_local = torch.stack([t.param_weight.detach().to(self._dtype) for t in lay.fc2])
        src = pp_ranks[lay.owner_pp_idx]
        torch.distributed.broadcast(fc1_local, src=src, group=pp_group)
        torch.distributed.broadcast(fc2_local, src=src, group=pp_group)
        got1 = torch.empty((E, 2 * F, H), dtype=self._dtype, device=device)
        got2 = torch.empty((E, H, F), dtype=self._dtype, device=device)
        torch.distributed.all_gather_into_tensor(got1.view(lay.ep_size, -1), fc1_local.reshape(-1), group=ep_group)
        torch.distributed.all_gather_into_tensor(got2.view(lay.ep_size, -1), fc2_local.reshape(-1), group=ep_group)
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

    def _extend_layer_experts(
        self, lay: _ExpertLayer, adapter_ctx: Optional[tuple], names: list, tensors: list
    ) -> None:
        """Gather one layer's expert stacks and append per-expert entries.

        The per-expert tensors are views into the contiguous [E, 2F, H] /
        [E, H, F] stacks — each a contiguous slab (identical storage/offset/
        stride to the old per-expert ``fc1_stack[e, :F].contiguous()``, which
        was already a no-copy self-return). ``unbind`` batches the view
        creation: 3 dispatcher calls per layer instead of 3E ``__getitem__``
        + 3E no-op ``.contiguous()`` — the old path's main per-tensor cost."""
        # memory_reserved() is one cheap C call, unlike memory_stats(), so it can
        # bracket every layer without perturbing the timing below. Growth here is
        # segment creation attributable to OUR stacks (fc1 is 3.2GB at 235B), which
        # is what separates our allocator cost from the bridge's.
        _res0 = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        _t0 = time.perf_counter()
        fc1_stack, fc2_stack = self._gather_layer_stacks(lay, adapter_ctx)
        self._phase_add("expert_gather", time.perf_counter() - _t0)
        if _res0:
            self._phase_add("mem_expert_reserved_growth_gb", (torch.cuda.memory_reserved() - _res0) / 2**30)
        _t0 = time.perf_counter()
        F, E = lay.F, lay.E
        gates = fc1_stack[:, :F].unbind(0)
        ups = fc1_stack[:, F:].unbind(0)
        downs = fc2_stack.unbind(0)
        names.extend(self._expert_names_for(lay.layer, E))
        for e in range(E):
            tensors.append(gates[e])
            tensors.append(ups[e])
            tensors.append(downs[e])
        self._phase_add("src_expert", time.perf_counter() - _t0)

    def _yield_layer_experts(self, lay: _ExpertLayer, adapter_ctx: Optional[tuple]) -> Iterator[tuple]:
        """Per-(name, tensor) view of one layer's experts (verification path)."""
        names: list = []
        tensors: list = []
        self._extend_layer_experts(lay, adapter_ctx, names, tensors)
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

    def _build_export_plan(self) -> tuple:
        """Per-sync plan: ``(non_expert_tasks, {layer: _ExpertLayer}, adapter_ctx)``.

        Runs once per walk, before any tensor is produced. Timed in its own
        phase buckets because it is NOT per-yield work and would otherwise hide
        in the gather loop's ``source_next`` residual (measured 0.74 s/sync at
        235B — more than every per-tensor cost combined, of which
        ``get_conversion_tasks`` alone is ~0.5 s).

        Conversion tasks are rebuilt every sync on purpose: they hold live
        parameter references, and the bridge builds fresh mapping objects with
        clean PP-collective caches each call (the same reason
        ``MegatronWeightExtractor`` rebuilds them per sync).
        """
        _t0 = time.perf_counter()
        tasks = self._bridge.get_conversion_tasks(self._module)
        self._phase_add("src_setup_tasks", time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        non_expert, raw_layers = self._partition_tasks(tasks)
        self._phase_add("src_setup_partition", time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        layers = self._expert_geometry(raw_layers)
        self._phase_add("src_setup_geometry", time.perf_counter() - _t0)
        # LoRA-wrapped experts: raw to_wrap weights need the adapter delta
        # merged in (the bridge would have merged during its per-tensor export).
        _t0 = time.perf_counter()
        # Decided from the FULL task list, which get_conversion_tasks builds
        # identically on every rank: build_adapter_conversion_tasks is collective
        # (a cached PP all_gather_object), so a condition that could differ per
        # stage would desync. In PP-local mode `layers` is this stage's only, so
        # it must not take part in this decision.
        wrapped = any(".to_wrap." in t.global_param_name for t in tasks)
        adapter_ctx = self._adapter_tasks_by_base() if wrapped else None
        self._phase_add("src_setup_adapter", time.perf_counter() - _t0)
        return non_expert, layers, adapter_ctx

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
        _mem0 = self._mem_probe_start()
        # Deterministic CPU attribution for the ne_bridge bucket: the profiler is
        # enabled ONLY around next() into the bridge's export generator, so the
        # dump decomposes exactly ne_bridge (at ~1.3-2x overhead on that bucket).
        _prof = cProfile.Profile() if os.environ.get("SKYRL_RDT_CPROFILE") == "1" else None
        _gen = self._walk_groups(probe=bool(_mem0), prof=_prof)
        if self._pp_local and not collect_meta:
            # The partition only exists once metadata() has assembled it, which is
            # why the metadata pass itself is not reordered — it is reordered
            # wholesale by layerwise_groups afterwards.
            _gen = self._walk_in_group_order(_gen)
        if os.environ.get("SKYRL_RDT_GPU_PROBE") == "1" and torch.cuda.is_available():
            _gen = self._walk_with_gpu_events(_gen)
        try:
            yield from _gen
        finally:
            # The gather loop abandons this generator before StopIteration, so the
            # whole-walk allocator delta has to close here or it never records.
            self._mem_probe_end(_mem0)
            if _prof is not None:
                self._dump_bridge_profile(_prof)

    def _walk_with_gpu_events(self, gen: Iterator[tuple]) -> Iterator[tuple]:
        """GPU-timeline probe (``SKYRL_RDT_GPU_PROBE=1``).

        Records a timing event on the current stream after each group's work is
        enqueued. Every collective in the walk ``wait()``s into this stream, so
        event k completes only once group k's gathers have actually FINISHED on
        the GPU — which no host-side bucket can see (NCCL enqueues don't block).
        ``gpu_walk_span_s`` (anchor -> last event on the GPU timeline) is then
        directly comparable to ``gpu_host_span_s``: if it tracks the sync wall,
        the pipeline is gather-bound; if it tracks the host span, the wall lives
        in the publish/credit round-trip instead. ``gpu_groups_behind`` counts
        groups whose GPU work was still unfinished when the NEXT group was
        produced (host yields resume only after the previous group's publish
        was admitted, so this flags the GPU lagging the pipeline itself).
        """
        events: list = []
        anchor = torch.cuda.Event(enable_timing=True)
        anchor.record()
        t0 = time.perf_counter()
        behind = 0
        try:
            for names, tensors in gen:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                if events and not events[-1].query():
                    behind += 1
                events.append(ev)
                yield names, tensors
        finally:
            host_span = time.perf_counter() - t0
            if events:
                _t = time.perf_counter()
                torch.cuda.synchronize()
                self._phase_add("gpu_drain_at_close_s", time.perf_counter() - _t)
                spans = [events[i - 1].elapsed_time(events[i]) for i in range(1, len(events))]
                self._phase_add("gpu_walk_span_s", anchor.elapsed_time(events[-1]) / 1000.0)
                self._phase_add("gpu_host_span_s", host_span)
                self._phase_add("gpu_groups_behind", float(behind))
                if spans:
                    self._phase_add("gpu_group_ms_max", max(spans))
                    self._phase_add("gpu_group_ms_mean", sum(spans) / len(spans))

    def _walk_groups(self, probe: bool, prof: Optional[cProfile.Profile] = None) -> Iterator[tuple]:
        """The group-batched walk itself; see :meth:`_iter_groups_impl`.

        PP-local mode narrows the walk to this stage: only locally-held
        conversion tasks are exported, and the export is CONSUMED inside
        ``_pp_local_export_ctx()`` (the bridge returns a generator, so its
        collectives run while we iterate — entering the context only around the
        call that builds it would do nothing). ``layers`` is already local-only,
        since ``_expert_geometry`` skips the cross-stage exchange in this mode."""
        non_expert, layers, adapter_ctx = self._build_export_plan()
        pending = dict(layers)  # layers whose experts are not yet emitted
        if self._pp_local:
            non_expert = [t for t in non_expert if t.param_weight is not None]
            self._phase_add("pp_local", 1.0)
        self._phase_add("walk_ne_tasks", float(len(non_expert)))
        self._phase_add("walk_expert_layers", float(len(layers)))

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        target_dev = torch.device("cuda", device) if device is not None else None
        _ctx = contextlib.ExitStack()
        if self._pp_local:
            _ctx.enter_context(_pp_local_export_ctx())
        # Independent of PP-local, and the reason the megatron-bridge fork is no
        # longer needed for performance either (see _qkv_index_device_ctx).
        _ctx.enter_context(_qkv_index_device_ctx())
        if _qkv_device_fix_enabled():
            self._phase_add("qkv_device_fix", 1.0)
        _stream = self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=non_expert)
        # Manual next() so time INSIDE the bridge export ("ne_bridge": TP/PP
        # gathers, transforms, adapter merge — not our code) is split from our
        # dtype/device conversion ("ne_convert"). CPU-side wall time: kernel
        # launches are async, but the loop's pacing is CPU-bound, which is what
        # these buckets attribute.
        with _ctx:
            _it = iter(_stream)
            names: list = []
            tensors: list = []
            prev_layer: Optional[int] = None
            while True:
                _res0 = torch.cuda.memory_reserved() if probe else 0
                _t0 = time.perf_counter()
                if prof is not None:
                    prof.enable()
                item = next(_it, None)
                if prof is not None:
                    prof.disable()
                self._phase_add("ne_bridge", time.perf_counter() - _t0)
                if _res0:
                    self._phase_add("mem_bridge_reserved_growth_gb", (torch.cuda.memory_reserved() - _res0) / 2**30)
                if item is None:
                    break
                name, tensor = item
                layer = self._layer_of_hf_name(name)
                if names and layer != prev_layer:
                    # Group boundary (pre -> layer 0, layer N -> N+1, last layer
                    # -> post): append the closing layer's experts, then flush.
                    if prev_layer is not None and prev_layer in pending:
                        self._extend_layer_experts(pending.pop(prev_layer), adapter_ctx, names, tensors)
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
                self._extend_layer_experts(pending.pop(prev_layer), adapter_ctx, names, tensors)
            if names:
                yield names, tensors
            # Layers whose non-expert tensors were all emitted before a boundary
            # was seen (or post-block orderings): emit stragglers in layer order.
            for layer in sorted(pending):
                names, tensors = [], []
                self._extend_layer_experts(pending.pop(layer), adapter_ctx, names, tensors)
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
            self._meta = self._assemble_pp_metadata(meta) if self._pp_local else meta
        return self._meta

    def _assemble_pp_metadata(self, local: List[ParamMeta]) -> List[ParamMeta]:
        """Whole-model metadata from the per-stage PP-local walks.

        The RDT contract needs ``metadata()`` to describe the WHOLE model
        identically on every rank — the trainer engine cross-checks a digest of
        it, and the consumers' pull plans are built from the sender's copy alone —
        but a PP-local walk only covers this stage. So the stages exchange what
        each produced (one ``all_gather_object``, once, cached with the metadata)
        and every rank rebuilds the same list, ordered group-major by
        ``layerwise_groups`` rather than by stage, which is what the engine's
        group-contiguity check requires.

        The exchange doubles as the ownership probe: a PP-local export yields
        exactly the names its stage holds, so this is ownership at HF-NAME
        granularity — the only granularity that can tell whether one gather group
        is produced by two stages (tied embeddings, MTP). ``owned_groups`` acts on
        that.
        """
        _pp_size, my_pp = self._pp_geometry()
        gathered = self._exchange_pp_names([(m.name, list(m.shape)) for m in local])

        shapes: dict = {}
        stages_of: dict = {}
        for stage, entries in enumerate(gathered):
            for name, shape in entries or []:
                # A name two stages both produce (a tied weight) keeps the first
                # shape and records both stages, so its group reads as shared.
                shapes.setdefault(name, tuple(shape))
                stages_of.setdefault(name, set()).add(stage)
        groups = layerwise_groups(list(shapes))
        self._group_stages = [set().union(*(stages_of[n] for n in g)) for g in groups]
        self._owned_group_idx = [gi for gi, st in enumerate(self._group_stages) if my_pp in st]
        self._group_index_of_name = {n: gi for gi, g in enumerate(groups) for n in g}
        return [ParamMeta(n, self._dtype, shapes[n]) for g in groups for n in g]

    def _walk_in_group_order(self, gen: Iterator[tuple]) -> Iterator[tuple]:
        """Reorder a PP-local walk into the assembled partition's group order.

        The bridge streams a stage's tasks in ITS order, which need not agree with
        the partition: at 235B the last stage exports the output block before its
        layers, while ``layerwise_groups`` places that block last. The gather loop
        walks ``owned_groups()`` ascending and holds the source to it, so a group
        that arrives early is held back until its turn.

        Only the non-layer block is ever out of order, so at most a couple of
        groups are ever held. Holding a LAYER group would pin its gathered stacks
        (~4.6 GiB at 235B), so an unexpected permutation raises instead of
        quietly inflating trainer memory."""
        expect = list(self._owned_group_idx)
        held: dict = {}
        for names, tensors in gen:
            gi = self._group_index_of_name.get(names[0])
            if gi is None:
                raise RuntimeError(
                    f"PP-local walk yielded a group starting {names[0]!r}, which is "
                    "not in the assembled partition; metadata() and the walk disagree."
                )
            held[gi] = (names, tensors)
            while expect and expect[0] in held:
                yield held.pop(expect.pop(0))
            if len(held) > 2:
                raise RuntimeError(
                    f"PP-local walk is {len(held)} groups ahead of the partition order "
                    f"(holding {sorted(held)}, next expected {expect[:1]}); holding "
                    "gathered layer stacks would balloon trainer memory."
                )
        for gi in sorted(held):
            yield held.pop(gi)

    def _exchange_pp_names(self, mine: list) -> list:
        """All-gather ``[(name, shape)]`` over the PP group; one entry per stage."""
        from megatron.core import parallel_state

        pp_size, _my_pp = self._pp_geometry()
        if pp_size <= 1:
            return [mine]
        gathered: List[Optional[list]] = [None] * pp_size
        torch.distributed.all_gather_object(gathered, mine, group=parallel_state.get_pipeline_model_parallel_group())
        return gathered

    def owned_groups(self) -> Optional[List[int]]:
        """The gather groups this stage holds, or None to own everything.

        None (gather-to-all) unless PP-local mode is on AND every group turns out
        to be produced by exactly one stage. A group whose names come from two
        stages cannot be served PP-locally: each owner would publish only its half
        while the consumers' plan expects the whole group. Rather than truncate it,
        the whole sync reverts to gather-to-all — the layouts that do this (tied
        embeddings, MTP) are small models where the gather is not the bottleneck.
        Splitting the export per group instead would need a Megatron-task -> HF-group
        map the bridge does not expose.
        """
        if not self._pp_local:
            return None
        self.metadata()  # populates _group_stages / _owned_group_idx
        shared = [gi for gi, st in enumerate(self._group_stages) if len(st) > 1]
        if shared:
            logger.warning(
                "[stacked-source] PP-local gather disabled: gather groups %s are produced by "
                "more than one pipeline stage (tied embeddings / MTP), which cannot be served "
                "per-stage. Falling back to gather-to-all for the whole model.",
                shared[:4],
            )
            self._pp_local = False
            # The geometry cache holds this stage's layers only; a gather-to-all
            # walk needs every layer, and the cache key (local geometry) would
            # not change to force the rebuild.
            self._geo_cache = self._geo_sig = None
            return None
        return list(self._owned_group_idx)

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
        _, layers, adapter_ctx = self._build_export_plan()
        if not layers:
            return
        sample = sorted(layers)[:: max(1, (len(layers) - 1) // 2)][:3]  # first/mid/last
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for layer in sample:
            mine = dict(self._yield_layer_experts(layers[layer], adapter_ctx))
            expert_tasks = layers[layer].fc1 + layers[layer].fc2
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

    def _live_consumer_ids(self, live_server_urls: Optional[List[str]]) -> Optional[set]:
        """Map live server URLs to the consumer ids those servers own.

        Every index here comes from the PROVISIONED snapshot taken at construction —
        the consumer-id block of a server is a function of its position in that list,
        not of how many servers are currently alive. Re-deriving the geometry from a
        degraded list is the one genuinely dangerous mistake available on this path:
        it would silently renumber every surviving consumer, and each would then pull
        another consumer's slice with nothing downstream to notice.

        Returns ``None`` when the whole fleet is live, so the engine takes its
        historical (non-degraded) path exactly.
        """
        if live_server_urls is None:
            return None
        live = list(dict.fromkeys(live_server_urls))
        unknown = [u for u in live if u not in self._server_urls]
        if unknown:
            raise RuntimeError(
                f"live_server_urls contains URLs outside the provisioned set: {unknown}. "
                f"Provisioned: {self._server_urls}. The inference fleet geometry is frozen "
                "at provision; a new URL means something re-created a server, which Part 1 "
                "of engine fault tolerance does not support."
            )
        if len(live) == len(self._server_urls):
            return None
        if not live:
            raise RuntimeError("live_server_urls is empty: no inference server is alive to receive weights.")

        # replica ordinal -> its block of consumer ids. `num_replicas` is the
        # PROVISIONED deployment count, matching what the consumers were stamped with
        # at init (`replica_rank = index // dp`).
        num_replicas = len(self._server_urls) // self._data_parallel_size
        workers_per_replica = self._world_size // max(1, num_replicas)
        live_replicas = {self._server_urls.index(u) // self._data_parallel_size for u in live}
        return {rr * workers_per_replica + w for rr in live_replicas for w in range(workers_per_replica)}

    async def send(self, weight_extractor: Any, live_server_urls: Optional[List[str]] = None) -> None:
        """Sync weights once; every rank must call it (the gather is a
        collective). ``initialize()`` should already have run; the lazy fallback
        here only covers callers that skipped it (and reintroduces the
        first-send deadlock risk described there — do not rely on it).

        ``live_server_urls`` is the driver's current view of which inference servers
        are alive. ``None`` (the default) means the whole provisioned fleet, which is
        every non-fault-tolerant run. Every rank must be handed the SAME value —
        the ranks share gather collectives — which the driver guarantees by computing
        it once and dispatching it to all of them."""
        if weight_extractor is None:
            raise RuntimeError(
                "sharded_rdt weight sync requires the worker's weight_extractor " "(built in init_weight_sync_state)."
            )
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._trainer_init_blocking, weight_extractor)
        live_cids = self._live_consumer_ids(live_server_urls)
        if live_cids is not None:
            # A once-per-rank-per-sync summary alongside the vendored engine's
            # per-group `[rdt-degraded]`. Both land in SkyRL's node-local infra log
            # (/tmp/skyrl-logs/infra-*.log) -- NEITHER reaches the driver's stdout,
            # which is the log an operator is usually watching, so a degraded run
            # looks identical to a healthy one from there. Corrects an earlier note
            # here claiming the engine's line reached no log at all: it does, just
            # not where you would look first.
            _loguru.warning(
                f"[rdt-degraded] weight sync serving {len(live_cids)}/{self._world_size} live consumers "
                f"({len(live_server_urls or [])}/{len(self._server_urls)} inference servers alive); "
                f"groups routed to dead consumers are gathered and dropped"
            )
        if self._control_plane is not None:
            # Only rank 0 issues control-plane calls, but every rank holds a client
            # and setting this everywhere keeps the two views from drifting.
            self._control_plane.set_live(live_server_urls)
        await asyncio.to_thread(self._engine.send_weights, live_cids)
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
            # Resolved here, not in the sidecar: the sidecar is a Ray actor that
            # inherits the raylet's environment, so a launch-time SKYRL_* override
            # never reaches it. Env first, then the FT config, then the default.
            stall_timeout_s=float(
                os.environ.get("SKYRL_RDT_STALL_TIMEOUT_S")
                or getattr(getattr(self._ie_cfg, "fault_tolerance", None), "stall_timeout_s", None)
                or _DEFAULT_STALL_TIMEOUT_S
            ),
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
