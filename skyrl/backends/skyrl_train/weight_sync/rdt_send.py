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
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

import torch
from loguru import logger as _loguru

from skyrl.backends.skyrl_train.weight_sync.rdt_control_plane import (
    SyncRdtControlPlaneClient,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    WeightSource,
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
_DEFAULT_ARENA_PRESIZE_GB = 0.0
# Gathered-but-unfreed groups the trainer's gather loop runs ahead by; bounds
# trainer-resident memory at lookahead + 1 groups (see sharded_rdt_trainer.
# DEFAULT_GATHER_LOOKAHEAD). 1 = while the consumers pull group N, group N+1 is
# already gathered AND published; raise via SKYRL_RDT_LOOKAHEAD only if one
# group's gather is slower than its pulls.
_DEFAULT_GATHER_LOOKAHEAD = 1
# Producer stall watchdog (seconds). Mirrors sharded_rdt_trainer.DEFAULT_STALL_TIMEOUT_S
# and InferenceFaultToleranceConfig.stall_timeout_s; duplicated rather than imported so
# resolving the knob does not pull the vendored trainer module into every worker.
_DEFAULT_STALL_TIMEOUT_S = 300.0


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
    """One locally-held MoE layer's expert-stack plan.

    ``fc1``/``fc2`` are this rank's local expert conversion tasks, sorted by
    local expert index. The walk only ever reaches layers this stage holds
    (pp-local at pp>1, everything at pp==1), so ``owned`` is an assertable
    invariant rather than a mode.
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

    This source is single-mode — serve only what this rank holds — with two
    grains that engage independently but demote together: at pp>1 the source
    never gathers across pipeline stages (each stage yields only its own
    layers, declared through ``held_names()``), and at ep>1 it never
    all-gathers experts (only this coordinate's experts are materialized,
    also declared through ``held_names()``; foreign experts yield ``None``).
    The RDT consumers route every pull to a rank that holds the data. Layouts
    this cannot serve (a gather group produced by two stages: tied embeddings,
    MTP) demote the source, and ``make_weight_source`` delegates to the plain
    ``MegatronWeightSource`` — naive whole-model extraction, also reachable
    explicitly via ``SKYRL_RDT_STACKED_EXPERTS=0``.
    """

    _EXPERT_PRED = ".experts.linear_fc"  # model_bridge.py uses the same predicate

    def __init__(self, weight_extractor: Any, dtype: torch.dtype) -> None:
        self._bridge = weight_extractor.bridge
        self._module = weight_extractor.actor_module
        self._dtype = dtype
        self._meta: Optional[List[ParamMeta]] = None
        self._verified = False
        # Per-layer expert HF name lists, built once (see _expert_names_for).
        self._expert_names: dict = {}
        # The bridge's own mapping registry, cached: `mapping_registry()` rebuilds
        # the whole mapping list per call. Used ONLY for name lookup (a pure
        # string op), never to run a mapping, so it needs no process groups
        # installed on it.
        self._mapping_reg: Any = None
        self._expert_name_source: Optional[str] = None
        # [RDT-EXPORT-RING] Reused expert-stack buffers (see _stack_into_ring).
        # Depth reads the same knob as the trainer's gather_lookahead, so the
        # ring can never be shallower than the residency the credit gate allows.
        self._stack_ring: dict = {}
        self._stack_ring_pos = 0
        self._stack_ring_depth = max(1, int(os.environ.get("SKYRL_RDT_LOOKAHEAD", _DEFAULT_GATHER_LOOKAHEAD))) + 1
        self._stack_ring_on = os.environ.get("SKYRL_RDT_EXPORT_RING", "1") not in ("0", "false", "False")
        # This source is SINGLE-MODE: serve only what this rank holds. Two
        # grains that engage independently: PP-local (pp>1) — this stage
        # exports only its own parameters; EP-local (ep>1) —
        # only this coordinate's experts are materialized, foreign experts
        # yield None. Both are declared through held_names(). The escape hatch to naive whole-model
        # extraction is the plain MegatronWeightSource
        # (SKYRL_RDT_STACKED_EXPERTS=0), which make_weight_source also
        # delegates to automatically when a gather group spans pipeline stages
        # (tied embeddings, MTP) — see held_names / make_weight_source.
        etp = self._etp_size()
        if etp > 1:
            # Defense in depth: make_weight_source falls back to the plain
            # bridge source on ETP>1, so this should be unreachable. Shapes are
            # read off param_weight.shape and no rank holds a whole expert.
            raise RuntimeError(
                f"[stacked-source] expert_tensor_parallel_size={etp} is unsupported: "
                "no rank holds a whole expert. Use the plain bridge source "
                "(SKYRL_RDT_STACKED_EXPERTS=0) or set expert_tensor_parallel_size=1."
            )
        self._pp_local = self._pp_geometry()[0] > 1
        self._ep_size, self._my_ep_rank = self._ep_geometry()
        # ep_size==1 stays unstamped on purpose: every rank holds every expert,
        # and stamp 0 (!= -1) would only split each group into a pointless
        # second chunk on the consumer.
        self._ep_local = self._ep_size > 1
        # Set by held_names when a gather group spans pipeline stages: this
        # source cannot serve that layout, and make_weight_source delegates to
        # the plain MegatronWeightSource instead. Iteration refuses to run.
        self._demoted = False
        # Per-layer (F, H) recorded as layers are walked, so metadata() can
        # synthesize the shapes of foreign experts (their tensors are None).
        self._layer_geom: dict = {}
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

    @staticmethod
    def _ep_geometry() -> tuple:
        """``(ep_size, ep_rank)`` for this rank, or ``(1, 0)`` without Megatron."""
        try:
            from megatron.core import parallel_state

            ep_group = parallel_state.get_expert_model_parallel_group()
        except Exception:  # noqa: BLE001 - no mpu (CPU tests, decentralized PGs)
            return 1, 0
        return torch.distributed.get_world_size(ep_group), torch.distributed.get_rank(ep_group)

    @staticmethod
    def _etp_size() -> int:
        """Expert-tensor-parallel world size, or 1 when unprobeable (no mpu =>
        no ETP)."""
        try:
            from megatron.core import parallel_state

            return torch.distributed.get_world_size(parallel_state.get_expert_tensor_parallel_group())
        except Exception:  # noqa: BLE001 - no mpu (CPU tests, decentralized PGs)
            return 1

    def _expert_geometry(self, layers: dict) -> dict:
        """Per-layer expert geometry for the layers THIS rank holds, read off
        the local conversion-task weights. Returns ``{layer: _ExpertLayer}``.

        Local only, no communication: a stage never materializes another
        stage's layers (pp-local at pp>1; at pp==1 every layer is local), so
        foreign geometry must not appear in the walk — the layers dict IS what
        the walk iterates."""
        from megatron.core import parallel_state

        ep_size = torch.distributed.get_world_size(parallel_state.get_expert_model_parallel_group())

        merged: dict = {}
        for layer, d in layers.items():
            if not d["fc1"] or d["fc1"][0].param_weight is None:
                continue
            two_f, h = d["fc1"][0].param_weight.shape
            merged[layer] = _ExpertLayer(
                layer=layer,
                fc1=d["fc1"],
                fc2=d["fc2"],
                owned=True,
                n_local=len(d["fc1"]),
                F=two_f // 2,
                H=h,
                ep_size=ep_size,
            )
        return merged

    # ---------------- gather primitives ----------------

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

        mb = self._prepared_model_bridge()
        return mb, mb.build_adapter_conversion_tasks(unwrap_model(self._module))

    def _prepared_model_bridge(self):
        """A fresh ``_model_bridge`` with the AutoBridge's weights-backed
        ``hf_pretrained`` installed.

        The ``_model_bridge`` property hands each fresh bridge only the raw HF
        CONFIG, and ``build_conversion_tasks`` — the one call that installs the
        real ``hf_pretrained`` — is skipped on every path that supplies
        precomputed conversion tasks (ours all do) or builds adapter tasks.
        Architecture bridges whose ``mapping_registry`` inspects the CHECKPOINT
        then break: GLM-4.5's fused-expert probe reads
        ``hf_pretrained.state.source``, which a raw config lacks
        ("'Glm4MoeConfig' object has no attribute 'state'") — and its
        config-only fallback would guess fused=True, wrong for the per-expert
        zai-org checkpoints. Installing the wrapper up front reproduces what
        ``build_conversion_tasks`` does internally; bridges that never read it
        (Qwen, DeepSeek) are unaffected."""
        mb = self._bridge._model_bridge
        mb.hf_pretrained = self._bridge.hf_pretrained
        return mb

    @staticmethod
    def _mg_expert_template(sample_global_param_name: str) -> str:
        """``'<head>layers.{layer}.<tail>weight{e}'`` from a real expert task's
        Megatron name, e.g. ``decoder.layers.3.mlp.experts.linear_fc1.weight5``.

        Only the SHAPE of the name is taken from the sample; both indices are
        re-substituted, so the same template serves foreign experts this rank
        holds no task for.
        """
        head, _, rest = sample_global_param_name.partition("layers.")
        _layer, _, tail = rest.partition(".")
        base = tail.rsplit("weight", 1)[0]
        return head + "layers.{layer}." + base + "weight{e}"

    def _resolved_expert_names(self, lay: "_ExpertLayer", E: int) -> Optional[list]:
        """Expert HF names straight from the bridge's mapping registry, or None.

        ``megatron_to_hf_lookup`` resolves a Megatron param name to the CONCRETE
        HF name(s) — no model, no collective, pure string work — so whatever the
        architecture calls its decoder stack, its MoE container and its
        projections is what we emit. The fused fc1 resolves to a
        ``GatedMLPMapping`` whose ``hf_param`` names gate/up BY KEY, so the two
        halves are identified rather than assumed.

        Returns None (caller falls back) on any shape we do not recognize, since
        a wrong name here is SILENT: the consumer never bakes it, so it is
        skipped rather than raising.
        """
        try:
            # getattr: tests drive this source on instances built without __init__.
            reg = getattr(self, "_mapping_reg", None)
            if reg is None:
                reg = self._mapping_reg = self._prepared_model_bridge().mapping_registry()
            t1 = self._mg_expert_template(lay.fc1[0].global_param_name)
            t2 = self._mg_expert_template(lay.fc2[0].global_param_name)
            names: list = []
            for e in range(E):
                fc1 = getattr(reg.megatron_to_hf_lookup(t1.format(layer=lay.layer, e=e)), "hf_param", None)
                fc2 = getattr(reg.megatron_to_hf_lookup(t2.format(layer=lay.layer, e=e)), "hf_param", None)
                if not isinstance(fc1, dict) or "gate" not in fc1 or "up" not in fc1 or not isinstance(fc2, str):
                    return None
                names.append(fc1["gate"])
                names.append(fc1["up"])
                names.append(fc2)
            return names
        except Exception:  # noqa: BLE001 - any registry shape we cannot read -> fall back
            logger.warning("[rdt] expert-name lookup via the bridge failed; using the legacy names", exc_info=True)
            return None

    def _expert_names_for(self, lay: "_ExpertLayer", E: int) -> list:
        """Per-layer expert HF names in yield order (gate/up/down per expert),
        built once and cached — otherwise ~36k lookups per sync at 235B.

        Resolved through the bridge (see ``_resolved_expert_names``), falling
        back to the legacy synthesis. That fallback assumed
        ``model.layers.<L>.mlp.experts.<e>.{gate,up,down}_proj.weight`` — the
        Qwen3-MoE family's naming, which GLM and DeepSeek share and which
        Kimi K2.5-VL does NOT: it nests the stack under ``language_model.``, so
        every expert name missed and the experts silently never synced.
        Verified on CPU that the bridge reproduces the legacy names exactly for
        Qwen3-MoE, so this is a no-op wherever it already worked.
        """
        names = self._expert_names.get(lay.layer)
        if names is not None:
            return names
        names = self._resolved_expert_names(lay, E)
        source = "bridge mapping_registry"
        if names is None:
            prefix = f"model.layers.{lay.layer}.mlp.experts"
            names = []
            for e in range(E):
                names.append(f"{prefix}.{e}.gate_proj.weight")
                names.append(f"{prefix}.{e}.up_proj.weight")
                names.append(f"{prefix}.{e}.down_proj.weight")
            source = "legacy synthesis (model.layers.* assumed)"
        if getattr(self, "_expert_name_source", None) is None:
            # Positive marker, once: which naming path a run actually took.
            self._expert_name_source = source
            logger.info("[rdt-config] expert HF names from: %s", source)
        self._expert_names[lay.layer] = names
        return names

    def _stack_into_ring(self, kind: str, src: list) -> torch.Tensor:
        """``torch.stack(src)`` into a reused buffer instead of a fresh one.

        Every decoder layer has identical stack shapes, so a ring of
        ``lookahead + 1`` buffers serves the whole walk: the trainer then exports
        ONE CUDA-IPC handle per buffer rather than two per layer. Memory-neutral
        — ``torch.stack`` was allocating this tensor anyway. Falls back to a
        plain stack on any mismatch: correct-and-slower, never wrong bytes.
        """
        if not getattr(self, "_stack_ring_on", False) or not src:
            return torch.stack(src)
        shape = (len(src),) + tuple(src[0].shape)
        key = (kind, getattr(self, "_stack_ring_pos", 0) % max(1, getattr(self, "_stack_ring_depth", 2)))
        buf = self._stack_ring.get(key)
        if buf is None or tuple(buf.shape) != shape or buf.dtype != src[0].dtype or buf.device != src[0].device:
            buf = torch.empty(shape, dtype=src[0].dtype, device=src[0].device)
            self._stack_ring[key] = buf
        torch.stack(src, out=buf)
        return buf

    def _local_layer_stacks(self, lay: _ExpertLayer, adapter_ctx: Optional[tuple]) -> tuple:
        """This rank's LOCAL expert stacks (fc1 [n_local,2F,H], fc2 [n_local,H,F]),
        LoRA already merged — the shard-aware expert step. ZERO collectives: the
        stacks come straight off the local conversion-task weights, and the
        per-expert views handed out are slices of them, so foreign experts are
        never materialized at all. Shard-aware walks only owned layers (pp-local
        at pp>1, everything at pp==1), so ``lay`` is always locally held."""
        assert lay.owned, "shard-aware walk reached a layer this stage does not hold"
        fc1_local = self._stack_into_ring("fc1", [t.param_weight.detach().to(self._dtype) for t in lay.fc1])
        fc2_local = self._stack_into_ring("fc2", [t.param_weight.detach().to(self._dtype) for t in lay.fc2])
        self._stack_ring_pos = getattr(self, "_stack_ring_pos", 0) + 1
        if adapter_ctx is not None:
            _mb, tasks_by_base = adapter_ctx
            self._merge_lora_into_local_shards(lay.layer, fc1_local, fc2_local, tasks_by_base)
        return fc1_local, fc2_local

    def _extend_layer_experts(
        self, lay: _ExpertLayer, adapter_ctx: Optional[tuple], names: list, tensors: list
    ) -> None:
        """Produce one layer's expert stacks and append per-expert entries.

        Only the LOCAL stacks are built — zero collectives — and only this
        coordinate's ``n_local`` experts get real tensors; every foreign
        expert's three entries are ``None``. The NAME list always covers all E
        experts, so the gather loop's group-order check is rank-uniform and
        the trainer drops the ``None``s before publishing. At ep==1 the local
        window IS the full expert range, so no ``None`` is ever emitted — the
        degenerate case needs no branch.

        The per-expert tensors are views into contiguous stacks —
        each a contiguous slab (identical storage/offset/stride to the old
        per-expert ``fc1_stack[e, :F].contiguous()``, which was already a
        no-copy self-return). ``unbind`` batches the view creation: 3
        dispatcher calls per layer instead of 3E ``__getitem__`` + 3E no-op
        ``.contiguous()`` — the old path's main per-tensor cost."""
        # Recorded BEFORE the expert names are emitted, so metadata() can
        # synthesize foreign experts' shapes (their tensors are None).
        self._layer_geom[lay.layer] = (lay.F, lay.H)
        _t0 = time.perf_counter()
        fc1_local, fc2_local = self._local_layer_stacks(lay, adapter_ctx)
        self._phase_add("expert_gather", time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        F, E = lay.F, lay.E
        gates = fc1_local[:, :F].unbind(0)
        ups = fc1_local[:, F:].unbind(0)
        downs = fc2_local.unbind(0)
        names.extend(self._expert_names_for(lay, E))
        # At ep==1, lo == 0 and n_local == E: every expert is local.
        lo = lay.n_local * self._my_ep_rank
        for e in range(E):
            if lo <= e < lo + lay.n_local:
                i = e - lo
                tensors.append(gates[i])
                tensors.append(ups[i])
                tensors.append(downs[i])
            else:
                tensors.extend((None, None, None))
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
        if self._demoted:
            raise RuntimeError(
                "[stacked-source] this source was demoted (a gather group spans "
                "pipeline stages) and cannot iterate; make_weight_source "
                "delegates to the plain MegatronWeightSource automatically."
            )
        self._phase_prefix = "meta_" if collect_meta else ""
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
        _gen = self._walk_groups()
        if self._pp_local and not collect_meta:
            # The partition only exists once metadata() has assembled it, which is
            # why the metadata pass itself is not reordered — it is reordered
            # wholesale by layerwise_groups afterwards.
            _gen = self._walk_in_group_order(_gen)
        yield from _gen

    def _walk_groups(self) -> Iterator[tuple]:
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
        # Not export_hf_weights: with precomputed conversion_tasks it hands the
        # stream to a fresh CONFIG-ONLY model bridge (see _prepared_model_bridge)
        # whose adapter-merge pass re-derives the mapping registry.
        # cpu=False is REQUIRED and is why this cannot be defaulted: the
        # AutoBridge wrapper's default is False, but the underlying
        # stream_weights_megatron_to_hf defaults to cpu=True, which D2H-copies
        # every exported tensor and then forces our own .cuda() to copy it
        # straight back (measured at GLM-4.5-Air: ~9s/sync of pure PCIe
        # round-trip on the sync critical path).
        _stream = self._prepared_model_bridge().stream_weights_megatron_to_hf(
            self._module,
            self._bridge.hf_pretrained,
            cpu=False,
            show_progress=False,
            conversion_tasks=non_expert,
        )
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
        # [RDT-EXPORT-RING] Drop our refs so the buffers (a layer's experts each,
        # hundreds of MB) do not survive into the training step. The consumer of
        # this generator still holds the last groups' views, so nothing in flight
        # is freed here.
        self._stack_ring = {}
        self._stack_ring_pos = 0

    def _iter_impl(self, collect_meta: bool) -> Iterator[tuple]:
        """Flattened per-(name, tensor) view of the group walk — used by
        metadata() and the fallback per-name __iter__."""
        for names, tensors in self._iter_groups_impl(collect_meta):
            yield from zip(names, tensors)
            # Drop the group before the `for` asks for the next one: until the
            # loop variables are REBOUND they still reference this group, so the
            # next group would be gathered while this one is still alive.
            names = tensors = None

    def _foreign_expert_shape(self, name: str) -> tuple:
        """Shape of a foreign expert's entry, synthesized from local geometry.

        Foreign experts are never materialized (their yielded tensors are
        ``None``), so their shapes cannot be read off a tensor. They are uniform
        per layer — gate/up ``[F, H]``, down ``[H, F]`` — and the walk records
        each layer's ``(F, H)`` before that layer's expert names are emitted, so
        every ``None`` this rank yields belongs to a layer it walked."""
        F, H = self._layer_geom[self._layer_of_hf_name(name)]
        return (H, F) if ".down_proj." in name else (F, H)

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._iter_impl(collect_meta=True):
                shape = self._foreign_expert_shape(name) if tensor is None else tuple(tensor.shape)
                meta.append(ParamMeta(name, self._dtype, shape))
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
        is produced by two stages (tied embeddings, MTP). ``held_names`` acts on
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
        walks the held groups ascending and holds the source to it, so a group
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
            # `held` is the legitimate hold (bounded to 2 below); the loop
            # variables are not. Un-bind them so a group already yielded is not
            # ALSO pinned while the next one is gathered -- the same ~4.6 GiB of
            # stacks this function raises about holding.
            names = tensors = None
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

    def held_names(self) -> Optional[List[str]]:
        """The parameter names this rank holds — see the ABC.

        Two narrowings compose into one per-name answer:

        * PP-local: only this stage's gather groups. A group whose names come
          from two stages cannot be served per-stage (each owner would publish
          half while the consumers' plan expects the whole), so discovering one
          DEMOTES the source to gather-to-all — the layouts that do this (tied
          embeddings, MTP) are small models where the gather is not the
          bottleneck.
        * EP-local: within those groups, the replicated names plus this rank's
          own experts. Ownership derives from the assembled names alone (per-
          layer E off the expert indices, ``n_local = E // ep_size``, owner
          ``e // n_local``), so every rank computes the identical answer for
          names it does not hold — which is what lets the consumers route from
          the sender's copy alone.

        ``None`` (holds everything) when neither narrowing applies: naive mode,
        pp==1 with ep==1, or after a demotion.
        """
        if self._demoted:
            return None
        if not self._pp_local and not self._ep_local:
            return None
        return self._held_names_impl()

    def _owned_group_indices(self) -> Optional[List[int]]:
        """This stage's gather groups, or None for all of them; demotes the
        source when a group spans two pipeline stages (see ``held_names``)."""
        if not self._pp_local:
            return None
        self.metadata()  # populates _group_stages / _owned_group_idx
        shared = [gi for gi, st in enumerate(self._group_stages) if len(st) > 1]
        if shared:
            logger.warning(
                "[stacked-source] gather groups %s are produced by more than one pipeline "
                "stage (tied embeddings / MTP), which this source cannot serve per-stage. "
                "Demoting: make_weight_source delegates to the plain MegatronWeightSource "
                "(naive whole-model extraction, correct but slower).",
                shared[:4],
            )
            # Both narrowings flip TOGETHER: a name a rank no longer serves
            # per-stage would misroute pulls. A demoted source refuses to
            # iterate (see _iter_groups_impl).
            self._demoted = True
            self._pp_local = False
            self._ep_local = False
            return None
        return list(self._owned_group_idx)

    @staticmethod
    def _expert_index_of(name: str) -> Optional[int]:
        """Global expert index of an HF expert weight name, or None."""
        if ".experts." not in name:
            return None
        try:
            return int(name.split(".experts.")[1].split(".")[0])
        except (IndexError, ValueError):
            return None

    def _name_owner(self) -> List[int]:
        """Per-metadata-name EP owner: the expert-parallel rank holding it, or
        ``-1`` for names every EP rank replicates. Pure function of the names,
        so every rank derives the same list."""
        meta = self.metadata()
        e_max: dict = {}
        parsed: list = []
        for m in meta:
            e = self._expert_index_of(m.name)
            layer = self._layer_of_hf_name(m.name) if e is not None else None
            parsed.append((e, layer))
            if e is not None:
                e_max[layer] = max(e, e_max.get(layer, -1))
        owners: List[int] = []
        for e, layer in parsed:
            if e is None:
                owners.append(-1)
                continue
            n_experts = e_max[layer] + 1
            if n_experts % self._ep_size:
                raise RuntimeError(
                    f"[stacked-source] layer {layer} has {n_experts} experts, not divisible "
                    f"by ep_size={self._ep_size}; cannot derive expert ownership"
                )
            owners.append(e // (n_experts // self._ep_size))
        return owners

    def _held_names_impl(self) -> Optional[List[str]]:
        """``held_names`` proper: this stage's groups (PP-local) narrowed to the
        replicated names plus this rank's own experts (EP-local)."""
        owned = self._owned_group_indices()  # may demote at pp>1
        if self._demoted:
            return None
        meta = self.metadata()
        names = [m.name for m in meta]
        groups = layerwise_groups(names)
        gis = range(len(groups)) if owned is None else owned
        if not self._ep_local:
            return [n for gi in gis for n in groups[gi]]
        owner_of = dict(zip(names, self._name_owner()))
        return [n for gi in gis for n in groups[gi] if owner_of[n] in (-1, self._my_ep_rank)]

    def _maybe_verify(self) -> None:
        if not self._verified:
            if os.environ.get("SKYRL_RDT_VERIFY_STACKED") == "1":
                self._verify_against_bridge()
                logger.info("[stacked-verify] PASSED: all sampled layers match bridge export")
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
            for name, tensor in self._prepared_model_bridge().stream_weights_megatron_to_hf(
                self._module,
                self._bridge.hf_pretrained,
                cpu=False,  # see _walk_groups: the raw stream defaults to cpu=True
                show_progress=False,
                conversion_tasks=expert_tasks,
            ):
                ref = tensor.to(dtype=self._dtype)
                if name not in mine:
                    raise RuntimeError(f"[stacked-verify] layer {layer}: missing {name}")
                got = mine[name]
                if got is None:
                    # Foreign expert under shard-aware serving: never
                    # materialized here, so there is nothing to compare.
                    del ref, tensor
                    continue
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
                logger.info("[stacked-verify] layer %d: expert tensors match bridge export", layer)


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
                # ETP>1 means no rank holds a WHOLE expert (shapes would be read
                # off shards; EP-local stamping has no truthful answer), so probe
                # for EVERY config — Megatron defaults etp to tp when unset, not
                # to 1. Unprobeable (no mpu) means no ETP — except under LoRA,
                # where the stack merge is at stake and the conservative answer
                # is to fall back (the pre-existing behaviour).
                try:
                    from megatron.core import parallel_state

                    etp = torch.distributed.get_world_size(parallel_state.get_expert_tensor_parallel_group())
                except Exception:  # noqa: BLE001
                    etp = 1 if not wrapped else 0
                etp_ok = etp == 1
                # A DENSE model has no expert tasks, but the source's two grains
                # engage INDEPENDENTLY: PP-local ("this stage exports only its own
                # layers") needs no experts at all. At pp>1 that is the difference
                # between fitting and not -- measured on Qwen3-32B, where the plain
                # source's whole-model export OOMed at both tp4/pp2 (70.56 GiB) and
                # tp8/pp2 (73.06 GiB) of 79.18. Halving per-rank params did not help,
                # which is what identifies the export as model-sized rather than
                # shard-sized.
                #
                # At pp==1 the stacked source degenerates to the plain
                # filtered==full export, so there is nothing to win and the simpler
                # path is kept. The `_demoted` check below is unchanged and still
                # catches layouts a stage cannot serve alone (tied embeddings, MTP).
                pp_local_gain = False
                if not expert_tasks:
                    try:
                        from megatron.core import parallel_state

                        pp_local_gain = parallel_state.get_pipeline_model_parallel_world_size() > 1
                    except Exception:  # noqa: BLE001
                        pp_local_gain = False
                if (expert_tasks or pp_local_gain) and not grouped and etp_ok:
                    if wrapped:
                        logger.info("[rdt] stacked expert source with LoRA stack-merge (etp=1)")
                    src = MegatronStackedWeightSource(weight_extractor, dtype)
                    # pp>1: run the shared-group discovery NOW (one metadata
                    # walk, rank-identical on every rank) so a tied-embeddings/
                    # MTP layout delegates to the plain source here instead of
                    # failing mid-init. At pp==1 this returns without work.
                    src.held_names()
                    if src._demoted:
                        logger.warning(
                            "[rdt] a gather group spans pipeline stages; delegating to the "
                            "plain MegatronWeightSource (naive whole-model extraction)"
                        )
                        return MegatronWeightSource(weight_extractor, dtype)
                    return src
                if not etp_ok and expert_tasks:
                    logger.info(
                        "[rdt] expert_tensor_parallel_size != 1: no rank holds a whole "
                        "expert; using plain MegatronWeightSource (the bridge gathers "
                        "ETP shards internally — correct, slower)"
                    )
                elif grouped:
                    logger.info("[rdt] grouped-export arch; using plain MegatronWeightSource")
                else:
                    logger.info(
                        "[rdt] dense model at pp==1: PP-local serving would narrow nothing "
                        "and there are no experts to stack; using plain MegatronWeightSource"
                    )
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
        # Snapshot only what the sync control plane needs (server URLs + slots + DP
        # size); the async client, its aiohttp session, and the event loop are NOT used.
        #
        # The URLs in this snapshot can go STALE -- a restarted engine returns on a
        # re-reserved port -- so nothing on this path may key on them. The slots
        # cannot: they are assigned at provision and a restart reuses its own. Hence
        # every per-sync decision below is a function of `_server_slots`, and the URLs
        # are only ever used as fan-out addresses, refreshed per sync by `set_live`.
        self._server_urls = list(inference_client.server_urls)
        self._server_slots = list(inference_client.server_slots or range(len(self._server_urls)))
        self._num_replicas = int(inference_client.num_provisioned_replicas)
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

    def get_worker_init_info(self) -> Optional[dict]:
        """The consumer init payload a RESTARTED inference engine needs to rejoin.

        Fetched ONCE by the driver, from rank 0, right after ``init_weight_sync_state``,
        and cached there for the run. Static, because it describes the PRODUCERS —
        their actor names and the group-major weight metadata — and producers never
        restart. Pure and collective-free (see the engine's
        ``get_worker_init_payload``), so the driver may call it while every rank is
        elsewhere.

        Returns ``None`` on a rank that is not the sender, and on a sender whose engine
        has not initialized — both mean "ask rank 0 instead", not an error.
        """
        engine = self._engine
        if engine is None or not getattr(engine, "is_sender", False):
            return None
        return engine.get_worker_init_payload()

    def _live_consumer_ids(self, live_url_slots: Optional[Sequence[Tuple[str, int]]]) -> Optional[set]:
        """Map the live servers' SLOTS to the consumer ids those servers own.

        A server's consumer-id block is a function of the slot it was stamped with at
        init (``replica_rank``), not of how many servers are currently alive and not
        of where it sits in a degraded list. Re-deriving the geometry from the live
        list is the one genuinely dangerous mistake available on this path: it would
        silently renumber every surviving consumer, and each would then pull another
        consumer's slice with nothing downstream to notice. So both inputs to the
        arithmetic — the slot, and ``_num_replicas`` — come from provision.

        Returns ``None`` when the whole fleet is live, so the engine takes its
        historical (non-degraded) path exactly.
        """
        if live_url_slots is None:
            return None
        live_slots = list(dict.fromkeys(s for _, s in live_url_slots))
        unknown = [s for s in live_slots if s not in self._server_slots]
        if unknown:
            raise RuntimeError(
                f"live_url_slots contains slots outside the provisioned set: {unknown}. "
                f"Provisioned: {sorted(set(self._server_slots))}. Slots are assigned at "
                "provision and a restart reuses its own, so an unknown slot means the fleet "
                "geometry changed shape -- which no part of engine fault tolerance supports."
            )
        if len(live_slots) == len(set(self._server_slots)):
            return None
        if not live_slots:
            raise RuntimeError("live_url_slots is empty: no inference server is alive to receive weights.")

        # replica ordinal -> its block of consumer ids. `_num_replicas` is the
        # PROVISIONED deployment count, matching what the consumers were stamped with
        # at init.
        workers_per_replica = self._world_size // max(1, self._num_replicas)
        return {rr * workers_per_replica + w for rr in live_slots for w in range(workers_per_replica)}

    async def send(self, weight_extractor: Any, live_url_slots: Optional[Sequence[Tuple[str, int]]] = None) -> None:
        """Sync weights once; every rank must call it (the gather is a
        collective). ``initialize()`` should already have run; the lazy fallback
        here only covers callers that skipped it (and reintroduces the
        first-send deadlock risk described there — do not rely on it).

        ``live_url_slots`` is the driver's current view of which inference servers are
        alive, as ``(url, slot)``. ``None`` (the default) means the whole provisioned
        fleet, which is every non-fault-tolerant run. Every rank must be handed the
        SAME value — the ranks share gather collectives — which the driver guarantees
        by computing it once and dispatching it to all of them."""
        if weight_extractor is None:
            raise RuntimeError(
                "sharded_rdt weight sync requires the worker's weight_extractor " "(built in init_weight_sync_state)."
            )
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._trainer_init_blocking, weight_extractor)
        live_cids = self._live_consumer_ids(live_url_slots)
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
                f"({len(live_url_slots or [])}/{len(self._server_urls)} inference servers alive); "
                f"each group's free barrier counts only the live consumers"
            )
        if self._control_plane is not None:
            # Only rank 0 issues control-plane calls, but every rank holds a client
            # and setting this everywhere keeps the two views from drifting.
            self._control_plane.set_live(live_url_slots)
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
        # Surface the vendored engine's per-round timing decomposition
        # (trainer-side sync buckets + producer-server counters) in the worker
        # log; the per-sync records also land in /tmp/rdt_profile/trainer.jsonl
        # above. Fail-soft — never let timing collection break a sync.
        try:
            if hasattr(self._engine, "get_sync_timing"):
                # loguru formats with str.format, NOT %-style: a "%s" message with
                # args renders the literal message and silently drops the dict.
                _loguru.info("[rdt-sync-timing] sync_timing={}", self._engine.get_sync_timing())
            if hasattr(self._engine, "get_produce_timing"):
                produce_timing = await asyncio.to_thread(self._engine.get_produce_timing)
                _loguru.info("[rdt-sync-timing] produce_timing={}", produce_timing)
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
        self._control_plane = SyncRdtControlPlaneClient(
            self._server_urls,
            self._data_parallel_size,
            slots=self._server_slots,
            num_replicas=self._num_replicas,
        )
        dtype = str_to_torch_dtype(self._ie_cfg.model_dtype)
        source = make_weight_source(weight_extractor, dtype)
        # Positive configuration tripwire. make_weight_source's own logs use the
        # vLLM logger, which does NOT forward to the driver log from a Megatron
        # rank actor (only loguru does), and the STACKED_EXPERTS=0 short-circuit
        # logs nothing at all -- so an ablation could silently run the wrong
        # source. Log the CHOSEN CLASS (not the env var) plus both knobs.
        _loguru.info(
            "[rdt-config] source={} lookahead_env={} stacked_experts_env={}",
            type(source).__name__,
            os.environ.get("SKYRL_RDT_LOOKAHEAD", "<default>"),
            os.environ.get("SKYRL_RDT_STACKED_EXPERTS", "<default>"),
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Pipeline-depth knobs are env-driven (SKYRL_* env vars are forwarded
        # into every Ray worker by prepare_runtime_environment). The env
        # override exists because the sync is a credit-limited latency
        # pipeline: its steady-state period is (publish->serve->pull->free
        # RTT) / credits, so deepening K/lookahead hides per-link latency
        # without touching any link.
        def _knob(env: str, default):
            v = os.environ.get(env)
            return v if v is not None else default

        init_info = ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=self._world_size,
            trainer_actor_namespace=self._namespace,
            num_rdt_buffers=int(_knob("SKYRL_RDT_NUM_BUFFERS", _DEFAULT_NUM_RDT_BUFFERS)),
            arena_presize_gb=float(_knob("SKYRL_RDT_ARENA_PRESIZE_GB", _DEFAULT_ARENA_PRESIZE_GB)),
            pack_check=os.environ.get("SKYRL_RDT_PACK_CHECK") == "1",
            gather_lookahead=int(_knob("SKYRL_RDT_LOOKAHEAD", _DEFAULT_GATHER_LOOKAHEAD)),
            # Resolved here, not in the sidecar: the sidecar is a Ray actor that
            # inherits the raylet's environment, so a launch-time SKYRL_* override
            # never reaches it. Env first, then the FT config, then the default.
            stall_timeout_s=float(
                os.environ.get("SKYRL_RDT_STALL_TIMEOUT_S")
                or getattr(getattr(self._ie_cfg, "fault_tolerance", None), "stall_timeout_s", None)
                or _DEFAULT_STALL_TIMEOUT_S
            ),
        )
        engine = ShardedRDTTrainerWeightTransferEngine.trainer_init(
            init_info,
            client=self._control_plane,
            source=source,
        )
        # The producer sidecar already freezes its object graph "so gen-2 GC
        # never stops the world mid-serve". The trainer needs it more: it
        # rebuilds the whole conversion-task graph every sync on top of a 235B
        # heap, and a gen-2 pass costs up to 1.2s on a rank -- which the sync's
        # PP all_gather_object then propagates to that rank's partner, so the
        # slowest pair sets the sync. Freezing leaves gen-0/1 untouched and makes
        # gen-2 ~40x cheaper by skipping the static graph.
        if os.environ.get("SKYRL_RDT_GC_FREEZE", "1") not in ("0", "false", "False"):
            import gc

            gc.collect()
            gc.freeze()
        return engine

    def teardown(self) -> None:
        engine = self._engine
        control_plane = self._control_plane
        self._engine = None
        self._control_plane = None
        if engine is not None:
            engine.shutdown()
        if control_plane is not None:
            control_plane.close()
