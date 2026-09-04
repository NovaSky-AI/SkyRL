"""Sharded-RDT (NIXL pull) weight sources and trainer init info.

Sources here are :class:`GroupedWeightSource`s, not the plain
``weight_sync/sources.py`` ones: a pull backend needs per-rank ownership
(``held_names()``) and a group index its free barrier counts. See
``sharded_rdt_base``.

Three flavors, chosen in :func:`make_megatron_weight_source` /
:func:`make_fsdp_weight_source`: :class:`RdtFsdpWeightSource` (all-gather via
``full_tensor()``), :class:`MegatronStackedWeightSource` (PP-local + EP-local,
stack-granularity gathers) and :class:`RdtMegatronWeightSource` (the whole-model
Megatron-Bridge ``export_hf_weights`` fallback).
"""

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import torch
from loguru import logger as _loguru

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
    GroupedWeightSource,
    ParamMeta,
    layerwise_groups,
    materialize_full_tensor,
)
from skyrl.backends.skyrl_train.weight_sync.sources import (
    FsdpWeightSource,
    MegatronWeightSource,
)

logger = logging.getLogger(__name__)

# Defaults for the must-agree wire knobs (mirror the vLLM sharded_rdt defaults).
_DEFAULT_NUM_RDT_BUFFERS = 2
_DEFAULT_BUFFER_PRESIZE_GB = 0.0
# Producer stall watchdog (seconds). Mirrors sharded_rdt_trainer.
# DEFAULT_STALL_TIMEOUT_S; duplicated rather than imported so resolving the knob
# does not pull the vendored trainer module into every worker.
_DEFAULT_STALL_TIMEOUT_S = 300.0
# Gathered-but-unfreed groups the trainer's gather loop runs ahead by; bounds
# trainer-resident memory at lookahead + 1 groups (see sharded_rdt_trainer.
# DEFAULT_GATHER_LOOKAHEAD). 1 = while the consumers pull group N, group N+1 is
# already gathered AND published; raise via SKYRL_RDT_LOOKAHEAD only if one
# group's gather is slower than its pulls.
_DEFAULT_GATHER_LOOKAHEAD = 1


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


class RdtFsdpWeightSource(FsdpWeightSource, GroupedWeightSource):
    """The shared FSDP ``WeightSource``, re-ordered group-major.

    Same state_dict and ``full_tensor()`` gather as
    ``weight_sync.sources.FsdpWeightSource``; only the order differs. RDT needs
    each ``model.layers.<N>.*`` block contiguous so ``layerwise_groups``
    partitions ``metadata()`` exactly -- that partition is the group index the
    consumers' pull plans and the producer's free barrier are keyed on. The push
    backends transfer any permutation identically, so they use the plain source.
    """

    def __init__(self, model: Any, dtype: torch.dtype, weight_prefix: str = "") -> None:
        super().__init__(model, dtype, weight_prefix)
        sd = model.state_dict()
        prefix = weight_prefix or ""
        names = [f"{prefix}{key}" for key in sd.keys()]
        shapes = {f"{prefix}{key}": tuple(param.shape) for key, param in sd.items()}
        idx = {n: i for i, n in enumerate(names)}
        order = [idx[n] for g in layerwise_groups(names) for n in g]
        self._names = [names[i] for i in order]
        self._shapes = [shapes[names[i]] for i in order]

    def metadata(self) -> List[ParamMeta]:
        return [ParamMeta(name, self._dtype, tuple(shape)) for name, shape in zip(self._names, self._shapes)]

    def __iter__(self) -> Iterator[tuple]:
        # The caller selects this rank's CUDA device before iterating; a worker
        # thread does not inherit it (see Worker._weight_sync_thread).
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        prefix = self.weight_prefix
        # The same handle metadata() was built from, so names and gather match
        # what the consumer engine baked its plan over.
        sd = self.model.state_dict()
        for name in self._names:
            raw = name[len(prefix) :] if prefix and name.startswith(prefix) else name
            param = sd[raw]
            if device is not None:
                param = param.to(device, non_blocking=True)
            yield name, materialize_full_tensor(param).to(self._dtype).detach().contiguous()


class RdtMegatronWeightSource(MegatronWeightSource, GroupedWeightSource):
    """The shared Megatron ``WeightSource``, with RDT's group channels.

    Identical stream to ``weight_sync.sources.MegatronWeightSource``: the
    whole-model export, in HF-canonical (already group-contiguous) order, with
    every rank getting full tensors. For a pull backend that whole-model
    residency is the point -- each producer must be able to serve its bound
    consumer the complete model -- so the inherited "hold everything" defaults
    are correct and this subclass adds nothing else.

    :class:`MegatronStackedWeightSource` is the narrower alternative;
    ``make_megatron_weight_source`` falls back here when it cannot serve a layout.
    """


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


class MegatronStackedWeightSource(GroupedWeightSource):
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

    def __init__(self, bridge: Any, module: Any, dtype: torch.dtype) -> None:
        self._bridge = bridge
        self._module = module
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
        # extraction is the plain RdtMegatronWeightSource
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
        # the plain RdtMegatronWeightSource instead. Iteration refuses to run.
        self._demoted = False
        # Per-layer (F, H) recorded as layers are walked, so metadata() can
        # synthesize the shapes of foreign experts (their tensors are None).
        self._layer_geom: dict = {}
        self._group_stages: List[set] = []  # group idx -> stages that produce it
        self._owned_group_idx: List[int] = []
        self._group_index_of_name: dict = {}
        # Per-sync source phase timing (expert_gather / ne_* buckets), drained
        # into trainer.jsonl by RdtWeightSyncSender after each send.

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

    def _resolved_expert_names(self, lay: "_ExpertLayer", E: int) -> list:
        """Expert HF names straight from the bridge's mapping registry.

        ``megatron_to_hf_lookup`` resolves a Megatron param name to the CONCRETE
        HF name(s) — no model, no collective, pure string work — so whatever the
        architecture calls its decoder stack, its MoE container and its
        projections is what we emit. The fused fc1 resolves to a
        ``GatedMLPMapping`` whose ``hf_param`` names gate/up BY KEY, so the two
        halves are identified rather than assumed.

        Raises on any shape we cannot read, rather than guessing. A wrong name
        here fails SILENTLY -- the consumer never baked it, so the slice is simply
        never pulled and that expert keeps its stale weights, with nothing
        downstream to notice. Failing the sync loudly is the only safe response.
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
                    raise RuntimeError(
                        "sharded_rdt could not resolve expert HF names from the "
                        f"Megatron-Bridge mapping registry for layer {lay.layer} expert {e}: "
                        f"expected fc1 -> {{'gate': str, 'up': str}} and fc2 -> str, got "
                        f"fc1={fc1!r} fc2={fc2!r}. Refusing to guess: an unresolved expert "
                        "name is never baked by the consumer, so that expert would silently "
                        "keep stale weights."
                    )
                names.append(fc1["gate"])
                names.append(fc1["up"])
                names.append(fc2)
            return names
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 - add context, then fail the sync
            raise RuntimeError(
                "sharded_rdt failed to look up expert HF names via the Megatron-Bridge "
                f"mapping registry (layer {lay.layer}, {E} experts): {type(e).__name__}: {e}. "
                "Refusing to fall back to synthesized names, which would silently skip "
                "these experts."
            ) from e

    def _expert_names_for(self, lay: "_ExpertLayer", E: int) -> list:
        """Per-layer expert HF names in yield order (gate/up/down per expert),
        built once and cached — otherwise ~36k lookups per sync at 235B.

        Always resolved through the bridge (see ``_resolved_expert_names``); names
        are never synthesized from an assumed layout. Architectures differ (Kimi
        K2.5-VL nests the stack under ``language_model.`` where Qwen3-MoE, GLM and
        DeepSeek do not), and a wrong name is never baked by the consumer, so those
        experts would silently keep stale weights.
        """
        names = self._expert_names.get(lay.layer)
        if names is not None:
            return names
        names = self._resolved_expert_names(lay, E)
        if getattr(self, "_expert_name_source", None) is None:
            # Positive marker, once, so a run records where its names came from.
            self._expert_name_source = "bridge mapping_registry"
            logger.info("[rdt-config] expert HF names from: bridge mapping_registry")
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
        each a contiguous slab. ``unbind`` batches the view creation: 3 dispatcher
        calls per layer instead of 3E ``__getitem__`` plus 3E no-op
        ``.contiguous()``."""
        # Recorded BEFORE the expert names are emitted, so metadata() can
        # synthesize foreign experts' shapes (their tensors are None).
        self._layer_geom[lay.layer] = (lay.F, lay.H)
        fc1_local, fc2_local = self._local_layer_stacks(lay, adapter_ctx)
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
        tasks = self._bridge.get_conversion_tasks(self._module)
        non_expert, raw_layers = self._partition_tasks(tasks)
        layers = self._expert_geometry(raw_layers)
        # LoRA-wrapped experts: raw to_wrap weights need the adapter delta
        # merged in (the bridge would have merged during its per-tensor export).
        # Decided from the FULL task list, which get_conversion_tasks builds
        # identically on every rank: build_adapter_conversion_tasks is collective
        # (a cached PP all_gather_object), so a condition that could differ per
        # stage would desync. In PP-local mode `layers` is this stage's only, so
        # it must not take part in this decision.
        wrapped = any(".to_wrap." in t.global_param_name for t in tasks)
        adapter_ctx = self._adapter_tasks_by_base() if wrapped else None
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
                "delegates to the plain RdtMegatronWeightSource automatically."
            )
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
                item = next(_it, None)
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
                # Warm steady state: bridge already yields target dtype on-device —
                # skip the no-op .to()/.contiguous() dispatches.
                if tensor.dtype != self._dtype or (target_dev is not None and tensor.device != target_dev):
                    tensor = tensor.to(device=target_dev, dtype=self._dtype)
                tensor = tensor.detach()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
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
                "Demoting: make_weight_source delegates to the plain RdtMegatronWeightSource "
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


def make_fsdp_weight_source(model: Any, dtype: torch.dtype, weight_prefix: str = "") -> GroupedWeightSource:
    """The RDT ``WeightSource`` for an FSDP policy model."""
    return RdtFsdpWeightSource(model, dtype, weight_prefix)


def make_megatron_weight_source(bridge: Any, module: Any, dtype: torch.dtype) -> GroupedWeightSource:
    """Pick the RDT ``WeightSource`` for a Megatron policy model.

    Prefer :class:`MegatronStackedWeightSource` (stack-granularity expert
    gathers; ~20x fewer collectives on per-expert MoE archs, and PP-local export
    at pp>1) unless the arch uses grouped-export mappings (fused HF expert names
    — a different contract) or ``SKYRL_RDT_STACKED_EXPERTS=0``. Everything else
    falls back to the whole-model :class:`RdtMegatronWeightSource`.
    """
    if os.environ.get("SKYRL_RDT_STACKED_EXPERTS", "1") != "0":
        try:
            tasks = bridge.get_conversion_tasks(module)
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
                src = MegatronStackedWeightSource(bridge, module, dtype)
                # pp>1: run the shared-group discovery NOW (one metadata
                # walk, rank-identical on every rank) so a tied-embeddings/
                # MTP layout delegates to the plain source here instead of
                # failing mid-init. At pp==1 this returns without work.
                src.held_names()
                if src._demoted:
                    logger.warning(
                        "[rdt] a gather group spans pipeline stages; delegating to the "
                        "plain RdtMegatronWeightSource (naive whole-model extraction)"
                    )
                    return RdtMegatronWeightSource(bridge, module, dtype)
                return src
            if not etp_ok and expert_tasks:
                logger.info(
                    "[rdt] expert_tensor_parallel_size != 1: no rank holds a whole "
                    "expert; using plain RdtMegatronWeightSource (the bridge gathers "
                    "ETP shards internally — correct, slower)"
                )
            elif grouped:
                logger.info("[rdt] grouped-export arch; using plain RdtMegatronWeightSource")
            else:
                logger.info(
                    "[rdt] dense model at pp==1: PP-local serving would narrow nothing "
                    "and there are no experts to stack; using plain RdtMegatronWeightSource"
                )
        except Exception:  # noqa: BLE001
            logger.warning("[rdt] stacked-source probe failed; using plain RdtMegatronWeightSource", exc_info=True)
    return RdtMegatronWeightSource(bridge, module, dtype)


def build_rdt_trainer_init_info(
    rank: int,
    inference_world_size: int,
    server_urls: List[str],
    data_parallel_size: int,
):
    """Build ``ShardedRDTTrainerInitInfo`` for this rank.

    Every knob is resolved on the **trainer**: the producer sidecar is a Ray
    actor that inherits the raylet's environment, so a launch-time ``SKYRL_*``
    override never reaches it.

    Args:
        rank: this trainer process's rank. Rank 0 is the sender.
        inference_world_size: total inference workers across the fleet — the
            consumer count the producers' free barrier counts against.
        server_urls: every inference server, ordered
            ``[engine0_dp0, ..., engine1_dp0, ...]``. Only its length and the DP
            size are used, to derive the deployment count.
        data_parallel_size: DP replicas per deployment.
    """
    from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer import (
        ShardedRDTTrainerInitInfo,
    )

    if not inference_world_size:
        raise ValueError(
            f"sharded_rdt requires the inference world size (consumer count); got {inference_world_size!r}."
        )

    def _knob(env: str, default):
        v = os.environ.get(env)
        return v if v is not None else default

    # Deployment count, derived the same way the control plane derives each
    # server's ``replica_rank`` (see ``control_plane.rdt_init_payloads``): the DP
    # servers of one deployment share a parallel config, so they share one
    # ordinal. [RDT-SHARE-SLOTS] reads it to size a deployment's consumer-id block.
    dp = max(1, int(data_parallel_size))
    num_replicas = max(1, len(server_urls) // dp)

    # [RDT-SHARE-SLOTS] Consumers per deployment, which is what groups the
    # workers that can share one serve slot on each producer. 0 turns sharing
    # off, and one deployment makes it a no-op anyway (the width equals the
    # consumer count, so every group is a singleton).
    share_slots = os.environ.get("SKYRL_RDT_SHARE_SLOTS", "1") not in ("0", "false", "False")
    workers_per_replica = int(inference_world_size) // num_replicas if share_slots else 0
    _loguru.info(
        "[rdt-config] num_consumers={} num_replicas={} workers_per_replica={} (slot sharing {})",
        inference_world_size,
        num_replicas,
        workers_per_replica,
        "on" if share_slots and num_replicas > 1 else "off",
    )

    return ShardedRDTTrainerInitInfo(
        rank=rank,
        num_consumers=int(inference_world_size),
        workers_per_replica=workers_per_replica,
        trainer_actor_namespace=ray_namespace(),
        num_rdt_buffers=int(_knob("SKYRL_RDT_NUM_BUFFERS", _DEFAULT_NUM_RDT_BUFFERS)),
        buffer_presize_gb=float(_knob("SKYRL_RDT_BUFFER_PRESIZE_GB", _DEFAULT_BUFFER_PRESIZE_GB)),
        gather_lookahead=int(_knob("SKYRL_RDT_LOOKAHEAD", _DEFAULT_GATHER_LOOKAHEAD)),
        stall_timeout_s=float(_knob("SKYRL_RDT_STALL_TIMEOUT_S", _DEFAULT_STALL_TIMEOUT_S)),
    )


def ray_namespace() -> Optional[str]:
    """This process's Ray namespace, or None outside a Ray runtime.

    The producer sidecars are named actors, so the consumers need the namespace
    they were created in to resolve them.
    """
    try:
        import ray

        return ray.get_runtime_context().namespace or None
    except Exception:  # noqa: BLE001 - no Ray runtime: the caller falls back to the default namespace
        return None


def freeze_trainer_heap() -> None:
    """``gc.freeze()`` the trainer's static object graph after the rendezvous.

    The trainer rebuilds the whole conversion-task graph every sync on top of a
    235B heap, and a gen-2 pass costs up to 1.2s on a rank -- which the sync's PP
    ``all_gather_object`` propagates to that rank's partner, so the slowest pair
    sets the sync. Freezing leaves gen-0/1 untouched and makes gen-2 ~40x cheaper.
    """
    if os.environ.get("SKYRL_RDT_GC_FREEZE", "1") in ("0", "false", "False"):
        return
    import gc

    gc.collect()
    gc.freeze()


def log_source_choice(source: Any) -> None:
    """Log the chosen source class and both knobs.

    ``make_megatron_weight_source``'s own logs use the vLLM logger, which does
    not forward to the driver log from a Megatron rank actor (only loguru does),
    and the ``STACKED_EXPERTS=0`` short-circuit logs nothing -- so without this
    an ablation could silently run the wrong source.
    """
    _loguru.info(
        "[rdt-config] source={} lookahead_env={} stacked_experts_env={}",
        type(source).__name__,
        os.environ.get("SKYRL_RDT_LOOKAHEAD", "<default>"),
        os.environ.get("SKYRL_RDT_STACKED_EXPERTS", "<default>"),
    )
