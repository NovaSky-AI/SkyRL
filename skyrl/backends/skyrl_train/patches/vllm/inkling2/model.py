"""Inkling2 for RL rollouts: BF16 routed experts and weight-sync-friendly loading.

The vLLM branch serves the release's NVFP4 / MXFP8 experts only (fused quantized routing, no
BF16 mode), and its ``load_weights``
requires every checkpoint tensor in one call with matching dtypes. Training syncs BF16 weights
in chunks, so :class:`Inkling2ForRL` builds BF16 experts and loads any subset of tensors. vLLM
picks the ``inkling2`` quantization config by itself for the release (its
``hf_quant_config.json``); the experts stay BF16 regardless, and the packed release experts are
dequantized as they load (``patches/megatron/inkling2/quant.py``, the dequantization Megatron's
bridge applies). ``SKYRL_INKLING2_QUANTIZED_EXPERTS=1`` keeps the branch's quantized experts.

``SKYRL_INKLING2_ROLLOUT_EXPERTS=mxfp8`` instead serves every layer's experts with the branch's
fused MXFP8 kernels (:class:`Inkling2MXFP8Experts`) while still taking BF16 weights: each expert
is quantized to MXFP8 as it is loaded, at boot and on every weight sync, in place. The fused
MXFP8 Triton kernels need Triton >= 3.8 (they miscompute under 3.7.1 on B200).

The BF16 experts run their own kernels (``ops.py``, vendored from an earlier revision of the branch) and
implement the branch's expert LoRA hook (``moe.ExpertsLoRA``, from the LoRA PR).
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling2 import model as inkling2_model
from vllm.models.inkling2 import moe as inkling2_moe
from vllm.models.inkling2.attention import Inkling2Attention
from vllm.models.inkling2.model import Inkling2ForConditionalGeneration
from vllm.models.inkling2.ops.block_scales import shuffle_rows, swizzle_block_scales
from vllm.models.inkling2.ops.sconv import shift_rel_proj

from skyrl.backends.skyrl_train.patches.megatron.inkling2.quant import (
    dequantize_expert_weight,
)
from skyrl.backends.skyrl_train.patches.vllm.inkling2.ops import (
    combine,
    grouped_gemm,
    linear_k_groups,
    select_experts,
    swiglu,
)
from skyrl.backends.skyrl_train.patches.vllm.patch_moe_expert_lora_rank import (
    register_moe_layer_type,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import mx_cast_to_fp8

# "bf16" (default): BF16 experts. "mxfp8": the branch's MXFP8 experts, quantized on load.
ROLLOUT_EXPERTS = os.environ.get("SKYRL_INKLING2_ROLLOUT_EXPERTS", "bf16").lower()
if ROLLOUT_EXPERTS not in ("bf16", "mxfp8"):
    raise ValueError(f"SKYRL_INKLING2_ROLLOUT_EXPERTS must be bf16 or mxfp8, got {ROLLOUT_EXPERTS!r}")

_LAYER_INDEX_RE = re.compile(r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)")
# One expert of a packed expert weight, as weight sync sends them (Inkling2Bridge.split_for_weight_sync).
_SINGLE_EXPERT_RE = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(w13_weight|w2_weight)$")
_PACKED_EXPERT_RE = re.compile(r"(\.mlp\.experts\.w(?:13|2)_weight)(_scale|_amax|_outer_scale_denominator)?$")
# Companions a packed expert weight needs, by its storage dtype.
_QUANT_COMPANIONS = {
    torch.uint8: ("_scale", "_amax", "_outer_scale_denominator"),
    torch.float8_e4m3fn: ("_scale",),
}


class Inkling2BF16Experts(inkling2_moe.Inkling2Experts):
    """Routed experts in BF16, sharded contiguously over the TP group like the quantized ones.

    ``w13_weight [E, 2I, H]`` keeps the checkpoint's interleaved gate / up rows, which
    ``ops.swiglu`` reads. A subclass of the branch's experts for ``isinstance`` checks; none of
    its parameters or kernels.
    """

    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter
        for name, shape in (("w13_weight", (2 * inter, hidden)), ("w2_weight", (hidden, inter))):
            param = nn.Parameter(torch.empty(self.num_experts, *shape, dtype=torch.bfloat16), requires_grad=False)
            set_weight_attrs(param, {"weight_loader": self._load_expert_shard})
            self.register_parameter(name, param)
        start = self.ep_rank * self.num_experts
        # A buffer, so that sleep mode (level 2) saves and restores it with the model's buffers.
        # vLLM's layerwise reload would count it in the layer's size and never see it loaded, but
        # weight sync loads this model in place (Inkling2ForRL.skyrl_load_weights_in_place).
        self.register_buffer(
            "expert_bounds",
            torch.arange(start, start + self.num_experts + 1, dtype=torch.int32),
            persistent=False,
        )

    def skyrl_local_expert_ids(self) -> tuple[range, int]:
        """This rank's routed experts and the layer's total, for the colocated local-shard
        weight sync (``weight_sync_local_shard``): only these experts are sent to this rank."""
        start = self.ep_rank * self.num_experts
        return range(start, start + self.num_experts), self.num_experts * self.ep_size

    def is_local(self, expert: int) -> bool:
        return 0 <= expert - self.ep_rank * self.num_experts < self.num_experts

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor, expert: int | None = None) -> None:
        """All experts of the layer, or (``expert`` set) one of this rank's experts."""
        start = self.ep_rank * self.num_experts
        if expert is not None:
            param.data[expert - start].copy_(loaded)
            return
        shard = loaded[start : start + self.num_experts]
        assert shard.shape == param.shape, f"{shard.shape} != {param.shape}"
        param.data.copy_(shard)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        lora: inkling2_moe.ExpertsLoRA | None = None,
        reduce: bool = True,
        push=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Like the branch's experts: without ``reduce``, this rank's unreduced
        ``(down, src2dst)`` with ``src2dst`` -1 for other ranks' rows (the branch's ``combine``
        contract, which its fused all-reduce consumes); with ``push`` (the branch's
        sequence-parallel ``SeqParallel``), the combine of those rows is
        pushed to the token shard owners by the branch's ``combine_push``."""
        num_rows = ids.numel()
        sorted_ids, perm = torch.sort(ids.flatten(), stable=True)
        offsets = torch.searchsorted(sorted_ids, self.expert_bounds, out_int32=True)
        src2dst = torch.empty_like(perm)
        src2dst[perm] = torch.arange(num_rows, device=x.device)
        gate_up = grouped_gemm(x, self.w13_weight, perm, offsets, num_rows, self.topk, gather=True)
        if lora is not None:
            lora.add_w13(gate_up, x, perm, offsets, self.topk)
        row_bounds = (offsets[:1], offsets[-1:])
        # The routing weight is applied inside the SwiGLU, before the down projection.
        activated = swiglu(gate_up, weights.flatten()[perm], self.gate_cap, self.up_cap, row_bounds)
        down = grouped_gemm(activated, self.w2_weight, perm, offsets, num_rows, self.topk, gather=False)
        if lora is not None:
            lora.add_w2(down, activated, perm, offsets, self.topk)
        if push is not None or not reduce:
            local = (src2dst >= offsets[:1]) & (src2dst < offsets[-1:])
            local_src2dst = torch.where(local, src2dst, -1).to(torch.int32)
            if push is None:
                return down, local_src2dst
            from vllm.models.inkling2.ops.moe import combine_push

            combine_push(down, local_src2dst, self.topk, push)
            return None
        if self.ep_size == 1:
            return combine(down, src2dst, offsets, self.topk, x.dtype)
        out = combine(down, src2dst, offsets, self.topk, torch.float32)
        return tensor_model_parallel_all_reduce(out).to(x.dtype)


class Inkling2MXFP8Experts(inkling2_moe.Inkling2Experts):
    """The branch's MXFP8 experts (fused routing + TRT-LLM-gen GEMMs) that load BF16 weights.

    Each expert's BF16 ``[N, K]`` weight is quantized with ``mx_cast_to_fp8`` (1x32 blocks along
    ``K``, E8M0 scales; the cast the MXFP8 weight-sync wire uses), then written into its slot in
    the branch's layout: rows shuffled in blocks of 32, scales row-shuffled and R128c4-swizzled.
    Both rearrangements act within one expert, so a single expert can be loaded on its own, in
    place (the kernels hold the parameters' addresses). The ``*_weight_scale`` parameters are
    derived, never loaded.
    """

    def __init__(self, config) -> None:
        super().__init__(config, "mxfp8")
        for name in ("w13", "w2"):
            weight = getattr(self, f"{name}_weight")
            # Read by Inkling2ForRL.load_weights: pass BF16 through instead of casting to the param dtype.
            weight.skyrl_bf16_source = True
            weight.weight_loader = functools.partial(self._load_bf16, name)
            getattr(self, f"{name}_weight_scale").weight_loader = self._reject_scale

    def skyrl_local_expert_ids(self) -> tuple[range, int]:
        """See ``Inkling2BF16Experts.skyrl_local_expert_ids``."""
        start = self.ep_rank * self.num_experts
        return range(start, start + self.num_experts), self.num_experts * self.ep_size

    def is_local(self, expert: int) -> bool:
        return 0 <= expert - self.ep_rank * self.num_experts < self.num_experts

    @staticmethod
    def _reject_scale(param: nn.Parameter, loaded: torch.Tensor) -> None:
        raise ValueError("MXFP8 rollout experts derive their scales from BF16 weights; got a scale tensor")

    def _load_bf16(
        self,
        name: str,
        param: nn.Parameter,
        loaded: torch.Tensor,
        expert: int | None = None,
        local: bool = False,
    ) -> None:
        """BF16 ``loaded``: all experts of the layer ``[E, N, K]``, this rank's experts
        (``local``), or (``expert`` set) one of this rank's experts ``[N, K]``."""
        start = self.ep_rank * self.num_experts
        if expert is not None:
            self._quantize_into(name, expert - start, loaded)
            return
        if not local:
            loaded = loaded[start : start + self.num_experts]
        assert loaded.shape[0] == self.num_experts, f"{tuple(loaded.shape)}: expected {self.num_experts} experts"
        for i in range(self.num_experts):
            self._quantize_into(name, i, loaded[i])

    def _quantize_into(self, name: str, index: int, weight: torch.Tensor) -> None:
        param = getattr(self, f"{name}_weight")
        scale = getattr(self, f"{name}_weight_scale")
        assert weight.shape == param.shape[1:], f"{name}: {tuple(weight.shape)} != {tuple(param.shape[1:])}"
        weight = weight.to(device=param.device, dtype=torch.bfloat16)
        codes, scales = mx_cast_to_fp8(weight)
        param.data[index].copy_(shuffle_rows(codes))
        scale.data[index].copy_(swizzle_block_scales(shuffle_rows(scales)))


class _CapturedRouting:
    """A vLLM routed-experts capture source (``RoutedExpertsCaptureSource``): with
    ``enable_return_routed_experts``, vLLM binds ``capture_fn`` to record each token's expert ids
    at ``layer_id`` (the decoder layer, set by :class:`Inkling2ForRL`), for rollout router replay."""

    layer_id: int = -1
    capture_fn = None


class _MXFP8MoE(_CapturedRouting, inkling2_moe.Inkling2MoE):
    """The branch's MoE and fused routing on :class:`Inkling2MXFP8Experts`, for every layer."""

    def __init__(self, config, expert_format) -> None:
        nn.Module.__init__(self)
        assert config.n_shared_experts == 0
        self.gate = inkling2_moe.Inkling2Gate(config)
        self.experts = Inkling2MXFP8Experts(config)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.capture_fn is not None:
            # The fused routing kernel keeps its expert ids to itself. Recompute them with the
            # reference gate (K-grouped fp32 logits, ties to the lower expert): the same selection.
            x = x.contiguous()
            gate = self.gate
            _, ids = select_experts(linear_k_groups(x, gate.weight), gate.bias, gate.topk, gate.route_scale)
            self.capture_fn(ids)
        return super().forward(x, *args, **kwargs)


class _LocalExperts:
    """This rank's experts of a packed expert weight, already sliced and dequantized."""

    def __init__(self, local: torch.Tensor) -> None:
        self.local = local
        self.dtype = local.dtype


class _BF16MoE(_CapturedRouting, inkling2_moe.Inkling2MoE):
    """The branch's MoE (its gate parameters, its LoRA wrapper's ``isinstance`` check) on BF16
    experts, routed by ``ops.select_experts``."""

    def __init__(self, config, expert_format) -> None:
        nn.Module.__init__(self)
        assert config.n_shared_experts == 0
        self.gate = inkling2_moe.Inkling2Gate(config)
        self.experts = Inkling2BF16Experts(config)

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        push=None,
        lora: inkling2_moe.ExpertsLoRA | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        # ``push`` sits before ``lora``, like the branch's signature, which its LoRA wrapper
        # calls positionally.
        x = x.contiguous()
        gate = self.gate
        logits = linear_k_groups(x, gate.weight)
        weights, ids = select_experts(logits, gate.bias, gate.topk, gate.route_scale)
        if self.capture_fn is not None:
            self.capture_fn(ids)
        return self.experts(x, weights, ids, lora=lora, reduce=reduce, push=push)


@contextmanager
def _rollout_experts():
    original = inkling2_model.Inkling2MoE
    inkling2_model.Inkling2MoE = _MXFP8MoE if ROLLOUT_EXPERTS == "mxfp8" else _BF16MoE
    try:
        yield
    finally:
        inkling2_model.Inkling2MoE = original


def _load_head_rows(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """This TP rank's rows (heads) of a per-head parameter."""
    rows = param.shape[0]
    param.data.copy_(loaded_weight.narrow(0, get_tensor_model_parallel_rank() * rows, rows))


def _make_reloadable(attention: Inkling2Attention) -> None:
    """Make an attention layer survive vLLM's layerwise reload (weight sync).

    ``q_scale`` / ``r_scale`` are loaded with ``lambda p, w: ...``; the reload rebinds a loader's
    ``param`` argument to the re-materialized tensor, so with a differently named argument the
    copy lands in the discarded one and the new tensor stays uninitialized.
    """
    for param in (attention.q_scale, attention.r_scale):
        param.weight_loader = _load_head_rows


def _process_attention_weights_after_loading(self, act_dtype=None) -> None:
    """``Inkling2Attention.process_weights_after_loading`` that updates ``rel_proj_shifted`` in
    place once it exists: CUDA graphs hold the tensor, so a weight sync that rebinds it would leave
    the graphs reading the old one. ``act_dtype`` is accepted for vLLMs that pass it."""
    shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
    existing = getattr(self, "rel_proj_shifted", None)
    if existing is not None and existing.shape == shifted.shape and existing.dtype == shifted.dtype:
        existing.copy_(shifted)
    else:
        self.rel_proj_shifted = shifted


Inkling2Attention.process_weights_after_loading = _process_attention_weights_after_loading

# The branch's MoE LoRA wrapper (``Inkling2MoEWithLoRA``) sizes its per-expert slots at the
# ``max_lora_rank`` of the config it is built with; with
# ``generator.inference_engine.scale_moe_lora_rank`` that is the expert rank (rank // top_k).
register_moe_layer_type(inkling2_moe.Inkling2MoE)


def _serialize_trtllm_bmm_build() -> None:
    """Take a per-node file lock around the first build of the build's ``inkling2_trtllm_bmm`` module.

    ``ops.trtllm_bmm._module`` calls FlashInfer's ``gen_trtllm_gen_fused_moe_sm100_module()``, which
    re-exports the TRT-LLM headers into the shared FlashInfer cache on every call, outside FlashInfer's
    build lock. A TP engine's workers start together and race on those files (``FileNotFoundError``
    for a header another worker is rewriting, engine startup fails). One process at a time does it;
    the rest find the export and the compiled module in place."""
    try:
        from vllm.models.inkling2.ops import trtllm_bmm
    except ImportError:  # a build without the TRT-LLM batched GEMM path
        return
    if getattr(trtllm_bmm, "_skyrl_locked", False):
        return
    import tempfile

    from filelock import FileLock

    build = trtllm_bmm._module
    lock_path = os.path.join(tempfile.gettempdir(), "skyrl_inkling2_trtllm_bmm.lock")

    @functools.cache
    def _module():
        with FileLock(lock_path):
            return build()

    trtllm_bmm._module = _module
    trtllm_bmm._skyrl_locked = True


_serialize_trtllm_bmm_build()


def _install_local_expert_lora() -> None:
    """Let the branch's MoE LoRA layer take expert adapters already cut to this rank's experts.

    Weight sync stages only this rank's slice of each ``[E, r, in]`` / ``[E, out, r]`` expert
    adapter (``SkyrlLoraStagingMixin._skyrl_lora_expert_slice``, opted into by
    ``Inkling2ForRL.skyrl_lora_local_experts``). The branch's ``_local`` slices
    ``[expert_start : expert_start + num_local_experts]`` out of a whole-layer tensor; a tensor
    that already has ``num_local_experts`` rows is this rank's slice and is used as is (when
    every expert is local the two are the same tensor).
    """
    try:
        from vllm.models.inkling2 import lora as inkling2_lora
    except ImportError:  # a branch without LoRA support
        return
    layer = inkling2_lora.Inkling2MoEWithLoRA
    if getattr(layer, "_skyrl_local_experts", False):
        return
    branch_local = layer._local

    def _local(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[0] == self.num_local_experts:
            return tensor
        return branch_local(self, tensor)

    layer._local = _local
    layer._skyrl_local_experts = True


_install_local_expert_lora()


class Inkling2ForRL(Inkling2ForConditionalGeneration):
    """``Inkling2ForConditionalGeneration`` with BF16 experts and a ``load_weights`` that takes any
    subset of the checkpoint (weight sync streams chunks)."""

    # Weight sync loads straight into the parameters (SkyrlReceiveLifecycleMixin), refreshing what
    # is derived after loading (``_refresh_derived``); vLLM's layerwise reload, which re-materializes
    # each layer next to the live one, does not fit beside the BF16 model at TP8.
    skyrl_load_weights_in_place = os.environ.get("SKYRL_INKLING2_QUANTIZED_EXPERTS") != "1"
    # LoRA weight sync may stage only this rank's experts of each expert adapter
    # (see _install_local_expert_lora).
    skyrl_lora_local_experts = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        if os.environ.get("SKYRL_INKLING2_QUANTIZED_EXPERTS") == "1":
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        else:
            with _rollout_experts():
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        for module in self.modules():
            if isinstance(module, Inkling2Attention):
                _make_reloadable(module)
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, _CapturedRouting):
                layer.mlp.layer_id = index
        self._keep_comm_state_across_sleep()

    # Small device state of the tensor-parallel communication kernels: generation flags, barrier
    # epochs and peer-buffer pointer tables. Their large symmetric / multicast buffers are allocated
    # outside vLLM's memory pool and survive sleep; these are plain tensors created with the model.
    _COMM_STATE_OWNERS = ("seq_parallel", "allreduce_workspace")
    _COMM_STATE_DTYPES = (torch.int32, torch.int64, torch.uint32)

    def _keep_comm_state_across_sleep(self) -> None:
        """Register the communication kernels' state tensors as (non-persistent) buffers.

        They are allocated in vLLM's ``weights`` memory pool, which a level-2 sleep (full-weight
        sync) discards. vLLM saves and restores the model's buffers around that sleep, but these are
        plain attributes of non-module objects, so they would come back as garbage: the fused
        all-reduce and the sequence-parallel barriers then spin on stale flags and the engine hangs
        on the first step after wake. As buffers they are restored with the rest."""
        holder = nn.Module()
        for owner_name in self._COMM_STATE_OWNERS:
            owner = getattr(self.model, owner_name, None)
            if owner is None:
                continue
            for attr, value in vars(owner).items():
                if isinstance(value, torch.Tensor) and value.is_cuda and value.dtype in self._COMM_STATE_DTYPES:
                    holder.register_buffer(f"{owner_name}__{attr}", value, persistent=False)
        if holder._buffers:
            self.skyrl_comm_state = holder

    def _dequantized_experts(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Pass ``weights`` through, replacing each packed (NVFP4 / MXFP8) expert weight and its
        scale tensors by this rank's experts dequantized to BF16, once all of them have arrived.
        """
        pending: dict = {}
        for name, weight in weights:
            m = _PACKED_EXPERT_RE.search(name)
            if m is None or (m.group(2) is None and weight.dtype not in _QUANT_COMPANIONS):
                yield name, weight
                continue
            base = name[: m.end(1)]
            parts = pending.setdefault(base, {})
            parts[name] = weight
            packed = parts.get(base)
            if packed is None or any(base + s not in parts for s in _QUANT_COMPANIONS[packed.dtype]):
                continue
            del pending[base]
            experts = self.text_config.n_routed_experts // get_tensor_model_parallel_world_size()
            sl = slice(get_tensor_model_parallel_rank() * experts, (get_tensor_model_parallel_rank() + 1) * experts)
            device = torch.cuda.current_device()
            local = {k: (v if k.endswith("_outer_scale_denominator") else v[sl]).to(device) for k, v in parts.items()}
            yield base, _LocalExperts(dequantize_expert_weight(base, local))
        if pending:
            raise ValueError(f"Packed expert weights without all their scales: {sorted(pending)}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if os.environ.get("SKYRL_INKLING2_QUANTIZED_EXPERTS") == "1":
            return super().load_weights(weights)
        # LoRA wrappers (vLLM's manager) hold the base parameters under ``base_layer``.
        params = {n.replace(".base_layer.", "."): p for n, p in self.named_parameters()}
        num_layers = self.text_config.num_hidden_layers
        layer_limits = {"layers": num_layers, "attn_queries": num_layers - 1, "mlp_queries": num_layers}
        loaded: set[str] = set()
        visual_weights = []
        for name, weight in self._dequantized_experts(weights):
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            single = _SINGLE_EXPERT_RE.match(name)
            if single is not None:
                name = f"{single.group(1)}.{single.group(3)}"
            if name not in params:
                m = _LAYER_INDEX_RE.search(name)
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue  # a truncated model (fewer layers) skips the rest
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            bf16_source = getattr(param, "skyrl_bf16_source", False)
            if isinstance(weight, _LocalExperts):
                if bf16_source:
                    param.weight_loader(param, weight.local, local=True)
                    loaded.add(name + "_scale")  # derived by the loader
                else:
                    param.data.copy_(weight.local)
                loaded.add(name)
                continue
            if single is not None:
                expert = int(single.group(2))
                if self.get_submodule(single.group(1)).is_local(expert):
                    param.weight_loader(param, weight if bf16_source else weight.to(param.dtype), expert)
                loaded.add(name)
                if bf16_source:
                    loaded.add(name + "_scale")
                continue
            if bf16_source:
                param.weight_loader(param, weight)
                loaded |= {name, name + "_scale"}
                continue
            if param.dtype != weight.dtype:
                if not (param.dtype.is_floating_point and weight.dtype.is_floating_point) or param.element_size() < 2:
                    raise ValueError(f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}")
                weight = weight.to(param.dtype)
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(param, weight)
            else:
                loader(param, weight, shard_id)
            loaded.add(name)
        if self.visual is not None and visual_weights:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        self._zero_alignment_shards(loaded)
        self._refresh_derived(loaded)
        return loaded

    def _zero_alignment_shards(self, loaded: set[str]) -> None:
        """Zero ``qkvr``'s trailing alignment shard (``Inkling2Attention`` pads its fused output to
        a multiple of 8 columns; the extra columns are discarded). No checkpoint tensor fills it,
        and vLLM's loader rejects a stacked linear with an unloaded shard when the checkpoint is
        unquantized (the BF16 checkpoints SkyRL trains)."""
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        for name in [n for n in loaded if n.endswith(".attn.qkvr.weight")]:
            attention = self.get_submodule(name.removesuffix(".qkvr.weight"))
            # Shards 0..5 are checkpoint tensors (the branch's ``_shards``); the pad shard follows.
            pad_shard = sum(new == ".attn.qkvr." for new, _ in self._shards.values())
            if len(attention.split_sizes) <= pad_shard:
                continue
            qkvr = getattr(attention.qkvr, "base_layer", attention.qkvr)
            param = qkvr.weight
            zeros = torch.zeros(
                attention.split_sizes[pad_shard] * tp_size, param.shape[1], dtype=param.dtype, device=param.device
            )
            param.weight_loader(param, zeros, pad_shard)

    def _refresh_derived(self, loaded: set[str]) -> None:
        """Recompute what ``process_weights_after_loading`` derives from reloaded weights, in place
        (CUDA graphs hold the tensors). Weight sync loads in place, without that pass; on the
        initial load it has not run yet and runs after this."""
        for name in loaded:
            if name.endswith(".attn.rel_logits_proj"):
                attention = self.get_submodule(name.removesuffix(".rel_logits_proj"))
                shifted = getattr(attention, "rel_proj_shifted", None)
                if shifted is not None:
                    shifted.copy_(shift_rel_proj(attention.rel_logits_proj, attention.is_local))
