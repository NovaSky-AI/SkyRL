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

Looped checkpoints (``loop_config``) are served by :class:`Inkling2LoopModel` (see its docstring).

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
from vllm.models.inkling2.ops.attention import compute_log_scaling_tau
from vllm.models.inkling2.ops.block_scales import shuffle_rows, swizzle_block_scales
from vllm.models.inkling2.ops.sconv import shift_rel_proj
from vllm.models.inkling2.sconv import Inkling2ShortConv

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
    at ``layer_id`` (the decoder layer, set by :class:`Inkling2ForRL`), for rollout router replay.

    A looped model runs a core layer's MoE once per pass: :class:`Inkling2LoopModel` sets ``run_id``
    (the layer run's index in execution order) before each call, and the ids are recorded there."""

    layer_id: int = -1
    capture_fn = None
    run_id: int | None = None

    def _capture(self, ids: torch.Tensor) -> None:
        fn = self.capture_fn
        if self.run_id is not None and isinstance(fn, functools.partial):
            fn.func(self.run_id, ids)  # vLLM binds partial(capturer.capture, layer_id)
        else:
            fn(ids)


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
            self._capture(ids)
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
            self._capture(ids)
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


_UNROLLED_KERNEL_RE = re.compile(r"^t\d+x\d+x(\d+)u2_")


def _fit_mxfp8_kernels_to_shapes(hidden: int, intermediate: int) -> None:
    """Pick MXFP8 expert GEMM kernels that support this model's K dims.

    The build's ``ops.trtllm_bmm._KERNELS`` uses, for the MXFP8 gate / up projection at token tiles 16, 64 and 128, a
    kernel with a 256-wide K tile and its main loop unrolled twice (``...x256u2_...``). TRT-LLM-gen's config check
    (``GemmOptions.h``, "Size K / splitK must be a multiple of TileK * 2") rejects the unrolled loop unless
    K is a multiple of 2 * TileK: not true of every hidden size; when it fails ``expert_plan`` asserts
    (``bmm_workspace_size`` returns -1) as soon as the MoE loads its kernels. Each such kernel is swapped for its
    non-unrolled sibling (the same name without ``u2``; the table already uses it for the down projection), which
    accepts any K whose scale-factor count is a multiple of 4 (TMA zero-fills the K tail). No-op for shapes the
    table already fits. Must run before the first MoE is built (``load_kernels``)."""
    try:
        from vllm.models.inkling2.ops import trtllm_bmm
    except ImportError:  # a build without the TRT-LLM batched GEMM path
        return
    ks = (hidden, intermediate)  # K of the gate / up and of the down projection
    changed = False
    for (nvfp4, tile), names in list(trtllm_bmm._KERNELS.items()):
        if nvfp4:
            continue
        fitted = []
        for name, k in zip(names, ks):
            m = _UNROLLED_KERNEL_RE.match(name)
            if m is not None and k % (2 * int(m.group(1))):
                name = name.replace("u2_", "_", 1)
                changed = True
            fitted.append(name)
        trtllm_bmm._KERNELS[(nvfp4, tile)] = tuple(fitted)
    if changed:
        trtllm_bmm.expert_plan.cache_clear()
    _fit_combine_block(hidden)
    _fit_swiglu_quantize_block(intermediate)


def _fit_combine_block(hidden: int) -> None:
    """Let the expert combine run on hidden sizes that are not a multiple of 512.

    ``ops.moe._combine`` (behind ``combine`` / ``combine_push``) sums each token's top-k expert rows in
    512-column blocks and asserts ``hidden % 512 == 0``. The kernel is elementwise over columns, so any
    power-of-two block that divides ``hidden`` gives the same sums; the largest one up to 512 is used.
    No-op when 512 divides ``hidden``."""
    if hidden % 512 == 0:
        return
    from vllm.models.inkling2.ops import moe as moe_ops

    block = 512
    while block > 16 and hidden % block:
        block //= 2
    if hidden % block:
        raise ValueError(f"Inkling2 expert combine needs a hidden size divisible by 16, got {hidden}")
    if getattr(moe_ops._combine, "_skyrl_block", None) == block:
        return

    def _combine(down, src2dst, topk, out, sp):
        tokens = src2dst.numel() // topk
        moe_ops._combine_kernel[(tokens, hidden // block)](
            down,
            src2dst,
            out,
            None if sp is None else sp.slot_bases,
            tokens,
            1 if sp is None else sp.shard(tokens)[2],
            hidden,
            topk,
            block,
            tokens <= moe_ops.LAUNCH_DEPENDENTS_MAX_TOKENS,
            sp is not None,
            moe_ops.HALO_ROWS,
            num_warps=2,
            launch_pdl=True,
        )

    _combine._skyrl_block = block
    moe_ops._combine = _combine


def _fit_swiglu_quantize_block(intermediate: int) -> None:
    """Let the MXFP8 SwiGLU quantize run on few rows when ``intermediate`` is not a multiple of 256.

    With at most 512 routed rows, ``ops.activation.swiglu_quantize`` launches a program per 256 columns
    and asserts ``n % 256 == 0`` (MXFP8 only; more rows, and NVFP4, quantize whole rows as a power-of-two
    head and tail, which fits sizes like 3 * 2**k). The kernel quantizes in 32-column scale blocks,
    so any power-of-two column chunk of at least 32 that divides ``n`` gives the same values and scales;
    that case is launched here with the largest such chunk up to 256, everything else goes to the
    branch's function. No-op when 256 divides ``intermediate``."""
    if intermediate % 256 == 0:
        return
    from vllm.models.inkling2 import moe as moe_module
    from vllm.models.inkling2.ops import activation

    original = getattr(activation.swiglu_quantize, "_skyrl_original", activation.swiglu_quantize)
    head = 256
    while head > 32 and intermediate % head:
        head //= 2
    if intermediate % head:
        raise ValueError(f"Inkling2 MXFP8 experts need an intermediate size divisible by 32, got {intermediate}")

    def swiglu_quantize(x, gammas, gate_cap, up_cap, rows, tile, weight_amax=None, weight_denominator=None):
        num_rows, n = x.shape[0], x.shape[1] // 2
        routed, num_routed, cta_expert = rows
        max_rows = routed.numel()
        if weight_amax is not None or max_rows > 512 or n % 256 == 0 or n % head:
            return original(x, gammas, gate_cap, up_cap, rows, tile, weight_amax, weight_denominator)
        values = torch.empty((num_rows, n), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty((num_rows, n // 32), dtype=torch.uint8, device=x.device)
        activation._situ_quantize_kernel[(max_rows, n // head)](
            x,
            gammas,
            values,
            scales,
            None,
            routed,
            num_routed,
            cta_expert,
            None,
            None,
            N=n,
            GATE_CAP=gate_cap,
            UP_CAP=up_cap,
            HEAD=head,
            TAIL=0,
            TILE=tile,
            NVFP4=False,
            LAUNCH_DEPENDENTS=max_rows <= activation.LAUNCH_DEPENDENTS_MAX_ROWS,
            num_warps=2,
            maxnreg=None,
            enable_fp_fusion=False,
            launch_pdl=True,
        )
        return values, scales, None

    swiglu_quantize._skyrl_original = original
    activation.swiglu_quantize = swiglu_quantize
    moe_module.swiglu_quantize = swiglu_quantize


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


# --------------------------------------------------------------------------------------------
# Looped checkpoints
# --------------------------------------------------------------------------------------------

LOOP_MODULES = ("attn", "k", "v", "mlp")


def loop_core(text_config) -> range:
    loop = getattr(text_config, "loop_config", None)
    return range(0) if loop is None else range(loop["core_start"], loop["core_end"])


def num_layer_visits(text_config) -> int:
    loop = getattr(text_config, "loop_config", None)
    n = text_config.num_hidden_layers
    return n if loop is None else n + (loop["num_passes"] - 1) * len(loop_core(text_config))


def cache_layer_id(text_config, layer: int, pass_idx: int) -> int:
    """The KV / conv-state cache index of a layer run (the SGLang fork's ``cache_layer_id``): the layer itself in
    the first pass, ``num_hidden_layers + (pass - 1) * n_core + (layer - core_start)`` after."""
    if pass_idx == 0:
        return layer
    core = loop_core(text_config)
    return text_config.num_hidden_layers + (pass_idx - 1) * len(core) + layer - core.start


class _SharedFromBase:
    """Non-data descriptor: the attribute of the module's ``_loop_base`` once set (before, the module's own,
    so that the base class's ``__init__`` can build and use it)."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        base = obj.__dict__.get("_loop_base")
        if base is not None:
            return getattr(base, self.name)
        return nn.Module.__getattr__(obj, self.name)


def _share_base(module: nn.Module, base: nn.Module, names) -> None:
    """Drop ``module``'s own ``names`` (submodules / parameters) and read them from ``base`` from now on."""
    for name in names:
        for store in (module._modules, module._parameters, module._buffers):
            store.pop(name, None)
    object.__setattr__(module, "_loop_base", base)


class _PassAttention(Inkling2Attention):
    """A later pass of a looped core layer's attention: its own KV cache and conv state (registered under the
    pass's cache-layer name), the base layer's weights (and their LoRA wrappers and derived tables)."""

    _SHARED = ("qkvr", "wo_ud", "k_sconv", "v_sconv", "rel_logits_proj", "q_scale", "r_scale", "rel_proj_shifted")
    qkvr = _SharedFromBase()
    wo_ud = _SharedFromBase()
    k_sconv = _SharedFromBase()
    v_sconv = _SharedFromBase()
    rel_logits_proj = _SharedFromBase()
    q_scale = _SharedFromBase()
    r_scale = _SharedFromBase()
    rel_proj_shifted = _SharedFromBase()

    def __init__(self, base: Inkling2Attention, config, conv_state, prefix: str) -> None:
        super().__init__(config, base.is_local, conv_state, prefix=prefix)
        _share_base(self, base, self._SHARED)

    def process_weights_after_loading(self, *args, **kwargs) -> None:
        pass  # derived from the base layer's weights by the base layer


class _PassShortConv(Inkling2ShortConv):
    """A later pass's attn- / mlp-output short convolution: its own conv state, the base layer's taps."""

    weight = _SharedFromBase()

    def __init__(self, base: Inkling2ShortConv, owner) -> None:
        super().__init__(base.weight.shape[0], base.weight.shape[-1], owner, base.stream)
        _share_base(self, base, ("weight",))


class Inkling2LoopController(nn.Module):
    """``(1 - tanh(f)) * dest + tanh(g) * msg`` (FP32), ``[g, f] = h @ [gain, forget]^T`` (FP32 output),
    ``h = [rmsnorm(dest), rmsnorm(msg)] @ down^T`` (no gain, BF16); the SGLang fork's ``tml/kernels/loop.py``.
    Rank ``gate_rank`` is not split over TP: replicated weights."""

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        for name, shape in (
            ("down_weight", (rank, 2 * hidden)),
            ("gain_weight", (hidden, rank)),
            ("forget_weight", (hidden, rank)),
        ):
            self.register_parameter(name, nn.Parameter(torch.empty(*shape, dtype=torch.bfloat16), requires_grad=False))

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        x = torch.stack((dest.float(), message.float()), dim=1)
        normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).flatten(1)
        hidden = torch.nn.functional.linear(normalized, self.down_weight)
        gates = hidden.float() @ torch.cat([self.gain_weight, self.forget_weight]).float().t()
        gain, forget = gates.chunk(2, dim=-1)
        return (1.0 - torch.tanh(forget)) * dest.float() + torch.tanh(gain) * message.float()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMSNorm, one rounding to BF16 (``reference.rms_norm``)."""
    u = x.float()
    return (u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16)


def _mix_fp32(sources: list, addend, queries: torch.Tensor, normalize: bool) -> torch.Tensor:
    """FP32 attn-res mixtures ``[Q, T, H]`` of ``sources`` (``[T, H]`` each; ``addend`` added to the last in FP32)
    with ``queries [Q, H]`` already scaled by ``H**-0.5`` -- the fused ``attn_res`` before its norm."""
    values = torch.stack([s.float() for s in sources])
    if addend is not None:
        values = torch.cat([values[:-1], (values[-1] + addend.float()).unsqueeze(0)])
    inv_rms = torch.rsqrt(values.square().mean(-1) + 1e-6)  # [n, T]
    logits = torch.einsum("nth,qh,nt->qtn", values, queries.float(), inv_rms)
    probs = torch.softmax(logits, dim=-1)
    if normalize:
        values = values * inv_rms.unsqueeze(-1)
    return torch.einsum("qtn,nth->qth", probs, values)


class Inkling2LoopModel(inkling2_model.Inkling2Model):
    """The branch's ``Inkling2Model`` running a looped checkpoint (``loop_config``), the SGLang fork's ``_run_loop``.

    The prelude layers run once, the core layers ``[core_start, core_end)`` ``num_passes`` times with shared weights,
    the coda once. Every pass restarts the attn-res state from the prelude's; from the second pass on each core layer
    mixes messages read from the bank (the previous pass's block outputs, without the prelude's) into its attention
    q / k / v and MoE inputs (FP32, before their norms) through its controllers; local layers then project q, k and v
    from three different inputs (all through ``attn_norm``). Each later pass has its own attention KV cache and
    short-convolution state (``_PassAttention``, ``_PassShortConv``, cache index ``cache_layer_id``).

    The forward is the branch's unfused single-GPU path with explicit all-reduces (as the SGLang fork runs loops
    under tensor parallelism): no fused all-reduce / mix, no sequence-parallel token sharding. Routed-expert capture
    records every layer run, in execution order (``num_layer_visits`` entries per token).
    """

    def install_loop(self, config, root: str) -> None:
        core, hidden = loop_core(config), config.hidden_size
        loop = config.loop_config
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(2 * len(core), hidden, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(2 * len(core), hidden, dtype=torch.float32), requires_grad=False
        )
        for i in core:
            layer = self.layers[i]
            layer.controllers = nn.ModuleDict(
                {m: Inkling2LoopController(hidden, loop["gate_rank"]) for m in LOOP_MODULES}
            )
            passes = nn.ModuleList()
            for p in range(1, loop["num_passes"]):
                cid = cache_layer_id(config, i, p)
                base_state = layer.conv_state
                conv_state = type(base_state)(
                    num_kv_heads=base_state.num_kv_heads * get_tensor_model_parallel_world_size(),
                    head_dim=layer.attn.head_dim,
                    hidden_size=hidden,
                    kernel_size=config.sconv_kernel_size,
                    prefix=f"{root}.layers.{cid}.conv_state",
                    replicate_hidden=True,
                )
                assert conv_state.head_size == base_state.head_size
                run = nn.Module()
                run.conv_state = conv_state
                run.attn = _PassAttention(layer.attn, config, conv_state, f"{root}.layers.{cid}.attn")
                run.attn_sconv = _PassShortConv(layer.attn_sconv, conv_state)
                run.mlp_sconv = _PassShortConv(layer.mlp_sconv, conv_state)
                passes.append(run)
            layer.loop_passes = passes

    def _bank_messages(self, bank: torch.Tensor, offset: int) -> torch.Tensor:
        """``[4, T, H]`` FP32: the attn, k, v, mlp messages of core layer ``offset`` read from ``bank [m, T, H]``."""
        q, kv = self.middle_cycle_cross_pass_queries, self.middle_cycle_cross_pass_kv_queries
        queries = torch.stack([q[2 * offset], kv[2 * offset], kv[2 * offset + 1], q[2 * offset + 1]])
        return _mix_fp32(list(bank.unbind(0)), None, queries * self.config.hidden_size**-0.5, True)

    @staticmethod
    def _attend(attn, positions, inputs, tau):
        """``attn``'s partial output; local layers with three different inputs project like global ones."""
        if not attn.is_local:
            return attn(positions, inputs[0], inputs[1], inputs[2], tau)
        if len(inputs) == 1:
            return attn(positions, inputs[0])
        from vllm.models.inkling2.lora import add_linear_lora
        from vllm.models.inkling2.ops.linear import linear

        x_q, x_k, x_v = inputs
        weight = attn.qkvr.weight
        q_side, kv_size = sum(attn.split_sizes[:4]), attn.split_sizes[4]
        k_end = q_side + kv_size
        projected = linear(x_q, weight[: sum(attn.q_side_splits)])
        k = linear(x_k, weight[q_side:k_end])
        v = linear(x_v, weight[k_end : k_end + kv_size])
        add_linear_lora(attn.qkvr, projected, x_q, range(4))
        add_linear_lora(attn.qkvr, k, x_k, [4])
        add_linear_lora(attn.qkvr, v, x_v, [5])
        q, r, gate, sinks = projected.split(attn.q_side_splits, dim=-1)[:4]
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        attn._attention(positions, q, r, gate, sinks, k, v, None, None, None, output)
        out = linear(output, attn.wo_ud.weight)
        add_linear_lora(attn.wo_ud, out, output)
        return out

    def forward(self, positions: torch.Tensor, inputs_embeds: torch.Tensor):
        if self.aux_hidden_state_layers:
            raise NotImplementedError("Looped Inkling2 does not support draft-model hidden-state capture")
        config = self.config
        loop, core = config.loop_config, loop_core(config)
        num_tokens, hidden = inputs_embeds.shape
        num_layers = len(self.layers)
        normalize = self.normalize_values
        tau = None
        if config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(positions, config.log_scaling_n_floor, config.log_scaling_alpha)
        max_blocks = -(-num_layers // self.block_size)
        stack = inputs_embeds.new_empty((max_blocks + 1, num_tokens, hidden))
        stack[0].copy_(inputs_embeds)
        num_blocks = 0

        def run_layer(i: int, p: int, run_id: int, bank) -> int:
            """One layer run on ``stack[:num_blocks + 1]``; returns the new block count."""
            layer = self.layers[i]
            nb = num_blocks_box[0]
            partial = stack[nb]
            mods = layer if p == 0 else layer.loop_passes[p - 1]
            attn = layer.attn if p == 0 else mods.attn
            messages = None
            if bank is None:
                if nb == 0:
                    inputs = [layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)]
                else:
                    inputs = self._mix(stack[:nb], partial, self.attn_res.attn_queries[i - 1], layer.attn_norm)
            else:
                messages = self._bank_messages(bank, i - core.start)
                streams = list(
                    _mix_fp32(list(stack[: nb + 1].unbind(0)), None, self.attn_res.attn_queries[i - 1], normalize)
                )
                if layer.is_local:
                    streams = streams * 3
                streams = [layer.controllers[m](x, messages[j]) for j, (m, x) in enumerate(zip(LOOP_MODULES, streams))]
                w = layer.attn_norm.weight
                inputs = [
                    _rms_norm(x, w[0 if layer.is_local else j], layer.attn_norm.eps) for j, x in enumerate(streams)
                ]
            attn_partial = self._attend(attn, positions, inputs, None if layer.is_local else tau)
            attn_out = mods.attn_sconv(tensor_model_parallel_all_reduce(attn_partial), positions)
            first = i % self.block_size == 0
            mlp_query = self.attn_res.mlp_queries[i]
            if messages is None:
                if first:
                    mlp_in = self._mix(stack[: nb + 1], attn_out, mlp_query, layer.mlp_norm)[0]
                else:
                    mlp_in = self._mix(stack[:nb], partial, mlp_query, layer.mlp_norm, attn_out)[0]
            else:
                if first:
                    mix = _mix_fp32([*stack[: nb + 1].unbind(0), attn_out], None, mlp_query, normalize)[0]
                else:
                    mix = _mix_fp32([*stack[: nb + 1].unbind(0)], attn_out, mlp_query, normalize)[0]
                mix = layer.controllers["mlp"](mix, messages[3])
                mlp_in = _rms_norm(mix, layer.mlp_norm.weight[0], layer.mlp_norm.eps)
            residual = None if first else partial
            if first:
                nb += 1
            moe = getattr(layer.mlp, "base_layer", layer.mlp)
            moe.run_id = run_id
            mods.mlp_sconv(layer.mlp(mlp_in), positions, addend=attn_out, residual=residual, out=stack[nb])
            num_blocks_box[0] = nb
            return nb

        num_blocks_box = [num_blocks]
        run_id = 0
        for i in range(core.start):
            run_layer(i, 0, run_id, None)
            run_id += 1
        prelude_blocks = num_blocks_box[0]  # stack[:prelude_blocks + 1] is never overwritten by the core
        bank = None
        for p in range(loop["num_passes"]):
            if p:
                # The previous pass's block outputs; the next pass overwrites them in place.
                bank = stack[prelude_blocks + 1 : num_blocks_box[0] + 1].clone()
                num_blocks_box[0] = prelude_blocks
            for i in core:
                run_layer(i, p, run_id, bank)
                run_id += 1
        for i in range(core.stop, num_layers):
            run_layer(i, 0, run_id, None)
            run_id += 1
        nb = num_blocks_box[0]
        return self._mix(stack[:nb], stack[nb], self.attn_res.output_query, self.norm)[0]


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
            if ROLLOUT_EXPERTS == "mxfp8":
                text = vllm_config.model_config.hf_config.text_config
                _fit_mxfp8_kernels_to_shapes(text.hidden_size, text.intermediate_size)
            with _rollout_experts():
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        for module in self.modules():
            if isinstance(module, Inkling2Attention):
                _make_reloadable(module)
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, _CapturedRouting):
                layer.mlp.layer_id = index
        if loop_core(self.text_config):
            # Before the comm-state registration below (it only reads the model's existing attributes).
            self.model.__class__ = Inkling2LoopModel
            root = self.model.layers[0].conv_state.prefix.rsplit(".layers.", 1)[0]
            self.model.install_loop(self.text_config, root)
            for module in self.modules():
                if isinstance(module, Inkling2Attention) and not isinstance(module, _PassAttention):
                    _make_reloadable(module)
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
