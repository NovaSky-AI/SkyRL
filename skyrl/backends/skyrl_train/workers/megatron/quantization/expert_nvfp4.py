"""Routed-expert NVFP4 training (``trainer.policy.megatron_config.expert_nvfp4``) on the rollouts' grid, in two modes
(``expert_nvfp4_mode``). The rollouts serve W4A4 either way; the mode names what the *trainer* rounds.

``w4a16`` (the default): the routed-expert weights, in every fprop and dgrad GEMM. A routed-expert
``TEGroupedLinear``'s fprop and dgrad use ``dequant(quant(W))`` in place of ``W``, while wgrad (which does not read
``W``) and the optimizer update stay on the BF16/fp32 masters. Training stays BF16; only the weight operand is
rounded.

Grid: the rollouts' own quantizer, FlashInfer's ``fp4_quantize`` (the ModelOpt weight sync calls the same
``kernels.nvfp4_quant.quantize_experts``): per expert projection one fp32 global scale over the whole weight
(``6 * 448 / amax``, vLLM's arithmetic; gate and up share one, as in vLLM's fused ``w13``), E4M3 scales per 16
elements along K, E2M1 values, round-to-nearest. The block scale is ``expert_nvfp4_block_scale``: ``absmax``
(amax/6) or ``four_over_six`` (FlashInfer's 4over6: amax/6 or amax/4, lower squared error on its fast error path).
All experts of a grouped GEMM go through one call, read as one ``(E, N, K)`` tensor that is a view of Megatron's
param buffer where it can be (``kernels.nvfp4_quant.fake_experts``); the bf16 round trip is the decode of its bytes.

Hooked at TE's ``general_grouped_gemm`` rather than as an autograd function around the weight: the parameters still
reach TE unchanged, so its fused wgrad accumulation (``main_grad``, ``grad_added_to_main_grad``) and Megatron's grad
hooks are untouched, and the gradient is straight-through by construction. The dequantized copy is transient: it is
not saved for backward, and dgrad re-quantizes. The routed experts of the BF16 layers
(``num_layers_at_{start,end}_in_bf16``) are left alone.

``w4a4``: the per-token NVFP4 inputs of both expert GEMMs too, as W4A4 rollouts (FlashInfer TRT-LLM) quantize them:
per token (row) one fp32 scale ``amax / (6 * 448)``, E4M3 scales per 16 elements, E2M1 values, round-to-nearest,
the same block scale. A forward pre-hook rounds each routed-expert ``TEGroupedLinear``'s input in place,
straight-through (:func:`pack_input_pre_hook`): dgrad passes through the rounding unchanged, wgrad reads the rounded
input (TE saves it; no second copy exists). The fprops run on NVFP4 tensor cores: inputs and weights packed on
exactly that grid and multiplied by a grouped NVFP4 GEMM, accumulating the exact products in fp32 as vLLM's W4A4
kernel does (:func:`fp4_fprop`). dgrad and wgrad stay bf16, on the round trips. Any token dispatcher.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import torch
from loguru import logger

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (
    BLOCK_SCALE_ABSMAX,
    BLOCK_SCALE_FOUR_OVER_SIX,
    NVFP4_GROUP_SIZE,
    check_block_scale,
)

EXPERT_NVFP4_MODE_W4A16 = "w4a16"
EXPERT_NVFP4_MODE_W4A4 = "w4a4"
EXPERT_NVFP4_MODES = (EXPERT_NVFP4_MODE_W4A16, EXPERT_NVFP4_MODE_W4A4)
# TEGroupedMLP's routed-expert grouped linears.
FC1_NAME = "linear_fc1"
FC2_NAME = "linear_fc2"
# Blackwell compute capabilities with NVFP4 tensor cores and FlashInfer's NVFP4 quantizers.
NVFP4_DEVICE_CAPABILITIES = ((10, 0), (10, 3))

# Routed-expert weight params by id(); the value pins the object, so an id is never reused.
_ROUTED_EXPERT_WEIGHTS: dict[int, torch.nn.Parameter] = {}
# Routed-expert TEGroupedLinears with an input rounding hook, by id().
_INPUT_ROUNDING_MODULES: dict[int, torch.nn.Module] = {}


def fake_nvfp4_quantize_weights(weights: Sequence[torch.Tensor], block_scale: str) -> list[torch.Tensor]:
    """``dequant(quant(w))`` of each ``[N, K]`` expert weight on the rollouts' NVFP4 grid, in one
    fused round trip; one global scale per expert."""
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fake_experts import (
        fake_quantize_expert_weights,
    )

    check_block_scale(block_scale)
    with torch.no_grad():
        return fake_quantize_expert_weights([w.detach() for w in weights], block_scale)


def fake_nvfp4_quantize_activations(
    x: torch.Tensor, block_scale: str = BLOCK_SCALE_ABSMAX, inplace: bool = False
) -> torch.Tensor:
    """``dequant(quant(x))`` on vLLM's per-token NVFP4 activation grid (one scale per row of ``x``),
    in bf16: the decode of FlashInfer's ``nvfp4_quantize(per_token_activation=True)`` bytes, with its
    4over6 fast path for ``four_over_six``. ``inplace`` overwrites ``x``."""
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.per_token import (
        fake_quantize_per_token,
    )

    four_over_six = check_block_scale(block_scale) == BLOCK_SCALE_FOUR_OVER_SIX
    with torch.no_grad():
        rows = x.detach().view(-1, x.shape[-1])
        y = fake_quantize_per_token(rows, inplace=inplace, four_over_six=four_over_six)
    return x if inplace else y.view(x.shape)


def fprop_packed_weights(weights: Sequence[torch.Tensor], block_scale: str):
    """``fake_experts.pack_expert_weights`` of ``weights``: the NVFP4 tensor-core operand of a routed-expert fprop."""
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fake_experts import (
        pack_expert_weights,
    )

    return pack_expert_weights([w.detach() for w in weights], block_scale)


def fp4_fprop(
    weights: list[torch.Tensor],
    inputs: list[torch.Tensor],
    out: torch.Tensor,
    m_splits: list[int],
    bf16_gemm: Callable[..., Any],
    block_scale: str,
) -> None:
    """Routed-expert fprop on NVFP4 tensor cores: the expert inputs (per token) and weights packed on
    exactly the fake-quant grid, one grouped NVFP4 GEMM, bf16 ``out``. The module's :func:`pack_input_pre_hook`
    already overwrote the input with its round trip, so TE saves the rounded input for wgrad, and packed it
    (:func:`take_fprop_operand`). Expert shapes the packed GEMM cannot take (``N % 128`` or ``K % 64``) run the bf16
    GEMM of the same round trips instead."""
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fp4_grouped_gemm import (
        fp4_grouped_gemm,
    )

    with torch.no_grad():
        n, k = weights[0].shape
        if n % 128 or k % 64:
            _warn_fp4_fallback(n, k)
            bf16_gemm(fake_nvfp4_quantize_weights(weights, block_scale), inputs)
            return
        x_packed, layout = take_fprop_operand(inputs, m_splits)
        w_packed = fprop_packed_weights(weights, block_scale)
        fp4_grouped_gemm(x_packed, w_packed, layout, out)


@functools.cache
def _warn_fp4_fallback(n: int, k: int) -> None:
    logger.warning(f"expert_nvfp4 w4a4: a [{n}, {k}] expert shard needs N % 128 == 0 and K % 64 == 0; bf16 GEMM")


def _is_registered(weight: Any) -> bool:
    return _ROUTED_EXPERT_WEIGHTS.get(id(weight)) is weight


# fprop(weights, inputs, out, m_splits, bf16_gemm): bf16_gemm(A, B) runs TE's own fprop GEMM on given operands.
Fprop = Callable[[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[int], Callable[..., Any]], None]


def wrap_grouped_gemm(
    gemm: Callable[..., Any],
    quantize_weights: Callable[[list[torch.Tensor]], list[torch.Tensor]],
    fprop: Fprop | None = None,
) -> Callable[..., Any]:
    """``gemm`` with ``quantize_weights`` applied to the A operand (the weights, all experts at once)
    of routed-expert GEMMs. ``fprop`` (if given) replaces the routed-expert fprop GEMM entirely:
    ``fprop(weights, inputs, out, m_splits)`` writes TE's single bf16 output.

    fprop (``TN``) and dgrad (``NN``) pass the weights as A; wgrad (``NT``) passes the inputs, so it
    never matches.
    """

    def _gemm(A, B, *args, layout="TN", **kwargs):
        hits = [_is_registered(a) for a in A]
        if any(hits):
            if not all(hits):
                raise RuntimeError(
                    "expert_nvfp4: a grouped GEMM mixes routed-expert and other weights "
                    f"({sum(hits)} of {len(hits)} registered)"
                )
            if fprop is not None and layout == "TN":
                return _run_fprop(fprop, gemm, list(A), list(B), args, kwargs)
            A = quantize_weights(list(A))
        return gemm(A, B, *args, layout=layout, **kwargs)

    _gemm.__wrapped__ = gemm
    return _gemm


def _run_fprop(fprop: Fprop, gemm: Callable[..., Any], weights: list, inputs: list, args: tuple, kwargs: dict):
    """TE's fprop call is ``(weights, inputmats, [out], quantizers, dtype, single_output=True, m_splits=...,
    bias=..., use_bias=..., ...)``; the replacement handles exactly that: one bf16 output, no bias or
    output quantization."""
    out, quantizers, out_dtype = args[:3]
    if (
        len(args) != 3
        or not kwargs["single_output"]
        or kwargs["use_bias"]
        or any(q is not None for q in quantizers)
        or out_dtype != torch.bfloat16
    ):
        raise RuntimeError(
            "expert_nvfp4 w4a4 handles TE's bf16 single-output, bias-free routed-expert fprop only; got "
            f"{len(args)} positional args, single_output={kwargs['single_output']}, use_bias={kwargs['use_bias']}, "
            f"out_dtype={out_dtype}"
        )
    fprop(weights, inputs, out[0], list(kwargs["m_splits"]), lambda A, B: gemm(A, B, *args, **kwargs))
    return out, None, None


class _RoundInputInPlace(torch.autograd.Function):
    """Straight-through rounding in place: forward overwrites ``x`` with ``quantize_(x)`` (no second
    copy of the expert input), backward is the identity. Autograd's version check fails loudly if
    anything had saved ``x`` for backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, quantize_: Callable[[torch.Tensor], None]) -> torch.Tensor:
        quantize_(x)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def round_input_pre_hook(quantize_: Callable[[torch.Tensor], None]):
    """Forward pre-hook rounding a ``TEGroupedLinear``'s input ``(x, m_splits)`` in place,
    straight-through; ``quantize_`` overwrites its argument."""

    def hook(module: torch.nn.Module, args: tuple) -> tuple:
        x, *rest = args
        return (_RoundInputInPlace.apply(x, quantize_), *rest)

    return hook


# w4a4: the routed-expert GEMM input's NVFP4 operand, packed by its module's forward pre-hook (which also rounds the
# input in place, before TE takes views of it) for the fprop GEMM that follows. One slot; it pins the input it
# belongs to.
_fprop_operand: tuple | None = None


def pack_input_pre_hook(block_scale: str):
    """Forward pre-hook for ``w4a4``: packs a routed-expert ``TEGroupedLinear``'s input ``(x, m_splits)`` into the
    NVFP4 GEMM operand (FlashInfer's quantizer) and overwrites ``x`` with its round trip, straight-through, as
    :func:`round_input_pre_hook` does. Rounding here, before the module runs, keeps the in-place write off the views
    TE saves for wgrad; writing back from inside TE's forward (after it split the input) leaves those saved views
    stale for the recompute's backward."""
    four_over_six = check_block_scale(block_scale) == BLOCK_SCALE_FOUR_OVER_SIX

    def hook(module: torch.nn.Module, args: tuple) -> tuple:
        x, m_splits, *rest = args
        n, k = module._get_weight_tensors()[0].shape

        def pack_(t: torch.Tensor) -> None:
            global _fprop_operand
            from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fp4_grouped_gemm import (
                tile_layout,
            )
            from skyrl.backends.skyrl_train.kernels.nvfp4_quant.per_token import (
                pack_per_token,
            )

            rows = t.view(-1, t.shape[-1])
            if n % 128 or k % 64 or not rows.is_contiguous():
                # The fprop runs the bf16 GEMM of the round trips (_warn_fp4_fallback): round only.
                fake_nvfp4_quantize_activations(t, block_scale, inplace=True)
                return
            splits = tuple(int(m) for m in m_splits)
            layout = tile_layout(splits, t.device)
            with torch.no_grad():
                packed = pack_per_token(rows, layout.dst, layout.padded_rows, four_over_six, write_back=True)
            _fprop_operand = (rows, splits, packed, layout)

        return (_RoundInputInPlace.apply(x, pack_), m_splits, *rest)

    return hook


def same_rows(rows: torch.Tensor, inputs: Sequence[torch.Tensor]) -> bool:
    """Whether TE's fprop input splits ``inputs`` are the consecutive row blocks of ``rows`` ``[M, K]``."""
    if sum(t.shape[0] for t in inputs) != rows.shape[0] or any(t.shape[1:] != rows.shape[1:] for t in inputs):
        return False
    first = next((t for t in inputs if t.numel()), None)
    return first is None or first.data_ptr() == rows.data_ptr()


def take_fprop_operand(inputs: Sequence[torch.Tensor], m_splits) -> tuple:
    """Remove and return ``(packed, layout)`` that :func:`pack_input_pre_hook` built for the fprop input
    splits ``inputs``; raises if it is missing or belongs to other rows."""
    global _fprop_operand
    op, _fprop_operand = _fprop_operand, None
    splits = tuple(int(m) for m in m_splits)
    if op is None:
        raise RuntimeError(
            "expert_nvfp4 w4a4: no packed NVFP4 operand for this routed-expert fprop (its module's input pre-hook "
            "did not run; register_routed_expert_input_rounding(..., pack_fp4=True))"
        )
    rows, op_splits, packed, layout = op
    if op_splits != splits or not same_rows(rows, inputs):
        raise RuntimeError(
            "expert_nvfp4 w4a4: the packed NVFP4 operand belongs to other rows "
            f"({tuple(rows.shape)}, m_splits {op_splits}) than this fprop's (m_splits {splits})"
        )
    return packed, layout


def check_supported_transformer_config(config: Any) -> None:
    """Reject the Megatron settings under which routed-expert GEMMs bypass the per-expert
    ``general_grouped_gemm`` call, or receive already-quantized weights the hook would not recognize."""
    if config.fp8 is not None or config.fp4 is not None:
        raise ValueError(
            f"expert_nvfp4 needs BF16 expert compute, got transformer config fp8={config.fp8!r} fp4={config.fp4!r}"
        )
    if config.moe_use_grouped_tensor or config.moe_single_grouped_weight or config.use_transformer_engine_op_fuser:
        raise ValueError(
            "expert_nvfp4 supports TE's per-expert grouped GEMM only; unset moe_use_grouped_tensor, "
            "moe_single_grouped_weight and use_transformer_engine_op_fuser"
        )


def _check_grouped_linear(module: Any, name: str, params_dtype: torch.dtype) -> list[torch.nn.Parameter]:
    """The per-expert weight parameters of a routed-expert ``TEGroupedLinear``."""
    if module.use_grouped_tensor or module.single_grouped_weight:
        raise ValueError(
            f"expert_nvfp4 does not support TE's grouped-tensor path ({name}); unset "
            "moe_use_grouped_tensor / moe_single_grouped_weight and NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"
        )
    weights = module._get_weight_tensors()
    for w in weights:
        if not isinstance(w, torch.nn.Parameter) or w.dtype != params_dtype:
            raise ValueError(
                f"expert_nvfp4 needs {params_dtype} expert weight parameters, got {type(w).__name__} "
                f"{w.dtype} in {name}"
            )
        rows, cols = w.shape
        if rows % NVFP4_GROUP_SIZE or cols % NVFP4_GROUP_SIZE:
            raise ValueError(
                f"NVFP4 needs expert weight dims divisible by {NVFP4_GROUP_SIZE}, got {tuple(w.shape)} in {name}"
            )
    return weights


def nvfp4_modules(model_chunks: Iterable[torch.nn.Module], cls: type, bf16_layers: Iterable[int] = ()) -> Iterator:
    """``(name, module)`` of every ``cls`` module of ``model_chunks`` outside the BF16 layers: ``bf16_layers`` are
    global 0-based layer indices (``TransformerLayer.layer_number - 1``, so pipeline stages and custom layouts map
    to the HF layer numbers the vLLM ignore list and the weight sync use)."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    skip = frozenset(bf16_layers)
    for chunk in model_chunks:
        layer, prefix = None, None
        for name, module in chunk.named_modules():
            if isinstance(module, TransformerLayer):
                layer, prefix = module.layer_number - 1, f"{name}."
            if isinstance(module, cls):
                if prefix is not None and name.startswith(prefix) and layer in skip:
                    continue
                yield name, module


def routed_expert_grouped_linears(
    model_chunks: Iterable[torch.nn.Module], bf16_layers: Iterable[int] = ()
) -> list[tuple[str, Any]]:
    """``(name, TEGroupedLinear)`` of every routed expert projection outside the BF16 layers: the grouped linears
    inside a ``TEGroupedMLP`` (shared experts and dense MLPs use ``TELinear``, which is left alone). Raises if an
    MoE layer runs its experts some other way (e.g. ``SequentialMLP``, ``moe_grouped_gemm=False``)."""
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer

    found: list[tuple[str, Any]] = []
    for name, module in nvfp4_modules(model_chunks, MoELayer, bf16_layers):
        if not isinstance(module.experts, TEGroupedMLP):
            raise ValueError(
                f"expert_nvfp4 needs TEGroupedMLP routed experts (moe_grouped_gemm=True); "
                f"{name}.experts is {type(module.experts).__name__}"
            )
        check_supported_transformer_config(module.experts.config)
        found += [
            (f"{name}.experts.{sub_name}", sub)
            for sub_name, sub in module.experts.named_modules()
            if isinstance(sub, TEGroupedLinear)
        ]
    return found


def register_routed_expert_weights(
    model_chunks: Sequence[torch.nn.Module], params_dtype: torch.dtype, bf16_layers: Iterable[int] = ()
) -> int:
    """Register every routed-expert weight of ``model_chunks`` outside ``bf16_layers`` for rounding; returns the
    count."""
    count = 0
    for name, module in routed_expert_grouped_linears(model_chunks, bf16_layers):
        for w in _check_grouped_linear(module, name, params_dtype):
            _ROUTED_EXPERT_WEIGHTS[id(w)] = w
            count += 1
    return count


def check_supported_input_rounding_config(config: Any) -> None:
    """Reject Megatron settings that release or refill a routed-expert GEMM input after the in-place
    rounding: ``moe_act`` recompute rewrites the fc2 input with the unrounded activation before wgrad,
    and expert_fc1 / moe_act offloading force-release the expert inputs."""
    if config.recompute_granularity == "selective" and "moe_act" in (config.recompute_modules or []):
        raise ValueError(
            "expert_nvfp4_mode=w4a4 rounds the routed-expert inputs in place and does not support "
            "recompute_modules=[moe_act]; use full recompute or expert_nvfp4_mode=w4a16"
        )
    offloaded = {"expert_fc1", "moe_act"} & set(config.offload_modules or [])
    if config.fine_grained_activation_offloading and offloaded:
        raise ValueError(
            "expert_nvfp4_mode=w4a4 rounds the routed-expert inputs in place and does not support "
            f"offload_modules {sorted(offloaded)}"
        )


def register_routed_expert_input_rounding(
    model_chunks: Sequence[torch.nn.Module],
    block_scale: str = BLOCK_SCALE_ABSMAX,
    pack_fp4: bool = False,
    bf16_layers: Iterable[int] = (),
) -> int:
    """Round the input of every routed-expert ``TEGroupedLinear`` of ``model_chunks`` in place,
    straight-through (wgrad reads the rounded input); returns the number of hooked modules.
    ``pack_fp4`` (``w4a4``'s NVFP4 fprop): the hook also packs the NVFP4 GEMM operand (:func:`pack_input_pre_hook`).
    """
    check_block_scale(block_scale)

    def round_(x: torch.Tensor) -> None:
        fake_nvfp4_quantize_activations(x, block_scale, inplace=True)

    count = 0
    for _, module in routed_expert_grouped_linears(model_chunks, bf16_layers):
        check_supported_input_rounding_config(module.config)
        if id(module) not in _INPUT_ROUNDING_MODULES:
            hook = pack_input_pre_hook(block_scale) if pack_fp4 else round_input_pre_hook(round_)
            module.register_forward_pre_hook(hook)
            _INPUT_ROUNDING_MODULES[id(module)] = module
        count += 1
    return count


def install_grouped_gemm_hook(block_scale: str, fp4_forward: bool = False) -> None:
    """Hook TE ``GroupedLinear``'s grouped GEMM; a later call replaces the hook. Weights are opted in separately by
    :func:`register_routed_expert_weights`. ``fp4_forward``: the routed-expert fprop runs on NVFP4 tensor cores
    (:func:`fp4_fprop`; the inputs' pre-hook packs them)."""
    check_block_scale(block_scale)
    from transformer_engine.pytorch.module import grouped_linear

    gemm = grouped_linear.general_grouped_gemm
    grouped_linear.general_grouped_gemm = wrap_grouped_gemm(
        getattr(gemm, "__wrapped__", gemm),
        lambda weights: fake_nvfp4_quantize_weights(weights, block_scale),
        functools.partial(fp4_fprop, block_scale=block_scale) if fp4_forward else None,
    )


def install_expert_nvfp4(
    model_chunks: Sequence[torch.nn.Module], mode: str, block_scale: str, bf16_layers: Iterable[int] = ()
) -> int:
    """``expert_nvfp4`` on a built model (outside ``bf16_layers``): ``w4a16`` rounds the routed-expert weights,
    ``w4a4`` the per-token expert inputs as well (see the module docstring). Returns the number of rounded weights."""
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant import fused_rt

    if mode not in EXPERT_NVFP4_MODES:
        raise ValueError(f"expert_nvfp4_mode must be one of {EXPERT_NVFP4_MODES}, got {mode!r}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability() not in NVFP4_DEVICE_CAPABILITIES:
        raise RuntimeError(f"expert_nvfp4 requires SM100/SM103, got {torch.cuda.get_device_capability()}")
    bf16_layers = sorted(bf16_layers)
    w4a4 = mode == EXPERT_NVFP4_MODE_W4A4
    install_grouped_gemm_hook(block_scale, fp4_forward=w4a4)
    count = register_routed_expert_weights(model_chunks, torch.bfloat16, bf16_layers)
    if w4a4:
        # The pre-hooks round the inputs in place and pack them for the NVFP4 fprop.
        register_routed_expert_input_rounding(model_chunks, block_scale, pack_fp4=True, bf16_layers=bf16_layers)
    # Builds the fused round-trip kernels now (once per node, a few seconds) rather than in the first step.
    fused = torch.cuda.is_available() and fused_rt.available()
    logger.info(
        f"expert_nvfp4 ({mode}): {count} routed-expert weights rounded to NVFP4 (block_scale={block_scale}"
        + (", inputs per token" if w4a4 else "")
        + f"; BF16 layers {bf16_layers}); fprop on "
        + ("NVFP4 tensor cores" if w4a4 else "bf16 GEMMs of the round trips")
        + f"; round trips and packs on {'the fused sm_100 kernels' if fused else 'FlashInfer + the Triton decode'}."
    )
    return count
