"""Serialized NVFP4 weight sync for routed MoE experts (trainer-quantized, loaded as-is by vLLM).

With ``expert_nvfp4`` the trainer computes its routed experts on vLLM's NVFP4 grid, and the sync quantizes each
exported BF16 expert weight with the same quantizer (FlashInfer ``fp4_quantize`` with vLLM's per-expert global
scale, ``kernels.nvfp4_quant.quantize_experts``, and ``expert_nvfp4_block_scale``) and ships ModelOpt NVFP4
checkpoint tensors, so the sampler's expert weights are the bytes the trainer rounds to. vLLM boots with
``quantization=modelopt_fp4`` and dummy weights; its layerwise reload only re-lays the tensors out for the
TRT-LLM MoE kernel, so no re-quantization happens on the sampler.

Per expert projection the stream carries (HF names, as a ModelOpt checkpoint would):
  ``<p>.weight``         uint8 ``[N, K/2]``  packed E2M1, even K element in the low nibble
  ``<p>.weight_scale``   float8_e4m3fn ``[N, K/16]`` block scales, row-major (unswizzled)
  ``<p>.weight_scale_2`` fp32 scalar, global decode scale ``amax / (6 * 448)``
  ``<p>.input_scale``    fp32 scalar 1.0 (unused under per-token activations; keeps reload layerwise)
Gate and up of one expert are quantized as one ``[gate; up]`` matrix, matching Megatron's fused fc1
weight (one global scale per expert) and vLLM's requirement that w1/w3 share one global scale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Tuple

import torch

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (
    NVFP4_GROUP_SIZE,
    check_block_scale,
    nvfp4_amax,
    nvfp4_global_scale,
    quantize_experts,
)

NVFP4_BLOCK_SIZE = NVFP4_GROUP_SIZE
_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+))\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

# Modules that stay BF16 on the sampler: everything except the routed experts.
NVFP4_EXPERT_ONLY_IGNORE = (
    "*self_attn*",
    "*linear_attn*",
    "*mlp.gate",
    "*shared_expert*",
    "lm_head",
    "*lm_head*",
    "*visual*",
    "mtp*",
)


@dataclass(frozen=True)
class Nvfp4WeightQuantSettings:
    """The trainer's NVFP4 weight grid: ``expert_nvfp4_block_scale`` (absmax or 4/6)."""

    block_scale: str

    def __post_init__(self) -> None:
        check_block_scale(self.block_scale)


def _check_nvfp4_cols(cols: int) -> None:
    if cols % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"NVFP4 requires K divisible by {NVFP4_BLOCK_SIZE}, got {cols}")


def quantize_nvfp4_matrix(
    weight: torch.Tensor, settings: Nvfp4WeightQuantSettings
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one ``[N, K]`` matrix -> (packed uint8, e4m3 block scales, fp32 decode scale): vLLM's online
    arithmetic (one global scale over the whole matrix) and FlashInfer's quantizer, the trainer's own grid."""
    _check_nvfp4_cols(weight.shape[1])
    global_scale, decode = nvfp4_global_scale(nvfp4_amax(weight))
    packed, block_scale = quantize_experts(weight[None], global_scale.reshape(1), settings.block_scale)
    return packed[0], block_scale[0], decode


def _emit(prefix: str, proj: str, packed: torch.Tensor, block_scale: torch.Tensor, decode: torch.Tensor):
    base = f"{prefix}.{proj}"
    yield f"{base}.weight", packed
    yield f"{base}.weight_scale", block_scale
    yield f"{base}.weight_scale_2", decode.reshape(())
    yield f"{base}.input_scale", torch.ones((), dtype=torch.float32, device=decode.device)


class TensorSpec(NamedTuple):
    """Dtype and shape of a stream entry, without data (what ``metadata()`` declares)."""

    dtype: torch.dtype
    shape: Tuple[int, ...]


def _emit_spec(prefix: str, proj: str, rows: int, cols: int) -> Iterator[Tuple[str, TensorSpec]]:
    """The entries :func:`_emit` yields for a quantized ``[rows, cols]`` projection, as specs."""
    _check_nvfp4_cols(cols)
    base = f"{prefix}.{proj}"
    yield f"{base}.weight", TensorSpec(torch.uint8, (rows, cols // 2))
    yield f"{base}.weight_scale", TensorSpec(torch.float8_e4m3fn, (rows, cols // NVFP4_BLOCK_SIZE))
    yield f"{base}.weight_scale_2", TensorSpec(torch.float32, ())
    yield f"{base}.input_scale", TensorSpec(torch.float32, ())


class Nvfp4ExpertSerializer:
    """Stateful stream transform: routed-expert BF16 weights -> ModelOpt NVFP4 tensors; others pass through.

    Gate and up of an expert may arrive in either order and not adjacently, so the first one is held until its
    partner arrives. Experts of the layers in ``bf16_layers`` (see :func:`nvfp4_bf16_layers`) stay BF16.
    """

    def __init__(self, settings: Nvfp4WeightQuantSettings, bf16_layers: Iterable[int] = ()):
        self.settings = settings
        self.bf16_layers = frozenset(bf16_layers)
        self._pending: Dict[str, Tuple[str, Any]] = {}

    def __call__(self, name: str, tensor: Any) -> Iterator[Tuple[str, Any]]:
        m = _EXPERT_RE.match(name)
        if m is None or int(m.group("layer")) in self.bf16_layers:
            yield name, tensor
            return
        prefix, proj = m.group("prefix"), m.group("proj")
        if proj == "down_proj":
            yield from self._emit_single(prefix, proj, tensor)
            return
        other = self._pending.pop(prefix, None)
        if other is None:
            self._pending[prefix] = (proj, tensor)
            return
        other_proj, other_tensor = other
        gate, up = (other_tensor, tensor) if other_proj == "gate_proj" else (tensor, other_tensor)
        if tuple(gate.shape) != tuple(up.shape):
            raise ValueError(f"{prefix}: gate {tuple(gate.shape)} and up {tuple(up.shape)} must match")
        yield from self._emit_pair(prefix, gate, up)

    def _emit_single(self, prefix: str, proj: str, tensor: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from _emit(prefix, proj, *quantize_nvfp4_matrix(tensor, self.settings))

    def _emit_pair(self, prefix: str, gate: torch.Tensor, up: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        packed, block_scale, decode = quantize_nvfp4_matrix(torch.cat((gate, up), dim=0), self.settings)
        n = gate.shape[0]
        yield from _emit(prefix, "gate_proj", packed[:n].contiguous(), block_scale[:n].contiguous(), decode)
        yield from _emit(prefix, "up_proj", packed[n:].contiguous(), block_scale[n:].contiguous(), decode.clone())

    def finish(self) -> None:
        if self._pending:
            raise RuntimeError(f"NVFP4 sync: unpaired gate/up for {sorted(self._pending)[:4]} ...")


class Nvfp4ExpertSpecSerializer(Nvfp4ExpertSerializer):
    """The same stream transform over :class:`TensorSpec` inputs: names, order, dtypes and shapes, no data.

    Shares the pairing / pass-through / emission-order code with :class:`Nvfp4ExpertSerializer`; only the
    two quantizing leaves are replaced, so every rank can declare the serialized stream's metadata without
    quantizing (or even materializing) anything. ``test_nvfp4_serializer.py`` pins the two against each
    other, and the sender's ``_checked_iter`` raises on any divergence at runtime.
    """

    def _emit_single(self, prefix: str, proj: str, spec: TensorSpec) -> Iterator[Tuple[str, TensorSpec]]:
        rows, cols = spec.shape
        yield from _emit_spec(prefix, proj, rows, cols)

    def _emit_pair(self, prefix: str, gate: TensorSpec, up: TensorSpec) -> Iterator[Tuple[str, TensorSpec]]:
        rows, cols = gate.shape
        yield from _emit_spec(prefix, "gate_proj", rows, cols)
        yield from _emit_spec(prefix, "up_proj", rows, cols)


def nvfp4_bf16_layers(num_layers: int, megatron_config: Any) -> List[int]:
    """Layer indices whose routed experts stay BF16: the first ``num_layers_at_start_in_bf16`` and last
    ``num_layers_at_end_in_bf16`` layers, the same knobs (and the same Megatron ``first_last_layers_bf16``
    mechanism) as FP8 training. Accepts a ``MegatronConfig`` or a Megatron provider."""
    kwargs = getattr(megatron_config, "transformer_config_kwargs", None) or {}
    enabled = kwargs.get("first_last_layers_bf16", getattr(megatron_config, "first_last_layers_bf16", False))
    if not enabled:
        return []
    start = int(
        kwargs.get("num_layers_at_start_in_bf16", getattr(megatron_config, "num_layers_at_start_in_bf16", 0)) or 0
    )
    end = int(kwargs.get("num_layers_at_end_in_bf16", getattr(megatron_config, "num_layers_at_end_in_bf16", 0)) or 0)
    if start < 0 or end < 0 or start + end >= num_layers:
        raise ValueError(
            f"num_layers_at_start_in_bf16={start} + num_layers_at_end_in_bf16={end} must leave at least one "
            f"NVFP4 layer of {num_layers}"
        )
    return list(range(start)) + list(range(num_layers - end, num_layers))


def get_nvfp4_vllm_engine_overrides(bf16_layers: Iterable[int] = ()) -> Dict[str, Any]:
    """vLLM engine kwargs for a ModelOpt NVFP4 expert-only engine with per-token activations."""
    ignore: List[str] = list(NVFP4_EXPERT_ONLY_IGNORE)
    ignore += [f"*layers.{i}.mlp.experts*" for i in sorted(bf16_layers)]
    return {
        "quantization": "modelopt_fp4",
        "load_format": "dummy",
        # Overrides vLLM's MoE activation scheme: per-token NVFP4 (requires moe_backend=flashinfer_trtllm).
        "quantization_config": {"moe": {"activation": "nvfp4_per_token"}},
        "hf_overrides": {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "group_size": NVFP4_BLOCK_SIZE,
                "kv_cache_scheme": None,
                "ignore": ignore,
            }
        },
    }
