"""Stacked per-layer NVFP4 expert sync (``expert_nvfp4_stacked_sync``): the wire format and the sender side.

The per-expert NVFP4 stream (``nvfp4.py``) carries 8 tensors per expert -- ~74k tensors for a 48-layer,
128-expert model -- and vLLM's layerwise reload spends its time in per-tensor Python bookkeeping
(``inspect.Signature.bind`` + a ``TorchDispatchMode`` copy counter per loader call, then a replay through
``FusedMoE.weight_loader`` per expert and shard). This mode ships each MoE layer's experts as 8 tensors already
in vLLM's ModelOpt-NVFP4 ``FusedMoE`` parameter layout, over all *global* experts:

  ``<layer>.w13_weight``          uint8          ``[E, 2N, K/2]``   per expert ``[gate; up]`` packed E2M1
  ``<layer>.w13_weight_scale``    float8_e4m3fn  ``[E, 2N, K/16]``  block scales, same row order
  ``<layer>.w13_weight_scale_2``  float32        ``[E, 2]``         decode scale of gate, up (one shared value)
  ``<layer>.w13_input_scale``     float32        ``[E, 2]``         1.0
  ``<layer>.w2_weight``           uint8          ``[E, H, N/2]``
  ``<layer>.w2_weight_scale``     float8_e4m3fn  ``[E, H, N/16]``
  ``<layer>.w2_weight_scale_2``   float32        ``[E]``
  ``<layer>.w2_input_scale``      float32        ``[E]``

named ``__skyrl_nvfp4_stacked__:<hf layer prefix>.<leaf>`` (``<layer>`` = ``model.layers.<i>.mlp.experts``).
Quantization is unchanged -- the same TE call per expert on the same ``[gate; up]`` matrix -- and each result is
written into a preallocated stack, so the bytes are exactly the per-expert stream's bytes, concatenated in the
order vLLM's ``FusedMoE.weight_loader`` would place them (gate into rows ``[0, N)``, up into ``[N, 2N)``).
BF16 layers (``num_layers_at_{start,end}_in_bf16``) are untouched, as are all non-expert tensors.

The receiver is ``nvfp4_stacked_receive.py``; ``unstack_to_per_expert`` turns a stacked layer back into the
per-expert stream (its fallback, and the tests' reference).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from skyrl.backends.skyrl_train.weight_sync.nvfp4 import (
    NVFP4_BLOCK_SIZE,
    Nvfp4ExpertSerializer,
    Nvfp4WeightQuantSettings,
    TensorSpec,
    _check_nvfp4_cols,
    quantize_nvfp4_matrix,
)

NVFP4_STACKED_PREFIX = "__skyrl_nvfp4_stacked__:"
# Emission order of one layer's stacked tensors.
STACKED_LEAVES: Tuple[str, ...] = (
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_scale_2",
    "w13_input_scale",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_scale_2",
    "w2_input_scale",
)
_EXPERT_PREFIX_RE = re.compile(r"^(?P<layer>.*\.layers\.\d+\.mlp\.experts)\.(?P<expert>\d+)$")


def stacked_wire_name(layer_prefix: str, leaf: str) -> str:
    return f"{NVFP4_STACKED_PREFIX}{layer_prefix}.{leaf}"


def parse_stacked_wire_name(name: str) -> Optional[Tuple[str, str]]:
    """``(hf layer prefix, leaf)`` for a stacked wire name, else None."""
    if not name.startswith(NVFP4_STACKED_PREFIX):
        return None
    layer_prefix, _, leaf = name[len(NVFP4_STACKED_PREFIX) :].rpartition(".")
    if leaf not in STACKED_LEAVES or not layer_prefix:
        raise ValueError(f"Malformed NVFP4 stacked wire name {name!r}")
    return layer_prefix, leaf


def _split_expert_prefix(prefix: str) -> Tuple[str, int]:
    m = _EXPERT_PREFIX_RE.match(prefix)
    if m is None:
        raise ValueError(f"Cannot split expert prefix {prefix!r} into layer and expert id")
    return m.group("layer"), int(m.group("expert"))


class _LayerStack:
    """One MoE layer's stacks, filled expert by expert. ``spec=True`` tracks shapes only (metadata pass)."""

    def __init__(self, layer_prefix: str, num_experts: int, spec: bool):
        self.layer_prefix = layer_prefix
        self.num_experts = num_experts
        self.spec = spec
        self.w13: Optional[List[Any]] = None  # [packed, block_scale, scale_2] stacks, or specs
        self.w2: Optional[List[Any]] = None
        self.w13_done = set()
        self.w2_done = set()

    def _check_expert(self, expert: int, done: set, what: str) -> None:
        if not 0 <= expert < self.num_experts:
            raise ValueError(f"{self.layer_prefix}: expert {expert} out of range [0, {self.num_experts})")
        if expert in done:
            raise ValueError(f"{self.layer_prefix}: expert {expert} {what} arrived twice")
        done.add(expert)

    def _alloc(self, packed: Any, block_scale: Any, scale_2_shape: Tuple[int, ...]) -> List[Any]:
        e = self.num_experts
        if self.spec:
            return [
                TensorSpec(packed.dtype, (e, *packed.shape)),
                TensorSpec(block_scale.dtype, (e, *block_scale.shape)),
                TensorSpec(torch.float32, (e, *scale_2_shape)),
            ]
        device = packed.device
        return [
            torch.empty((e, *packed.shape), dtype=packed.dtype, device=device),
            torch.empty((e, *block_scale.shape), dtype=block_scale.dtype, device=device),
            torch.empty((e, *scale_2_shape), dtype=torch.float32, device=device),
        ]

    def put(self, which: str, expert: int, packed: Any, block_scale: Any, decode: Any) -> None:
        is_w13 = which == "w13"
        self._check_expert(expert, self.w13_done if is_w13 else self.w2_done, which)
        stacks = self.w13 if is_w13 else self.w2
        if stacks is None:
            stacks = self._alloc(packed, block_scale, (2,) if is_w13 else ())
            if is_w13:
                self.w13 = stacks
            else:
                self.w2 = stacks
        if tuple(stacks[0].shape[1:]) != tuple(packed.shape) or tuple(stacks[1].shape[1:]) != tuple(block_scale.shape):
            raise ValueError(
                f"{self.layer_prefix}: expert {expert} {which} shape differs from the layer's other experts"
            )
        if self.spec:
            return
        stacks[0][expert].copy_(packed)
        stacks[1][expert].copy_(block_scale)
        # gate and up share one decode scale (vLLM fuses w13); the per-expert stream sends it once per shard.
        stacks[2][expert].copy_(decode.reshape(()).expand(stacks[2].shape[1:]))

    def complete(self) -> bool:
        return len(self.w13_done) == self.num_experts and len(self.w2_done) == self.num_experts

    def emit(self) -> Iterator[Tuple[str, Any]]:
        assert self.w13 is not None and self.w2 is not None
        e = self.num_experts
        if self.spec:
            ones13: Any = TensorSpec(torch.float32, (e, 2))
            ones2: Any = TensorSpec(torch.float32, (e,))
        else:
            device = self.w13[0].device
            ones13 = torch.ones((e, 2), dtype=torch.float32, device=device)
            ones2 = torch.ones((e,), dtype=torch.float32, device=device)
        tensors = (*self.w13, ones13, *self.w2, ones2)
        for leaf, tensor in zip(STACKED_LEAVES, tensors):
            yield stacked_wire_name(self.layer_prefix, leaf), tensor


class Nvfp4StackedExpertSerializer(Nvfp4ExpertSerializer):
    """:class:`Nvfp4ExpertSerializer` that emits each NVFP4 MoE layer as 8 stacked tensors once it is complete.

    Pairing, pass-through, the BF16 layers and the per-expert quantization are the parent's; only the two emission
    leaves change, from yielding per-expert tensors to writing them into the layer's stacks.
    """

    _spec = False

    def __init__(
        self,
        settings: Nvfp4WeightQuantSettings,
        bf16_layers: Iterable[int] = (),
        *,
        num_experts: int,
    ):
        super().__init__(settings, bf16_layers)
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        self.num_experts = num_experts
        self._layers: Dict[str, _LayerStack] = {}

    def _layer(self, layer_prefix: str) -> _LayerStack:
        stack = self._layers.get(layer_prefix)
        if stack is None:
            stack = self._layers[layer_prefix] = _LayerStack(layer_prefix, self.num_experts, self._spec)
        return stack

    def _flush(self, layer_prefix: str) -> Iterator[Tuple[str, Any]]:
        stack = self._layers[layer_prefix]
        if stack.complete():
            del self._layers[layer_prefix]
            yield from stack.emit()

    def _quantize_single(self, tensor: Any) -> Tuple[Any, Any, Any]:
        return quantize_nvfp4_matrix(tensor, self.settings)

    def _quantize_pair(self, gate: Any, up: Any) -> Tuple[Any, Any, Any]:
        return quantize_nvfp4_matrix(torch.cat((gate, up), dim=0), self.settings)

    def _emit_single(self, prefix: str, proj: str, tensor: Any) -> Iterator[Tuple[str, Any]]:
        layer_prefix, expert = _split_expert_prefix(prefix)
        self._layer(layer_prefix).put("w2", expert, *self._quantize_single(tensor))
        yield from self._flush(layer_prefix)

    def _emit_pair(self, prefix: str, gate: Any, up: Any) -> Iterator[Tuple[str, Any]]:
        layer_prefix, expert = _split_expert_prefix(prefix)
        self._layer(layer_prefix).put("w13", expert, *self._quantize_pair(gate, up))
        yield from self._flush(layer_prefix)

    def finish(self) -> None:
        super().finish()
        if self._layers:
            incomplete = {k: (len(v.w13_done), len(v.w2_done)) for k, v in sorted(self._layers.items())[:4]}
            raise RuntimeError(f"NVFP4 stacked sync: incomplete MoE layers (w13, w2 experts seen): {incomplete}")


class Nvfp4StackedExpertSpecSerializer(Nvfp4StackedExpertSerializer):
    """The same transform over :class:`TensorSpec` inputs: what ``metadata()`` declares, without any data."""

    _spec = True

    def _quantize_single(self, spec: TensorSpec) -> Tuple[TensorSpec, TensorSpec, TensorSpec]:
        return _quantized_specs(*spec.shape)

    def _quantize_pair(self, gate: TensorSpec, up: TensorSpec) -> Tuple[TensorSpec, TensorSpec, TensorSpec]:
        rows, cols = gate.shape
        return _quantized_specs(2 * rows, cols)


def _quantized_specs(rows: int, cols: int) -> Tuple[TensorSpec, TensorSpec, TensorSpec]:
    """Specs of ``quantize_nvfp4_matrix``'s outputs for a ``[rows, cols]`` input."""
    _check_nvfp4_cols(cols)
    return (
        TensorSpec(torch.uint8, (rows, cols // 2)),
        TensorSpec(torch.float8_e4m3fn, (rows, cols // NVFP4_BLOCK_SIZE)),
        TensorSpec(torch.float32, ()),
    )


def max_stacked_wire_bytes(num_experts: int, hidden_size: int, moe_intermediate_size: int) -> int:
    """Bytes of the largest stacked tensor (``w13_weight``), for the packed-buffer floor."""
    w13 = num_experts * 2 * moe_intermediate_size * (hidden_size // 2)
    w2 = num_experts * hidden_size * (moe_intermediate_size // 2)
    return max(w13, w2)


def unstack_to_per_expert(layer_prefix: str, stacks: Dict[str, torch.Tensor]) -> Iterator[Tuple[str, torch.Tensor]]:
    """The per-expert stream (``Nvfp4ExpertSerializer``'s names, shapes, dtypes and bytes) for one stacked layer.

    Per expert: gate, up, down, each as ``weight``, ``weight_scale``, ``weight_scale_2``, ``input_scale``.
    """
    w13, w13_scale = stacks["w13_weight"], stacks["w13_weight_scale"]
    num_experts, two_n = w13.shape[0], w13.shape[1]
    n = two_n // 2
    for e in range(num_experts):
        base = f"{layer_prefix}.{e}"
        for proj, rows, idx in (("gate_proj", slice(0, n), 0), ("up_proj", slice(n, two_n), 1)):
            yield f"{base}.{proj}.weight", w13[e, rows]
            yield f"{base}.{proj}.weight_scale", w13_scale[e, rows]
            yield f"{base}.{proj}.weight_scale_2", stacks["w13_weight_scale_2"][e, idx]
            yield f"{base}.{proj}.input_scale", stacks["w13_input_scale"][e, idx]
        yield f"{base}.down_proj.weight", stacks["w2_weight"][e]
        yield f"{base}.down_proj.weight_scale", stacks["w2_weight_scale"][e]
        yield f"{base}.down_proj.weight_scale_2", stacks["w2_weight_scale_2"][e]
        yield f"{base}.down_proj.input_scale", stacks["w2_input_scale"][e]


def local_to_global_expert_ids(global_to_local: Sequence[int], num_local: int) -> List[int]:
    """Invert an expert map (``global_to_local[g]`` = local slot or -1) into ``local slot -> global expert``.

    Raises unless every local slot is owned by exactly one global expert.
    """
    out: List[Optional[int]] = [None] * num_local
    for g, local in enumerate(global_to_local):
        if local < 0:
            continue
        if local >= num_local or out[local] is not None:
            raise ValueError(f"Expert map is not a bijection onto {num_local} local slots: {list(global_to_local)}")
        out[local] = g
    if any(g is None for g in out):
        raise ValueError(f"Expert map leaves local slots unowned: {list(global_to_local)}")
    return [int(g) for g in out]  # type: ignore[arg-type]
