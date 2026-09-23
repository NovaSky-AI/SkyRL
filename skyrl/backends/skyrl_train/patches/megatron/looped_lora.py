from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from skyrl.train.looped_lora import LayerExecution, build_looped_lora_schedule

_adapter_only: ContextVar[bool] = ContextVar("looped_lora_adapter_only", default=False)
_loop_schedule_active: ContextVar[bool] = ContextVar("looped_lora_schedule_active", default=False)


@contextmanager
def _set_context(variable: ContextVar[bool], value: bool) -> Iterator[None]:
    token = variable.set(value)
    try:
        yield
    finally:
        variable.reset(token)


def _normalize_adapter_input(linear: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    base = linear.to_wrap
    weight = getattr(base, "layer_norm_weight", None)
    if weight is None:
        return inputs

    normalization = getattr(base, "normalization", None)
    eps = base.eps
    if getattr(base, "zero_centered_gamma", False):
        weight = weight + 1
    if normalization == "RMSNorm":
        return F.rms_norm(inputs, (inputs.shape[-1],), weight, eps)
    if normalization == "LayerNorm":
        return F.layer_norm(
            inputs,
            (inputs.shape[-1],),
            weight,
            getattr(base, "layer_norm_bias", None),
            eps,
        )
    raise ValueError(f"Unsupported looped LoRA normalization: {normalization!r}")


def _looped_linear_forward(linear: nn.Module, inputs: torch.Tensor, *args: Any, **kwargs: Any):
    if not _adapter_only.get():
        return linear._looped_lora_original_forward(inputs, *args, **kwargs)
    if not linear._adapter_enabled:
        raise RuntimeError("Looped LoRA extra passes require enabled LoRA adapters")

    adapter_inputs = _normalize_adapter_input(linear, inputs).contiguous()
    output = linear.adapter_forward(linear.adapter, adapter_inputs, *args, **kwargs)
    if linear._base_returns_tuple:
        return output, None
    return output


class _LoopedModuleList(nn.ModuleList):
    def __init__(self, layers: Sequence[nn.Module], schedule: Sequence[LayerExecution]) -> None:
        super().__init__(layers)
        self._executions = tuple(
            (
                super(_LoopedModuleList, self).__getitem__(execution.physical_layer),
                execution.lora_only,
            )
            for execution in schedule
        )

    def __iter__(self):
        if _loop_schedule_active.get():
            return self._iter_executions()
        return super().__iter__()

    def _iter_executions(self):
        for layer, lora_only in self._executions:
            with _set_context(_adapter_only, lora_only):
                yield layer

    def __len__(self) -> int:
        if _loop_schedule_active.get():
            return len(self._executions)
        return super().__len__()

    def __getitem__(self, index):
        if _loop_schedule_active.get():
            if isinstance(index, slice):
                return tuple(layer for layer, _ in self._executions[index])
            layer, lora_only = self._executions[index]
            _adapter_only.set(lora_only)
            return layer
        return super().__getitem__(index)


def _looped_block_forward(block: nn.Module, *args: Any, **kwargs: Any):
    with _set_context(_loop_schedule_active, True), _set_context(_adapter_only, False):
        physical_layer_count = block.num_layers_per_pipeline_rank
        block.num_layers_per_pipeline_rank = len(block.layers)
        try:
            return block._looped_lora_original_forward(*args, **kwargs)
        finally:
            block.num_layers_per_pipeline_rank = physical_layer_count


def install_looped_lora(model: nn.Module | Sequence[nn.Module], sections: Sequence[dict[str, int]], mode: str) -> None:
    """Install the Megatron forward schedule matching SkyRL's vLLM loop semantics."""
    if mode not in {"lora_only", "full_block"}:
        raise ValueError(f"Unsupported looped LoRA mode: {mode!r}")

    from megatron.bridge.peft.lora_layers import LoRALinear, TEFusedLoRALinear

    roots = (model,) if isinstance(model, nn.Module) else tuple(model)
    modules = tuple(module for root in roots for module in root.modules())
    blocks = [module for module in modules if hasattr(module, "layers") and module.layers]
    blocks = [block for block in blocks if all(hasattr(layer, "layer_number") for layer in block.layers)]
    if len(blocks) != 1:
        raise ValueError("Looped LoRA training currently requires one non-empty Megatron transformer block (PP=1)")

    block = blocks[0]
    physical_layers = list(block.layers)
    num_hidden_layers = block.config.num_layers
    if len(physical_layers) != num_hidden_layers:
        raise ValueError("Looped LoRA training currently requires every transformer layer on one pipeline stage (PP=1)")
    if any(layer.layer_number != index + 1 for index, layer in enumerate(physical_layers)):
        raise ValueError("Looped LoRA requires contiguous, one-based Megatron layer numbers")
    if getattr(block.config, "enable_mhc_connections", False):
        raise ValueError("Looped LoRA does not support Megatron hyper-connections")

    schedule = build_looped_lora_schedule(num_hidden_layers, sections)
    if mode == "full_block":
        schedule = tuple(LayerExecution(execution.physical_layer, False) for execution in schedule)

    for module in modules:
        if isinstance(module, TEFusedLoRALinear):
            raise TypeError("Looped LoRA does not support fused Transformer Engine LoRA")
        if isinstance(module, LoRALinear):
            module._looped_lora_original_forward = module.forward
            module.forward = MethodType(_looped_linear_forward, module)

    block.layers = _LoopedModuleList(physical_layers, schedule)
    block._looped_lora_original_forward = block.forward
    block.forward = MethodType(_looped_block_forward, block)
