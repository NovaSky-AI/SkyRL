"""Runtime backport of Megatron-Bridge PR #6089 for the pinned Bridge revision."""

from __future__ import annotations

import inspect

import torch


class _ScaleForward(torch.autograd.Function):
    """Scale a TP-replicated forward contribution without scaling its gradient."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, scale: float) -> torch.Tensor:
        del ctx
        return input_ * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad_output, None


def apply_shared_expert_lora_tp_patch() -> None:
    """Install #6089 before constructing ``ParallelLinearAdapter`` instances."""
    from megatron.bridge.peft import utils

    adapter_cls = utils.ParallelLinearAdapter
    if getattr(adapter_cls, "_skyrl_shared_expert_lora_tp_patch", False):
        return
    # Skip the backport when Bridge includes #6089.
    if "_external_tp_reduce_scale" in inspect.getsource(adapter_cls):
        return

    original_init = adapter_cls.__init__
    original_forward = adapter_cls.forward
    signature = inspect.signature(original_init)
    required = {
        "base_linear_name",
        "input_is_parallel",
        "model_parallel_config",
        "disable_tensor_parallel_comm",
    }
    if not required.issubset(signature.parameters):
        raise RuntimeError(
            "Megatron-Bridge #6089 backport does not recognize ParallelLinearAdapter.__init__; "
            "update the SkyRL patch for this Megatron-Bridge version."
        )

    def patched_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_init(self, *args, **kwargs)
        base_name = bound.arguments["base_linear_name"]
        input_is_parallel = bound.arguments["input_is_parallel"]
        disable_tp = bound.arguments["disable_tensor_parallel_comm"]
        config = bound.arguments["model_parallel_config"] or self.config
        tp_size = utils._process_group_size(
            self.tp_group,
            getattr(config, "tensor_model_parallel_size", None) or 1,
        )
        uses_external_reduce = disable_tp and input_is_parallel and ".shared_experts." in base_name
        self._skyrl_external_tp_reduce_scale = 1.0 / tp_size if uses_external_reduce else 1.0

    def patched_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = original_forward(self, x, *args, **kwargs)
        scale = self._skyrl_external_tp_reduce_scale
        return _ScaleForward.apply(output, scale) if scale != 1.0 else output

    adapter_cls.__init__ = patched_init
    adapter_cls.forward = patched_forward
    adapter_cls._skyrl_shared_expert_lora_tp_patch = True
