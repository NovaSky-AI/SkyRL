from collections.abc import Callable
from types import MethodType

import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.patches.megatron.swiglu_triton import (
    install_triton_swiglu,
)


class _SequenceChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, hidden_states: torch.Tensor, chunk_size: int, fn: Callable
    ) -> torch.Tensor:
        ctx.chunk_size = chunk_size
        ctx.fn = fn
        ctx.save_for_backward(hidden_states)

        output = None
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            chunk_output = fn(hidden_states[start:end])
            if output is None:
                output_shape = (*hidden_states.shape[:-1], chunk_output.shape[-1])
                output = hidden_states.new_empty(output_shape)
            output[start:end].copy_(chunk_output)

        assert output is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        (hidden_states,) = ctx.saved_tensors
        hidden_states_grad = (
            torch.empty_like(hidden_states) if hidden_states.requires_grad else None
        )

        for start in range(0, hidden_states.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, hidden_states.shape[0])
            chunk = hidden_states[start:end].detach()
            chunk.requires_grad_(hidden_states.requires_grad)
            with torch.enable_grad():
                chunk_output = ctx.fn(chunk)
            torch.autograd.backward(chunk_output, grad_output[start:end])
            if hidden_states_grad is not None:
                hidden_states_grad[start:end].copy_(chunk.grad)

        return hidden_states_grad, None, None


def apply_sequence_chunked(
    fn: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Run a token-separable projection in bounded sequence chunks."""
    if hidden_states.shape[0] <= chunk_size:
        return fn(hidden_states)
    return _SequenceChunkedFunction.apply(hidden_states, chunk_size, fn)


def _get_tensor_output(
    output: tuple[torch.Tensor, torch.Tensor | None],
) -> torch.Tensor:
    tensor, bias = output
    if bias is not None:
        raise ValueError(
            "Sequence-chunked projections require bias-free Megatron linears"
        )
    return tensor


def _wrap_mlp_forward(module: torch.nn.Module, chunk_size: int) -> None:
    original_forward = module.forward

    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        per_token_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del self
        if (
            per_token_scale is not None
            or kwargs
            or hidden_states.shape[0] <= chunk_size
        ):
            return original_forward(
                hidden_states, per_token_scale=per_token_scale, **kwargs
            )

        def run_chunk(chunk: torch.Tensor) -> torch.Tensor:
            return _get_tensor_output(original_forward(chunk))

        return apply_sequence_chunked(run_chunk, hidden_states, chunk_size), None

    module.forward = MethodType(forward, module)
    module._skyrl_sequence_chunked = True


def _wrap_projection_forward(module: torch.nn.Module, chunk_size: int) -> None:
    original_forward = module.forward

    def forward(self: torch.nn.Module, hidden_states: torch.Tensor, *args, **kwargs):
        del self
        if args or kwargs or hidden_states.shape[0] <= chunk_size:
            return original_forward(hidden_states, *args, **kwargs)

        def run_chunk(chunk: torch.Tensor) -> torch.Tensor:
            return _get_tensor_output(original_forward(chunk))

        return apply_sequence_chunked(run_chunk, hidden_states, chunk_size), None

    module.forward = MethodType(forward, module)
    module._skyrl_sequence_chunked = True


def install_sequence_chunked_projections(
    model_chunks: list[torch.nn.Module], chunk_size: int
) -> tuple[int, int]:
    """Chunk dense SwiGLU blocks and GDN projections along Megatron's sequence axis."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNet2
    from megatron.core.transformer.mlp import MLP

    modules = [
        module for model_chunk in model_chunks for module in model_chunk.modules()
    ]
    swiglu_modules = [
        module
        for module in modules
        if isinstance(module, MLP)
        and module.config.gated_linear_unit
        and module.config.activation_func == F.silu
    ]
    for module in swiglu_modules:
        config = module.config
        if (
            config.activation_func_fp8_input_store
            or (
                config.cpu_offloading
                and getattr(config, "cpu_offloading_activations", False)
            )
            or config.activation_func_clamp_value is not None
            or getattr(config, "activation_func_tanh_clamp_scale", None) is not None
            or getattr(config, "activation_func_tanh_clamp_scale_linear", None)
            is not None
        ):
            raise ValueError(
                "Sequence-chunked Triton SwiGLU does not support activation storage or clamps"
            )
    if swiglu_modules:
        install_triton_swiglu()

    mlp_count = 0
    projection_count = 0
    for module in modules:
        if module in swiglu_modules and not getattr(
            module, "_skyrl_sequence_chunked", False
        ):
            _wrap_mlp_forward(module, chunk_size)
            mlp_count += 1
        if isinstance(module, (GatedDeltaNet, GatedDeltaNet2)):
            for projection in (module.in_proj, module.out_proj):
                if not getattr(projection, "_skyrl_sequence_chunked", False):
                    _wrap_projection_forward(projection, chunk_size)
                    projection_count += 1

    return mlp_count, projection_count
