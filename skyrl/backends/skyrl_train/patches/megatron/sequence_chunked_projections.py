from collections.abc import Callable
from types import MethodType

import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.patches.megatron.swiglu_triton import (
    install_triton_swiglu,
)


class _SequenceChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        tensor_inputs = args[:-3]
        aligned_input_count, chunk_size, fn = args[-3:]
        aligned_inputs = tensor_inputs[:aligned_input_count]
        ctx.chunk_size = chunk_size
        ctx.fn = fn
        ctx.aligned_input_count = aligned_input_count
        ctx.save_for_backward(hidden_states, *tensor_inputs)

        output = None
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            chunk_output = fn(
                hidden_states[start:end],
                *(aligned_input[start:end] for aligned_input in aligned_inputs),
            )
            if output is None:
                output_shape = (*hidden_states.shape[:-1], chunk_output.shape[-1])
                output = hidden_states.new_empty(output_shape)
            output[start:end].copy_(chunk_output)

        assert output is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        saved_inputs = ctx.saved_tensors
        data_input_count = 1 + ctx.aligned_input_count
        data_inputs = saved_inputs[:data_input_count]
        parameters = saved_inputs[data_input_count:]
        input_grads = [
            torch.empty_like(input) if input.requires_grad else None
            for input in data_inputs
        ]
        parameter_grads = [None] * len(parameters)

        for start in range(0, data_inputs[0].shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, data_inputs[0].shape[0])
            chunks = [input[start:end].detach() for input in data_inputs]
            for chunk, input in zip(chunks, data_inputs, strict=True):
                chunk.requires_grad_(input.requires_grad)
            with torch.enable_grad():
                chunk_output = ctx.fn(*chunks)
            grad_targets = [
                tensor
                for tensor in (*chunks, *parameters)
                if tensor.requires_grad
            ]
            chunk_grads = torch.autograd.grad(
                chunk_output,
                grad_targets,
                grad_output[start:end],
                allow_unused=True,
            )
            grad_iterator = iter(chunk_grads)
            for input_grad, chunk in zip(input_grads, chunks, strict=True):
                if input_grad is not None:
                    chunk_grad = next(grad_iterator)
                    if chunk_grad is not None:
                        input_grad[start:end].copy_(chunk_grad)
            for index, parameter in enumerate(parameters):
                if not parameter.requires_grad:
                    continue
                chunk_grad = next(grad_iterator)
                if chunk_grad is not None:
                    if parameter_grads[index] is None:
                        parameter_grads[index] = chunk_grad
                    else:
                        parameter_grads[index].add_(chunk_grad)

        return (*input_grads, *parameter_grads, None, None, None)


def apply_sequence_chunked(
    fn: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    chunk_size: int,
    *aligned_inputs: torch.Tensor,
    parameters: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run a token-separable projection in bounded sequence chunks."""
    if hidden_states.shape[0] <= chunk_size:
        return fn(hidden_states, *aligned_inputs)
    if any(input.shape[0] != hidden_states.shape[0] for input in aligned_inputs):
        raise ValueError("Aligned sequence inputs must have the same leading dimension")
    return _SequenceChunkedFunction.apply(
        hidden_states,
        *aligned_inputs,
        *parameters,
        len(aligned_inputs),
        chunk_size,
        fn,
    )


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
    trainable_parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )

    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        per_token_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del self
        if hidden_states.shape[0] <= chunk_size:
            return original_forward(
                hidden_states, per_token_scale=per_token_scale, **kwargs
            )

        if per_token_scale is None:

            def run_chunk(chunk: torch.Tensor) -> torch.Tensor:
                return _get_tensor_output(original_forward(chunk, **kwargs))

            output = apply_sequence_chunked(
                run_chunk,
                hidden_states,
                chunk_size,
                parameters=trainable_parameters,
            )
        else:

            def run_scaled_chunk(
                chunk: torch.Tensor, scale_chunk: torch.Tensor
            ) -> torch.Tensor:
                return _get_tensor_output(
                    original_forward(
                        chunk, per_token_scale=scale_chunk, **kwargs
                    )
                )

            output = apply_sequence_chunked(
                run_scaled_chunk,
                hidden_states,
                chunk_size,
                per_token_scale,
                parameters=trainable_parameters,
            )
        return output, None

    module.forward = MethodType(forward, module)
    module._skyrl_sequence_chunked = True


def install_sequence_chunked_projections(
    model_chunks: list[torch.nn.Module], chunk_size: int
) -> tuple[int, int]:
    """Chunk dense SwiGLU and recurrent GDN blocks along the sequence axis."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNet2
    from megatron.core.transformer.mlp import MLP

    from skyrl.backends.skyrl_train.patches.megatron.gdn_sequence_chunking import (
        wrap_gdn_forward,
    )

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
    gdn_count = 0
    for module in modules:
        if module in swiglu_modules and not getattr(
            module, "_skyrl_sequence_chunked", False
        ):
            _wrap_mlp_forward(module, chunk_size)
            mlp_count += 1
        if isinstance(module, (GatedDeltaNet, GatedDeltaNet2)) and not getattr(
            module, "_skyrl_sequence_chunked", False
        ):
            if module.tp_size != 1 or module.cp_size != 1 or module.sp_size != 1:
                raise ValueError(
                    "Sequence-chunked GDN currently requires TP1, CP1, and sequence_parallel=False"
                )
            wrap_gdn_forward(module, chunk_size)
            gdn_count += 1

    return mlp_count, gdn_count
