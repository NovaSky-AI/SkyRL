import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _swiglu_forward_kernel(
        input_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        output_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < output_elements
        rows = offsets // hidden_size
        columns = offsets % hidden_size
        row_offsets = rows * (2 * hidden_size)
        gate = tl.load(input_ptr + row_offsets + columns, mask=mask).to(tl.float32)
        linear = tl.load(input_ptr + row_offsets + hidden_size + columns, mask=mask).to(
            tl.float32
        )
        output = gate * tl.sigmoid(gate) * linear
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        input_ptr,
        grad_output_ptr,
        grad_input_ptr,
        hidden_size: tl.constexpr,
        output_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < output_elements
        rows = offsets // hidden_size
        columns = offsets % hidden_size
        row_offsets = rows * (2 * hidden_size)
        gate = tl.load(input_ptr + row_offsets + columns, mask=mask).to(tl.float32)
        linear = tl.load(input_ptr + row_offsets + hidden_size + columns, mask=mask).to(
            tl.float32
        )
        grad_output = tl.load(grad_output_ptr + offsets, mask=mask).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        silu = gate * sigmoid
        grad_gate = grad_output * linear * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_linear = grad_output * silu
        tl.store(grad_input_ptr + row_offsets + columns, grad_gate, mask=mask)
        tl.store(
            grad_input_ptr + row_offsets + hidden_size + columns,
            grad_linear,
            mask=mask,
        )


class TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        fp8_input_store: bool,
        cpu_offload_input: bool,
        clamp_value: float | None,
        gate_clamp_scale: float | None,
        linear_clamp_scale: float | None,
    ) -> torch.Tensor:
        if triton is None or not input.is_cuda:
            raise RuntimeError(
                "The Triton SwiGLU path requires Triton and a CUDA tensor"
            )
        if fp8_input_store or cpu_offload_input:
            raise ValueError(
                "The Triton SwiGLU path does not support activation storage transforms"
            )
        if (
            clamp_value is not None
            or gate_clamp_scale is not None
            or linear_clamp_scale is not None
        ):
            raise ValueError(
                "The Triton SwiGLU path does not support clamped activations"
            )
        if input.shape[-1] % 2:
            raise ValueError(f"SwiGLU input width must be even, got {input.shape[-1]}")

        input = input.contiguous()
        hidden_size = input.shape[-1] // 2
        output = input.new_empty((*input.shape[:-1], hidden_size))
        output_elements = output.numel()
        _swiglu_forward_kernel[(triton.cdiv(output_elements, 256),)](
            input,
            output,
            hidden_size,
            output_elements,
            BLOCK_SIZE=256,
        )
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        (input,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        hidden_size = input.shape[-1] // 2
        output_elements = grad_output.numel()
        grad_input = torch.empty_like(input)
        _swiglu_backward_kernel[(triton.cdiv(output_elements, 256),)](
            input,
            grad_output,
            grad_input,
            hidden_size,
            output_elements,
            BLOCK_SIZE=256,
        )
        return grad_input, None, None, None, None, None


def install_triton_swiglu() -> None:
    """Replace Megatron's bias-free SwiGLU autograd function with the Triton kernel."""
    if triton is None:
        raise RuntimeError("Sequence-chunked SwiGLU requires Triton")

    from megatron.core.fusions import fused_bias_swiglu

    fused_bias_swiglu.SwiGLUFunction = TritonSwiGLUFunction
