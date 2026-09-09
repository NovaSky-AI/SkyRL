import pytest
import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.patches.megatron.swiglu_triton import (
    TritonSwiGLUFunction,
    triton,
)

pytestmark = [
    pytest.mark.megatron,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and triton is not None),
        reason="Triton SwiGLU requires CUDA and Triton",
    ),
]


def _run_reference(input: torch.Tensor, grad_output: torch.Tensor):
    input = input.detach().clone().requires_grad_(True)
    gate, linear = input.chunk(2, dim=-1)
    output = F.silu(gate) * linear
    output.backward(grad_output)
    return output.detach(), input.grad.detach()


def _run_triton(input: torch.Tensor, grad_output: torch.Tensor):
    input = input.detach().clone().requires_grad_(True)
    output = TritonSwiGLUFunction.apply(input, False, False, None, None, None)
    output.backward(grad_output)
    return output.detach(), input.grad.detach()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_swiglu_matches_torch_output_and_input_gradient(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(17)
    input = torch.randn(7, 3, 260, device="cuda", dtype=dtype)
    grad_output = torch.randn(7, 3, 130, device="cuda", dtype=dtype)

    reference_output, reference_grad = _run_reference(input, grad_output)
    triton_output, triton_grad = _run_triton(input, grad_output)

    tolerances = (
        {"atol": 2e-2, "rtol": 2e-2}
        if dtype == torch.bfloat16
        else {"atol": 2e-5, "rtol": 2e-5}
    )
    torch.testing.assert_close(triton_output, reference_output, **tolerances)
    torch.testing.assert_close(triton_grad, reference_grad, **tolerances)
