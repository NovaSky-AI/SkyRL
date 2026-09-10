import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.sequence_chunked_projections import (
    apply_sequence_chunked,
)
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


def _run_triton(input: torch.Tensor, grad_output: torch.Tensor, clamp_args: tuple):
    input = input.detach().clone().requires_grad_(True)
    output = TritonSwiGLUFunction.apply(input, False, False, *clamp_args)
    output.backward(grad_output)
    return output.detach(), input.grad.detach()


class _TinySwiGLUProjection(nn.Module):
    def __init__(self, use_triton: bool) -> None:
        super().__init__()
        self.use_triton = use_triton
        self.up = nn.Linear(32, 128, bias=False)
        self.down = nn.Linear(64, 32, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.up(hidden_states)
        if self.use_triton:
            activated = TritonSwiGLUFunction.apply(projected, False, False)
        else:
            gate, linear = projected.chunk(2, dim=-1)
            activated = F.silu(gate) * linear
        return self.down(activated)


def _get_grad_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(tensor.float()) for tensor in tensors])
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "clamp_args", [(), (None, None, None)], ids=["legacy", "current"]
)
def test_triton_swiglu_matches_torch_output_and_input_gradient(
    dtype: torch.dtype, clamp_args: tuple
) -> None:
    torch.manual_seed(17)
    input = torch.randn(7, 3, 260, device="cuda", dtype=dtype)
    grad_output = torch.randn(7, 3, 130, device="cuda", dtype=dtype)

    reference_output, reference_grad = _run_reference(input, grad_output)
    triton_output, triton_grad = _run_triton(input, grad_output, clamp_args)

    tolerances = (
        {"atol": 2e-2, "rtol": 2e-2}
        if dtype == torch.bfloat16
        else {"atol": 2e-5, "rtol": 2e-5}
    )
    torch.testing.assert_close(triton_output, reference_output, **tolerances)
    torch.testing.assert_close(triton_grad, reference_grad, **tolerances)


def test_triton_swiglu_handles_production_chunk_shape() -> None:
    input = torch.randn(
        4096, 1, 34816, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    output = TritonSwiGLUFunction.apply(input, False, False)
    torch.cuda.synchronize()
    output.sum().backward()
    torch.cuda.synchronize()

    assert output.shape == (4096, 1, 17408)
    assert input.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(input.grad).all()


def test_triton_swiglu_large_sequence_has_finite_grpo_scaled_gradients() -> None:
    torch.manual_seed(31)
    input = torch.randn(65537, 1, 512, device="cuda", dtype=torch.bfloat16)
    grad_output = (
        torch.randn(65537, 1, 256, device="cuda", dtype=torch.bfloat16) / input.shape[0]
    )

    reference_output, reference_grad = _run_reference(input, grad_output)
    triton_output, triton_grad = _run_triton(input, grad_output, ())

    assert torch.isfinite(triton_output).all()
    assert torch.isfinite(triton_grad).all()
    assert torch.isfinite(_get_grad_norm([triton_grad]))
    torch.testing.assert_close(triton_output, reference_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(triton_grad, reference_grad, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("chunk_size", [4096, 65536])
def test_chunked_triton_swiglu_parameter_gradients_are_finite_and_match_reference(
    chunk_size: int,
) -> None:
    torch.manual_seed(37)
    reference = _TinySwiGLUProjection(use_triton=False).cuda().to(torch.bfloat16)
    chunked = copy.deepcopy(reference)
    chunked.use_triton = True
    hidden_states = torch.randn(65537, 1, 32, device="cuda", dtype=torch.bfloat16)
    reference_input = hidden_states.detach().clone().requires_grad_(True)
    chunked_input = hidden_states.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(reference_input) / hidden_states.shape[0]

    reference_output = reference(reference_input)
    chunked_output = apply_sequence_chunked(
        chunked,
        chunked_input,
        chunk_size,
        parameters=tuple(chunked.parameters()),
    )
    reference_output.backward(grad_output)
    chunked_output.backward(grad_output)

    reference_grads = [
        reference_input.grad,
        *(parameter.grad for parameter in reference.parameters()),
    ]
    chunked_grads = [
        chunked_input.grad,
        *(parameter.grad for parameter in chunked.parameters()),
    ]
    assert all(grad is not None for grad in chunked_grads)
    assert all(torch.isfinite(grad).all() for grad in chunked_grads)
    assert torch.isfinite(_get_grad_norm(chunked_grads))
    torch.testing.assert_close(chunked_output, reference_output, atol=3e-2, rtol=3e-2)
    for chunked_grad, reference_grad in zip(
        chunked_grads, reference_grads, strict=True
    ):
        torch.testing.assert_close(chunked_grad, reference_grad, atol=3e-2, rtol=3e-2)
