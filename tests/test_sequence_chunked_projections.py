import copy

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.sequence_chunked_projections import (
    _wrap_projection_forward,
    apply_sequence_chunked,
)


class _TinySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(8, 12, bias=False)
        self.up = nn.Linear(8, 12, bias=False)
        self.down = nn.Linear(12, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(
            torch.nn.functional.silu(self.gate(hidden_states)) * self.up(hidden_states)
        )


class _TinyProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(13, 8))
        nn.init.kaiming_uniform_(self.weight)
        self.bias = nn.Parameter(torch.empty(13))
        nn.init.uniform_(self.bias)
        self.forward_calls = 0

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        return torch.nn.functional.linear(hidden_states, self.weight), self.bias


def _run_backward(
    module: nn.Module, hidden_states: torch.Tensor, chunk_size: int | None
):
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    if chunk_size is None:
        output = module(hidden_states)
    else:
        output = apply_sequence_chunked(module, hidden_states, chunk_size)
    grad_output = torch.linspace(-0.7, 0.9, output.numel()).reshape_as(output)
    output.backward(grad_output)
    parameter_grads = [
        parameter.grad.detach().clone() for parameter in module.parameters()
    ]
    return output.detach(), hidden_states.grad.detach(), parameter_grads


def test_sequence_chunking_preserves_swiglu_output_and_gradients() -> None:
    torch.manual_seed(7)
    reference = _TinySwiGLU()
    chunked = copy.deepcopy(reference)
    hidden_states = torch.randn(11, 2, 8)

    reference_output, reference_input_grad, reference_parameter_grads = _run_backward(
        reference, hidden_states, None
    )
    chunked_output, chunked_input_grad, chunked_parameter_grads = _run_backward(
        chunked, hidden_states, 3
    )

    torch.testing.assert_close(chunked_output, reference_output)
    torch.testing.assert_close(chunked_input_grad, reference_input_grad)
    for chunked_grad, reference_grad in zip(
        chunked_parameter_grads, reference_parameter_grads, strict=True
    ):
        torch.testing.assert_close(chunked_grad, reference_grad)


def test_projection_wrapper_preserves_output_and_gradients() -> None:
    torch.manual_seed(11)
    reference = _TinyProjection()
    chunked = copy.deepcopy(reference)
    _wrap_projection_forward(chunked, 4)
    reference_input = torch.randn(13, 1, 8, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(13, 1, 13)

    reference_output, reference_bias = reference(reference_input)
    chunked_output, chunked_bias = chunked(chunked_input)
    (reference_output + reference_bias).backward(grad_output)
    (chunked_output + chunked_bias).backward(grad_output)

    torch.testing.assert_close(chunked_output, reference_output)
    torch.testing.assert_close(chunked_bias, reference_bias)
    torch.testing.assert_close(chunked_input.grad, reference_input.grad)
    torch.testing.assert_close(chunked.weight.grad, reference.weight.grad)
    torch.testing.assert_close(chunked.bias.grad, reference.bias.grad)
    assert chunked.forward_calls == 0


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_sequence_chunking_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        from skyrl.backends.skyrl_train.patches.megatron.sequence_chunked_projections import (
            install_sequence_chunked_projections,
        )

        install_sequence_chunked_projections([], chunk_size)
