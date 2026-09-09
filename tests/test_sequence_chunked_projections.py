import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from skyrl.backends.skyrl_train.patches.megatron.gdn_sequence_chunking import (
    _get_packed_sequence_ranges,
    apply_stateful_sequence_chunked,
    wrap_gdn_forward,
)
from skyrl.backends.skyrl_train.patches.megatron.sequence_chunked_projections import (
    _wrap_mlp_forward,
    apply_sequence_chunked,
)


class _TinyRecurrentProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 12, bias=False)
        self.out_proj = nn.Linear(4, 8, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.8))

    def forward_chunk(
        self, hidden_states: torch.Tensor, *states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.in_proj(hidden_states)
        value, gate, update = projected.chunk(3, dim=-1)
        state = states[0] if states else torch.zeros_like(value[0])
        outputs = []
        for value_t, gate_t, update_t in zip(value, gate, update, strict=True):
            state = self.decay * state + torch.tanh(update_t) * value_t
            outputs.append(torch.sigmoid(gate_t) * state)
        return self.out_proj(torch.stack(outputs)), state

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_chunk(hidden_states)[0]


class _TinyGDNWrapperTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original_forward_calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        self.original_forward_calls += 1
        return hidden_states, None

    def _resolve_cu_seqlens(
        self, cu_seqlens_padded, cu_seqlens, total_seq_len, name
    ) -> torch.Tensor:
        del total_seq_len, name
        return cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens


class _TinyPackedSequenceParams:
    qkv_format = "thd"
    cu_seqlens_q_padded = None
    cu_seqlens_kv_padded = None
    cu_seqlens_q = torch.tensor([0, 4, 9])
    cu_seqlens_kv = torch.tensor([0, 4, 9])


class _TinySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(8, 12, bias=False)
        self.up = nn.Linear(8, 12, bias=False)
        self.down = nn.Linear(12, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(
            torch.nn.functional.silu(self.gate(hidden_states))
            * self.up(hidden_states)
        )


class _TinyScaledMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False)
        self.forward_sizes = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_token_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        self.forward_sizes.append(hidden_states.shape[0])
        output = self.projection(hidden_states)
        if per_token_scale is not None:
            output = output * per_token_scale.unsqueeze(-1)
        return output, None


def _run_backward(
    module: nn.Module, hidden_states: torch.Tensor, chunk_size: int | None
):
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    if chunk_size is None:
        output = module(hidden_states)
    else:
        output = apply_sequence_chunked(
            module,
            hidden_states,
            chunk_size,
            parameters=tuple(module.parameters()),
        )
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


def test_stateful_chunking_preserves_gdn_projection_output_and_gradients() -> None:
    torch.manual_seed(17)
    reference = _TinyRecurrentProjection()
    chunked = copy.deepcopy(reference)
    reference_input = torch.randn(11, 2, 8, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(11, 2, 8)

    reference_output = reference(reference_input)
    chunked_output = apply_stateful_sequence_chunked(
        chunked.forward_chunk, chunked_input, 3
    )
    reference_output.backward(grad_output)
    chunked_output.backward(grad_output)

    torch.testing.assert_close(chunked_output, reference_output)
    torch.testing.assert_close(chunked_input.grad, reference_input.grad)
    for chunked_parameter, reference_parameter in zip(
        chunked.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(chunked_parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize("sequence_len_offset", [0, 262144])
def test_gdn_wrapper_chunks_training_sequence_offset(
    monkeypatch, sequence_len_offset: int
) -> None:
    module = _TinyGDNWrapperTarget()
    chunk_calls = 0

    def run_chunk(module, hidden_states, *states):
        del module, states
        nonlocal chunk_calls
        chunk_calls += 1
        return hidden_states, hidden_states[-1]

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.patches.megatron.gdn_sequence_chunking._run_gdn_chunk",
        run_chunk,
    )
    wrap_gdn_forward(module, 4)

    output, bias = module(
        torch.randn(9, 1, 8),
        None,
        sequence_len_offset=sequence_len_offset,
        ignored_training_kwarg=True,
    )

    assert output.shape == (9, 1, 8)
    assert bias is None
    assert chunk_calls == 3
    assert module.original_forward_calls == 0


def test_packed_sequence_ranges_pair_adjacent_boundaries() -> None:
    ranges = _get_packed_sequence_ranges(
        _TinyGDNWrapperTarget(), torch.randn(9, 1, 8), _TinyPackedSequenceParams()
    )

    assert ranges == [(0, 4), (4, 9)]


def test_mlp_wrapper_chunks_per_token_scale_and_preserves_gradients() -> None:
    torch.manual_seed(23)
    reference = _TinyScaledMLP()
    chunked = copy.deepcopy(reference)
    _wrap_mlp_forward(chunked, 4)
    reference_input = torch.randn(9, 2, 8, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    reference_scale = torch.randn(9, 2, requires_grad=True)
    chunked_scale = reference_scale.detach().clone().requires_grad_(True)
    grad_output = torch.randn(9, 2, 8)

    reference_output, _ = reference(
        reference_input, per_token_scale=reference_scale, ignored_kwarg=True
    )
    chunked_output, _ = chunked(
        chunked_input, per_token_scale=chunked_scale, ignored_kwarg=True
    )
    reference_output.backward(grad_output)
    chunked_output.backward(grad_output)

    torch.testing.assert_close(chunked_output, reference_output)
    torch.testing.assert_close(chunked_input.grad, reference_input.grad)
    torch.testing.assert_close(chunked_scale.grad, reference_scale.grad)
    torch.testing.assert_close(
        chunked.projection.weight.grad, reference.projection.weight.grad
    )
    assert max(chunked.forward_sizes) == 4


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_sequence_chunking_composes_with_outer_checkpoint(
    use_reentrant: bool,
) -> None:
    torch.manual_seed(29)
    recurrent = _TinyRecurrentProjection()
    mlp = _TinyScaledMLP()
    _wrap_mlp_forward(mlp, 4)
    hidden_states = torch.randn(9, 2, 8, requires_grad=True)

    def run_block(chunk_input: torch.Tensor) -> torch.Tensor:
        recurrent_output = apply_stateful_sequence_chunked(
            recurrent.forward_chunk, chunk_input, 4
        )
        return mlp(recurrent_output)[0]

    output = checkpoint(run_block, hidden_states, use_reentrant=use_reentrant)
    output.square().mean().backward()

    assert hidden_states.grad is not None
    assert all(parameter.grad is not None for parameter in recurrent.parameters())
    assert all(parameter.grad is not None for parameter in mlp.parameters())


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_sequence_chunking_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        from skyrl.backends.skyrl_train.patches.megatron.sequence_chunked_projections import (
            install_sequence_chunked_projections,
        )

        install_sequence_chunked_projections([], chunk_size)
