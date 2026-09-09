import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.gdn_sequence_chunking import (
    apply_stateful_sequence_chunked,
)

fla = pytest.importorskip("fla")
from fla.modules.convolution import causal_conv1d  # noqa: E402
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402

pytestmark = [
    pytest.mark.megatron,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA requires CUDA"),
]


class _TinyFlaGDN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.head_dim = 16
        self.qkv_dim = 3 * self.heads * self.head_dim
        self.in_proj = nn.Linear(32, self.qkv_dim + 2 * self.heads, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(self.qkv_dim, 4) * 0.02)
        self.out_proj = nn.Linear(self.heads * self.head_dim, 32, bias=False)

    def forward_chunk(
        self, hidden_states: torch.Tensor, *states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_state, recurrent_state = states if states else (None, None)
        seq_len, batch, _ = hidden_states.shape
        projected = self.in_proj(hidden_states).transpose(0, 1)
        qkv, g, beta = torch.split(
            projected, [self.qkv_dim, self.heads, self.heads], dim=-1
        )
        qkv, conv_state = causal_conv1d(
            qkv,
            self.conv_weight,
            activation="silu",
            initial_state=conv_state,
            output_final_state=True,
        )
        query, key, value = (
            part.reshape(batch, seq_len, self.heads, self.head_dim)
            for part in qkv.chunk(3, dim=-1)
        )
        output, recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            F.logsigmoid(g),
            beta.sigmoid(),
            initial_state=recurrent_state,
            output_final_state=True,
        )
        output = output.reshape(batch, seq_len, -1).transpose(0, 1)
        return self.out_proj(output), conv_state, recurrent_state

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_chunk(hidden_states)[0]


def test_chunked_fla_gdn_preserves_output_and_gradients() -> None:
    torch.manual_seed(29)
    reference = _TinyFlaGDN().cuda().to(torch.bfloat16)
    chunked = copy.deepcopy(reference)
    reference_input = torch.randn(
        128, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(reference_input)

    reference_output = reference(reference_input)
    chunked_output = apply_stateful_sequence_chunked(
        chunked.forward_chunk, chunked_input, 64
    )
    reference_output.backward(grad_output)
    chunked_output.backward(grad_output)

    torch.testing.assert_close(chunked_output, reference_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        chunked_input.grad, reference_input.grad, atol=3e-2, rtol=3e-2
    )
    for chunked_parameter, reference_parameter in zip(
        chunked.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            chunked_parameter.grad,
            reference_parameter.grad,
            atol=3e-2,
            rtol=3e-2,
        )
