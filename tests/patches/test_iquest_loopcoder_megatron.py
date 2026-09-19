import pytest
import torch
from torch import nn

pytest.importorskip("megatron.bridge")

from skyrl.backends.skyrl_train.workers.megatron.iquest_loopcoder import (
    LoopCoderCoreAttention,
)


class _ValueAttention(nn.Module):
    def forward(self, query, key, value, attention_mask, **kwargs):
        return value.flatten(-2)


class _ConstantGate(nn.Module):
    def forward(self, query):
        return query.new_full((*query.shape[:-2], query.shape[-2] * query.shape[-1]), 0.25)


def test_loopcoder_core_attention_reuses_first_pass_kv_with_gradients():
    attention = LoopCoderCoreAttention(_ValueAttention(), _ValueAttention(), _ConstantGate())
    query = torch.zeros(2, 1, 2, 3, requires_grad=True)
    first_value = torch.full_like(query, 4.0, requires_grad=True)
    second_value = torch.full_like(query, 8.0, requires_grad=True)

    attention.loop_idx = 0
    attention(query, query, first_value, None)
    attention.loop_idx = 1
    output = attention(query, query, second_value, None)

    assert torch.allclose(output, torch.full_like(output, 7.0))
    output.sum().backward()
    assert torch.allclose(first_value.grad, torch.full_like(first_value, 0.25))
    assert torch.allclose(second_value.grad, torch.full_like(second_value, 0.75))


def test_loopcoder_second_pass_requires_first_pass_kv():
    attention = LoopCoderCoreAttention(_ValueAttention(), _ValueAttention(), _ConstantGate())
    attention.loop_idx = 1
    states = torch.zeros(1, 1, 1, 1)

    with pytest.raises(RuntimeError, match="before first-pass KV"):
        attention(states, states, states, None)
