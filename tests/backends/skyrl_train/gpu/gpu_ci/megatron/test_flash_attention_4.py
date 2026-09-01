from importlib.metadata import version

import pytest
import torch

pytestmark = [pytest.mark.megatron, pytest.mark.h100]


@pytest.mark.parametrize(("query_heads", "kv_heads", "head_dim"), [(24, 4, 256), (32, 2, 128)])
def test_transformer_engine_detects_working_flash_attention_4(query_heads, kv_heads, head_dim):
    from flash_attn.cute.interface import flash_attn_varlen_func
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    assert version("flash-attn-4") == "4.0.0b28"
    assert backends.fa_utils.v4_is_installed

    q = torch.randn(
        384,
        query_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k, v = [
        torch.randn(
            384,
            kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(2)
    ]
    cu_seqlens = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int32)
    output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 256, 256, causal=True)
    output.float().square().mean().backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
