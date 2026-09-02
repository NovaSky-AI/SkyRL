from importlib.metadata import version

import pytest
import torch

pytestmark = pytest.mark.megatron


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
    output, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=256,
        max_seqlen_k=256,
        causal=True,
    )
    output.float().square().mean().backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
