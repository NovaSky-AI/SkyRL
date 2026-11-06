"""DeepSeek-V4.1 two-level indexer: candidate blocks as ids (no dense ``[sq, n_c]`` mask).

- ``select_candidate_block_ids`` + ``candidate_blocks_to_mask`` reproduce the reference position mask of
  ``select_candidate_blocks`` exactly (CPU).
- Reindex top-k through NVIDIA/Megatron-LM#7285's candidate kernel (``candidate_scores`` +
  ``candidate_topk_ids``) selects the same compressed positions as dense scoring masked to those blocks (GPU).
"""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41 import attention
from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.candidate_kernels import (
    candidate_kernel_supported,
    candidate_scores,
    candidate_topk_ids,
)

pytestmark = pytest.mark.megatron


@pytest.mark.parametrize("width,topk_blocks,block_size", [(1000, 16, 8), (4096, 64, 8), (37, 100, 4)])
def test_block_ids_reproduce_the_position_mask(width, topk_blocks, block_size):
    torch.manual_seed(0)
    rows = 64
    logits = torch.randn(rows, 1, width)
    compress_lens = torch.randint(1, width + 1, (rows, 1, 1))
    logits = logits.masked_fill(torch.arange(width) >= compress_lens, -torch.inf)
    expected = attention.select_candidate_blocks(logits, compress_lens, topk_blocks, block_size)
    ids = attention.select_candidate_block_ids(logits, compress_lens, topk_blocks, block_size)
    assert torch.equal(attention.candidate_blocks_to_mask(ids, block_size, width), expected)


def test_candidate_kernel_topk_matches_dense_masked_topk():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    rows, heads, dim, n_keys, topk_blocks, block_size, topk = 512, 32, 128, 8192, 256, 8, 128
    assert candidate_kernel_supported(heads, dim, block_size, topk_blocks)
    first = 6000  # rows are positions first .. first + rows - 1 (ratio 1)
    q = torch.randn(rows, heads, dim, device="cuda").to(torch.bfloat16)
    keys = torch.randn(n_keys, dim, device="cuda").to(torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda").to(torch.bfloat16)
    scale = (dim * heads) ** -0.5
    visible = torch.arange(first + 1, first + rows + 1, device="cuda")

    dense = torch.einsum("rhd,td->rht", q.float(), keys.float()).relu()
    dense = (dense * (weights.float() * scale).unsqueeze(-1)).sum(dim=1)  # [rows, n_keys]
    dense = dense.masked_fill(torch.arange(n_keys, device="cuda") >= visible.view(-1, 1), -torch.inf)
    blocks = attention.select_candidate_block_ids(
        dense + torch.randn_like(dense), visible.view(-1, 1), topk_blocks, block_size
    )  # some other layer's candidates
    masked = dense.masked_fill(~attention.candidate_blocks_to_mask(blocks, block_size, n_keys), -torch.inf)
    vals, idx = masked.topk(topk, dim=-1)
    want = torch.where(torch.isfinite(vals), idx, -1)

    got = candidate_topk_ids(
        candidate_scores(q, keys, weights, scale, blocks, block_size, visible), blocks, block_size, topk
    )
    for row in range(rows):
        assert set(got[row].tolist()) - {-1} == set(want[row].tolist()) - {-1}, row
