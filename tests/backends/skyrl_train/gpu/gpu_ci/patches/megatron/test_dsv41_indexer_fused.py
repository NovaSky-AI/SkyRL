"""DeepSeek-V4.1 indexer: fused (cuDNN DSA dense indexer kernel) vs reference scoring.

``fused_indexer_scores_rows`` (``SKYRL_DSV41_INDEXER_IMPL=fused``, backport of NVIDIA/Megatron-LM#7285) must
give the reference ``sum_h relu(q_h . k) * w_h * scale`` on every causally visible key -- for chunks that
start mid-sequence (``q_causal_offsets``) and for compress ratios 1 and 2 -- and the same top-k selection.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41 import attention

pytestmark = pytest.mark.megatron


@pytest.mark.parametrize(
    "ratio,seq_len,first_position,rows",
    [(1, 4096, 0, 1024), (1, 4096, 1500, 700), (2, 8192, 3000, 1000), (2, 8192, 0, 4096)],
)
def test_fused_indexer_scores_match_reference(ratio, seq_len, first_position, rows):
    if not torch.cuda.is_available() or attention._fused_indexer_score_fn() is None:
        pytest.skip("needs CUDA and cudnn-frontend's DSA namespace")
    torch.manual_seed(0)
    heads, dim, n_keys = 32, 128, seq_len // ratio
    q = torch.randn(rows, heads, dim, device="cuda").to(torch.bfloat16)
    keys = torch.randn(n_keys, dim, device="cuda").to(torch.bfloat16)
    head_weights = torch.randn(rows, heads, device="cuda").to(torch.bfloat16)
    scale = (dim * heads) ** -0.5

    fused = attention.fused_indexer_scores_rows(q, keys, head_weights, scale, ratio, first_position)
    reference = torch.einsum("rhd,td->rht", q.float(), keys.float()).relu()
    reference = (reference * (head_weights.float() * scale).unsqueeze(-1)).sum(dim=1)

    limits = (torch.arange(first_position + 1, first_position + rows + 1, device="cuda") // ratio).view(-1, 1)
    visible = torch.arange(n_keys, device="cuda").view(1, -1) < limits
    zero = torch.zeros_like(reference)
    rel_err = (
        torch.where(visible, (fused - reference).abs(), zero).max() / torch.where(visible, reference.abs(), zero).max()
    )
    assert rel_err < 1e-4, f"max rel err {rel_err:.2e}"

    topk = min(512, n_keys)
    fused_idx = fused.masked_fill(~visible, -torch.inf).topk(topk, dim=-1)
    reference_idx = reference.masked_fill(~visible, -torch.inf).topk(topk, dim=-1).indices
    for row in range(0, rows, max(1, rows // 64)):
        n_valid = int(torch.isfinite(fused_idx.values[row]).sum())
        assert set(fused_idx.indices[row, :n_valid].tolist()) == set(reference_idx[row, :n_valid].tolist())
