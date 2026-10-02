"""Real Transformer Engine coverage for the mixed-dtype norm compatibility patch."""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_grad_norm_mixed_dtype import (
    make_dtype_grouped_applier,
)


@pytest.mark.megatron
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine")
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_dtype_norm_with_transformer_engine(reverse):
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
    )

    grads = [
        torch.zeros(4096, device="cuda", dtype=torch.bfloat16),
        torch.tensor([1.003, 100000.125], device="cuda", dtype=torch.float32),
        torch.ones(128, device="cuda", dtype=torch.float16),
    ]
    if reverse:
        grads.reverse()
    snapshots = [g.clone() for g in grads]
    overflow = torch.zeros(1, device="cuda", dtype=torch.int32)
    grouped = make_dtype_grouped_applier(multi_tensor_applier, multi_tensor_l2norm)
    total, _ = grouped(multi_tensor_l2norm, overflow, [grads], False)
    torch.cuda.synchronize()
    reference = torch.linalg.vector_norm(torch.cat([g.float() for g in grads]))
    torch.testing.assert_close(total.reshape(()), reference, rtol=1e-6, atol=0)
    for grad, original in zip(grads, snapshots, strict=True):
        torch.testing.assert_close(grad, original, rtol=0, atol=0)
