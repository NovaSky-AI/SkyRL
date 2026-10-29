"""An EP rank whose experts receive no tokens must still back-propagate into the dispatched tokens and probs:
their backward is the token dispatcher's EP all-to-all, and a rank that skips it deadlocks the others (seen on the
slice: rank 0 got 0 rows in a layer and dropped the probs all-to-all; the other ranks hung at the next one)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
experts = pytest.importorskip("skyrl.backends.skyrl_train.patches.megatron.ml4.experts")


def _stub(with_adapter: bool):
    lin = torch.nn.Linear(4, 4)
    if with_adapter:
        fc = SimpleNamespace(to_wrap=torch.nn.Identity(), adapter=lin, _adapter_enabled=True)
    else:
        fc = torch.nn.Identity()
    return SimpleNamespace(linear_fc1=fc, linear_fc2=fc, inter=2, act_qat=True, fc2_shards=1, chunk=16), lin


class TestEmptyRankKeepsDispatchInGraph:
    @pytest.mark.parametrize("with_adapter", [False, True], ids=["no-lora", "lora"])
    def test_tokens_and_probs_get_gradients(self, with_adapter):
        stub, lin = _stub(with_adapter)
        x = torch.zeros(0, 4, requires_grad=True)
        probs = torch.zeros(0, requires_grad=True)
        out, _ = experts.ML4NVFP4GroupedMLP.forward(stub, x, torch.zeros(3, dtype=torch.long), probs)
        assert out.shape == (0, 4) and out.dtype == torch.float32
        inputs = [x, probs] + ([lin.weight] if with_adapter else [])
        grads = torch.autograd.grad(out.sum(), inputs, allow_unused=True)
        assert all(g is not None for g in grads)
        assert grads[0].shape == x.shape and grads[1].shape == probs.shape
        if with_adapter:
            assert torch.count_nonzero(grads[2]) == 0
