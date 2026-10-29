"""expert_factor_for_param: one grouped-expert LoRA factor from per-expert adapter tensors."""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.lora_import import expert_factor_for_param


def _per_expert(n=4, shape=(2, 3), shared=False):
    base = torch.randn(*shape)
    return {e: base.clone() if shared else torch.randn(*shape) for e in range(n)}


class TestExpertFactorForParam:
    def test_shared_factor(self):
        f = _per_expert(shared=True)
        assert torch.equal(expert_factor_for_param(f, (2, 3), 1, "x"), f[0])

    def test_shared_factor_rejects_divergent_expert(self):
        f = _per_expert(shared=True)
        f[3] = f[3].clone()
        f[3][0, 0] += 1e-3
        with pytest.raises(ValueError, match=r"experts \[3\] differ"):
            expert_factor_for_param(f, (2, 3), 0, "x")

    @pytest.mark.parametrize("ep_rank,experts", [(0, [0, 1]), (1, [2, 3])], ids=["ep0", "ep1"])
    def test_local_pack(self, ep_rank, experts):
        f = _per_expert()
        out = expert_factor_for_param(f, (2, 2, 3), ep_rank, "x")
        assert torch.equal(out, torch.stack([f[e] for e in experts]))

    @pytest.mark.parametrize(
        "factors,shape,match",
        [
            ({0: torch.zeros(2, 3), 2: torch.zeros(2, 3)}, (2, 3), "one tensor per expert"),
            (_per_expert(n=4), (3, 2, 3), "exceeds"),
            (_per_expert(n=4, shared=True), (3, 2), "shape"),
        ],
        ids=["missing-expert", "too-many-local", "shape-mismatch"],
    )
    def test_errors(self, factors, shape, match):
        with pytest.raises(ValueError, match=match):
            expert_factor_for_param(factors, shape, 1, "x")
