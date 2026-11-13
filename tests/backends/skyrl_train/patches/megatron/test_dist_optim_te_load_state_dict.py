"""CPU test for ``patch_dist_optim_te_load_state_dict`` with a fake FusedAdam and a fake upstream load."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron


class _FakeFusedAdam(torch.optim.Adam):
    def state_dict(self):  # TE's override makes fp32 copies; the patch must bypass it
        raise AssertionError("FusedAdam.state_dict() called during load")

    def load_state_dict(self, state_dict):  # TE's override re-creates the state
        raise AssertionError("FusedAdam.load_state_dict() called during load")


def _upstream_like(self, state_dict):
    inner_state_dict = self.optimizer.state_dict()
    groups = [
        {**saved, "params": inner["params"]}
        for inner, saved in zip(inner_state_dict["param_groups"], state_dict["optimizer"]["param_groups"])
    ]
    state = inner_state_dict["state"] if len(self.optimizer.state) else {"placeholder": True}
    self.optimizer.load_state_dict({"state": state, "param_groups": groups})
    return "upstream-done"


@pytest.fixture
def patched(monkeypatch):
    import transformer_engine.pytorch.optimizers as te_optimizers
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_dist_optim_te_load_state_dict as patch,
    )

    monkeypatch.setattr(te_optimizers, "FusedAdam", _FakeFusedAdam)
    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", _upstream_like)
    monkeypatch.setattr(patch, "_APPLIED", False)
    patch.apply_dist_optim_te_load_state_dict_patch()
    return DistributedOptimizer.load_state_dict


def _optimizer_with_state():
    params = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(3))]
    opt = _FakeFusedAdam([{"params": params[:1], "lr": 1e-3}, {"params": params[1:], "lr": 2e-3}])
    for p in params:
        opt.state[p] = {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
    return opt, params


def test_allocated_state_is_not_copied_and_groups_are_restored(patched):
    opt, params = _optimizer_with_state()
    before = {id(p): {k: v.data_ptr() for k, v in opt.state[p].items()} for p in params}
    saved = {
        "optimizer": {
            "param_groups": [{"params": [0], "lr": 5e-4, "step": 40}, {"params": [1], "lr": 7e-4, "step": 40}]
        }
    }

    assert patched(SimpleNamespace(optimizer=opt), saved) == "upstream-done"

    assert [g["lr"] for g in opt.param_groups] == [5e-4, 7e-4]
    assert [g["step"] for g in opt.param_groups] == [40, 40]
    assert [g["params"] for g in opt.param_groups] == [params[:1], params[1:]]
    assert {id(p): {k: v.data_ptr() for k, v in opt.state[p].items()} for p in params} == before
    # The method overrides are removed again.
    assert "state_dict" not in vars(opt) and "load_state_dict" not in vars(opt)


def test_empty_state_takes_upstream_path(patched):
    params = [torch.nn.Parameter(torch.ones(4))]
    opt = _FakeFusedAdam(params)
    with pytest.raises(AssertionError, match="state_dict"):
        patched(SimpleNamespace(optimizer=opt), {"optimizer": {"param_groups": [{"params": [0], "lr": 1.0}]}})
