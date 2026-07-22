"""CPU unit tests for the MTP separate grad-clip (mtp/grad_clip.py).

The MTP head shares the policy's grad buffer + optimizer; its draft-loss grads are ~20-30 per
microbatch at weight 1.0 (see the GPU probe test_mtp_grad_isolation), so a single global grad-norm
over-clips the policy ~20x. These tests pin the fix on a fake Megatron optimizer: the policy's clip
is computed from policy grads ONLY, the head is clipped by its own norm, and mapping failures fail
loud. The fake mirrors megatron's surface; the real-stack behavior (incl. DP-sharded ownership) is
covered by the GPU probe.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_grad_clip.py
"""

import sys
import types

import pytest
import torch

# Stub the megatron pieces grad_clip imports, with megatron's semantics (L2 norm; scale by
# min(1, max_norm/(total_norm + 1e-6))), so this runs on CPU.
_fake_clip_grads = types.ModuleType("megatron.core.optimizer.clip_grads")


def _get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None):
    if not grads_for_norm:
        return 0.0
    return float(torch.norm(torch.stack([torch.norm(g, 2) for g in grads_for_norm]), 2))


def _clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
    coeff = max_norm / (total_norm + 1.0e-6)
    if coeff < 1.0:
        for p in parameters:
            p.grad.mul_(coeff)


_fake_clip_grads.get_grad_norm_fp32 = _get_grad_norm_fp32
_fake_clip_grads.clip_grad_by_total_norm_fp32 = _clip_grad_by_total_norm_fp32
for _m in ("megatron", "megatron.core", "megatron.core.optimizer"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["megatron.core.optimizer.clip_grads"] = _fake_clip_grads

from skyrl.backends.skyrl_train.mtp.grad_clip import (  # noqa: E402
    install_mtp_separate_grad_clip,
)

CLIP = 1.0


class _FakeSubOptimizer:
    """Mimics the MegatronOptimizer surface grad_clip.py touches."""

    def __init__(self, model_params, main_params):
        # Megatron keeps model params and their optimizer-side "main" params positionally matched.
        self.model_float16_groups = [model_params]
        self.shard_fp32_from_float16_groups = [main_params]
        self._main_params = main_params
        self.config = types.SimpleNamespace(
            clip_grad=CLIP,
            grad_norm_skip_threshold=float("inf"),
            log_num_zeros_in_grad=False,
            use_precision_aware_optimizer=False,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        )

    def get_parameters(self):
        return list(self._main_params)

    def get_main_grads_for_grad_norm(self):
        return [p.grad for p in self.get_parameters() if p.grad is not None]

    def get_grad_norm(self):
        return _get_grad_norm_fp32(self.get_main_grads_for_grad_norm())


class _FakeChainedOptimizer:
    def __init__(self, sub):
        self.chained_optimizers = [sub]
        self.config = sub.config
        self.stepped_with_ready_grads = False

    def get_parameters(self):
        return [p for s in self.chained_optimizers for p in s.get_parameters()]

    def get_grad_norm(self):
        return self.chained_optimizers[0].get_grad_norm()

    def prepare_grads(self):
        return False  # no inf/nan

    def count_zeros(self):
        return 0

    def step_with_ready_grads(self):
        self.stepped_with_ready_grads = True
        return True


class _FakeChunk(torch.nn.Module):
    """A model chunk whose named_parameters() carry megatron-style names. The head is resolved BY
    NAME off the chunk, then to main params by position in the optimizer's group lists."""

    def __init__(self, policy_param, mtp_param):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ParameterList([policy_param])
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ParameterList([mtp_param])


def _build(policy_grad_norm, mtp_grad_norm):
    """Optimizer with one policy param and one MTP param, each carrying a grad of the given norm."""
    policy_model, mtp_model = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    policy_main, mtp_main = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    # A one-hot grad makes the L2 norm exactly the requested value.
    for main, norm in ((policy_main, policy_grad_norm), (mtp_main, mtp_grad_norm)):
        main.grad = torch.tensor([norm, 0.0, 0.0, 0.0])
    sub = _FakeSubOptimizer([policy_model, mtp_model], [policy_main, mtp_main])
    opt = _FakeChainedOptimizer(sub)
    n = install_mtp_separate_grad_clip(opt, [_FakeChunk(policy_model, mtp_model)])
    assert n == 1  # the head's main param was isolated on this (single) rank
    return opt, policy_main, mtp_main


def test_policy_clip_ignores_a_huge_mtp_grad():
    # The regression this exists to prevent: policy norm 0.8 (< clip, so NO clipping), head norm 21.
    # A shared global norm would be ~21 and shrink the policy update ~20x.
    opt, policy_main, mtp_main = _build(policy_grad_norm=0.8, mtp_grad_norm=21.0)
    _, grad_norm, _ = opt.step()

    # Reported grad_norm is the POLICY's, matching a no-MTP run; the head's is stashed separately.
    assert grad_norm == pytest.approx(0.8, rel=1e-4)
    assert opt.mtp_grad_norm == pytest.approx(21.0, rel=1e-4)
    # Policy grad is untouched: its norm is below the clip threshold.
    assert policy_main.grad[0].item() == pytest.approx(0.8, rel=1e-4)
    # The head IS clipped, by its own norm (21 -> 1.0).
    assert torch.norm(mtp_main.grad, 2).item() == pytest.approx(CLIP, rel=1e-3)


def test_policy_still_clipped_by_its_own_norm():
    # Policy above the threshold is clipped normally -- the head must not change that either.
    opt, policy_main, mtp_main = _build(policy_grad_norm=4.0, mtp_grad_norm=21.0)
    _, grad_norm, _ = opt.step()
    assert grad_norm == pytest.approx(4.0, rel=1e-4)
    assert torch.norm(policy_main.grad, 2).item() == pytest.approx(CLIP, rel=1e-3)


def test_head_grad_survives_when_below_clip():
    # The head trains at full strength when its norm is under the threshold (no scaling).
    opt, _, mtp_main = _build(policy_grad_norm=0.5, mtp_grad_norm=0.25)
    opt.step()
    assert mtp_main.grad[0].item() == pytest.approx(0.25, rel=1e-4)


def test_step_still_applies_the_update():
    opt, _, _ = _build(policy_grad_norm=0.5, mtp_grad_norm=0.5)
    update_successful, _, _ = opt.step()
    assert update_successful and opt.stepped_with_ready_grads


def test_raises_when_head_name_does_not_match():
    # If the head's parameter naming drifts, it would silently fall back into the policy's clip --
    # the exact coupling this module removes. Must fail loud instead.
    class _RenamedHeadChunk(torch.nn.Module):
        def __init__(self, policy_param, head_param):
            super().__init__()
            self.decoder = torch.nn.ParameterList([policy_param])
            self.draft_head = torch.nn.ParameterList([head_param])  # no ".mtp." anywhere

    policy_model, head_model = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    policy_main = torch.nn.Parameter(torch.zeros(4))
    policy_main.grad = torch.ones(4)
    opt = _FakeChainedOptimizer(_FakeSubOptimizer([policy_model], [policy_main]))
    with pytest.raises(RuntimeError, match="no parameter name matched"):
        install_mtp_separate_grad_clip(opt, [_RenamedHeadChunk(policy_model, head_model)])


def test_raises_when_named_head_params_are_absent_from_the_optimizer():
    # Names match but the optimizer's model->main groups do not contain them anywhere (megatron
    # renamed its group lists): fail loud rather than clip the head with the policy. (Without
    # torch.distributed initialized, "this rank" is the whole world, so zero locally == zero
    # globally; per-rank zero under real DP sharding is exercised by the GPU probe.)
    policy_model, mtp_model = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    policy_main = torch.nn.Parameter(torch.zeros(4))
    policy_main.grad = torch.ones(4)
    # The optimizer only knows about the policy param -- the head is missing from its groups.
    opt = _FakeChainedOptimizer(_FakeSubOptimizer([policy_model], [policy_main]))
    with pytest.raises(RuntimeError, match="mapped to optimizer main params"):
        install_mtp_separate_grad_clip(opt, [_FakeChunk(policy_model, mtp_model)])
