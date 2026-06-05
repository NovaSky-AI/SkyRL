"""CPU unit tests for the MTP hidden-state capture / decoupled replay plumbing.

These use a fake MTP block (no Megatron) to verify the capture records the block's
arguments and that the decoupled replay detaches the trunk hidden states so the
draft gradient never reaches the policy backbone.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_hidden_capture.py
"""

import sys
import types

import torch
import torch.nn as nn

# Stub out megatron.core.utils.unwrap_model so hidden_capture imports on CPU.
_fake_mcore_utils = types.ModuleType("megatron.core.utils")
_fake_mcore_utils.unwrap_model = lambda m: m
sys.modules.setdefault("megatron", types.ModuleType("megatron"))
sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
sys.modules["megatron.core.utils"] = _fake_mcore_utils

from skyrl.backends.skyrl_train.mtp.hidden_capture import MTPHiddenCapture  # noqa: E402


class _FakeMTPBlock(nn.Module):
    """Mimics MultiTokenPredictionBlock: returns cat([trunk; mtp_0], dim=0)."""

    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Parameter(torch.ones(hidden))

    def forward(self, hidden_states, **kwargs):
        mtp_0 = hidden_states * self.w  # depends on params AND trunk hidden
        return torch.cat([hidden_states, mtp_0], dim=0)


class _FakeGPT(nn.Module):
    def __init__(self, hidden=4, mtp_num_layers=1):
        super().__init__()
        self.mtp = _FakeMTPBlock(hidden)
        self.config = types.SimpleNamespace(mtp_num_layers=mtp_num_layers)


def _run(detach_trunk):
    gpt = _FakeGPT()
    capture = MTPHiddenCapture(gpt, detach_trunk=detach_trunk)
    s, b, h = 3, 2, 4
    trunk = torch.randn(s, b, h, requires_grad=True)
    with capture.capture():
        # Simulate GPTModel's in-forward MTP call (records kwargs via the pre-hook).
        _ = gpt.mtp(hidden_states=trunk, position_ids=torch.zeros(b, s))
    student = capture.compute_student_hidden_states()
    return gpt, trunk, student


def test_capture_returns_one_chunk_per_mtp_depth():
    _, _, student = _run(detach_trunk=True)
    assert student is not None and len(student) == 1
    assert student[0].shape == (3, 2, 4)


def test_replay_detaches_trunk_when_detach_trunk_true():
    gpt, trunk, student = _run(detach_trunk=True)
    student[0].sum().backward()
    # The MTP-head parameter receives gradient...
    assert gpt.mtp.w.grad is not None and gpt.mtp.w.grad.abs().sum() > 0
    # ...but the trunk hidden states do NOT (decoupled).
    assert trunk.grad is None


def test_replay_couples_trunk_when_detach_trunk_false():
    gpt, trunk, student = _run(detach_trunk=False)
    student[0].sum().backward()
    assert gpt.mtp.w.grad is not None and gpt.mtp.w.grad.abs().sum() > 0
    # With detach disabled, the gradient flows back into the trunk hidden states.
    assert trunk.grad is not None and trunk.grad.abs().sum() > 0


def test_no_capture_when_block_absent():
    gpt = _FakeGPT()
    gpt.mtp = None
    capture = MTPHiddenCapture(gpt, detach_trunk=True)
    with capture.capture():
        pass
    assert capture.compute_student_hidden_states() is None
