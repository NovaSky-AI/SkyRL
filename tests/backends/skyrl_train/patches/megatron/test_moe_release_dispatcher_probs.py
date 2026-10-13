"""``patch_moe_release_dispatcher_probs``: ``MoELayer.postprocess`` leaves no ``probs`` on the dispatcher.

CPU, fake layer state; the memory effect itself is covered end to end by
``gpu_ci/patches/megatron/test_mhc_full_recompute.py``.

Run with:
uv run --isolated --extra dev --extra megatron pytest \
    tests/backends/skyrl_train/patches/megatron/test_moe_release_dispatcher_probs.py
"""

import types

import pytest
import torch

pytestmark = pytest.mark.megatron


def test_postprocess_releases_probs_and_returns_inner_output(monkeypatch):
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_moe_release_dispatcher_probs as mod,
    )

    calls = []

    def fake_postprocess(self, output, shared_expert_output):
        # The dispatcher still holds probs while the layer's own postprocess runs.
        calls.append(self.token_dispatcher.probs is not None)
        return output + 1

    monkeypatch.setattr(MoELayer, "postprocess", fake_postprocess)
    monkeypatch.setattr(mod, "_APPLIED", False)
    assert mod.patch_moe_release_dispatcher_probs()
    assert mod.patch_moe_release_dispatcher_probs()  # idempotent: wrapped once

    layer = object.__new__(MoELayer)
    probs = torch.rand(4, 2, requires_grad=True) * 2  # non-leaf, has a grad_fn
    object.__setattr__(layer, "token_dispatcher", types.SimpleNamespace(probs=probs))
    out = MoELayer.postprocess(layer, torch.zeros(3), None)

    assert torch.equal(out, torch.ones(3))
    assert calls == [True]
    assert layer.token_dispatcher.probs is None


def test_dispatchers_without_probs_are_left_alone(monkeypatch):
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_moe_release_dispatcher_probs as mod,
    )

    monkeypatch.setattr(MoELayer, "postprocess", lambda self, output, shared: output)
    monkeypatch.setattr(mod, "_APPLIED", False)
    mod.patch_moe_release_dispatcher_probs()

    layer = object.__new__(MoELayer)
    dispatcher = types.SimpleNamespace(local_probs=torch.ones(2))  # e.g. the all-gather dispatcher
    object.__setattr__(layer, "token_dispatcher", dispatcher)
    MoELayer.postprocess(layer, torch.zeros(1), None)
    assert not hasattr(dispatcher, "probs") and torch.equal(dispatcher.local_probs, torch.ones(2))
