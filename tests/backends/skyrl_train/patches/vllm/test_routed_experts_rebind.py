"""CPU tests for ``patch_routed_experts_rebind`` (vllm-project/vllm#59455 backport).

Ported from the upstream PR's ``test_monolithic_kernel_rebuild_*`` tests: a concrete monolithic
experts object built without ``__init__`` stands in for FlashInfer TRT-LLM's, so no GPU is needed.
"""

import types
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm")

pytestmark = pytest.mark.vllm


@pytest.fixture(autouse=True)
def rebind_patch(monkeypatch):
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    from skyrl.backends.skyrl_train.patches.vllm import patch_routed_experts_rebind

    had_property = isinstance(FusedMoEMethodBase.__dict__.get("moe_kernel"), property)
    monkeypatch.setattr(patch_routed_experts_rebind, "_PATCHED", False)
    installed = patch_routed_experts_rebind.apply_routed_experts_rebind_patch()
    assert installed or had_property
    yield
    if installed:
        del FusedMoEMethodBase.moe_kernel


def _monolithic_experts(supports_capture: bool = True):
    from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import CPUExpertsInt4

    fused_experts = CPUExpertsInt4.__new__(CPUExpertsInt4)
    fused_experts.supports_routing_replay_capture = lambda: supports_capture
    bound = []

    def set_capture_fn(capture_fn):
        bound.append(capture_fn)
        fused_experts.routing_replay_capture_fn = capture_fn

    fused_experts.set_capture_fn = set_capture_fn
    return fused_experts, bound


def _moe_method():
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    class DummyMoEMethod(FusedMoEMethodBase):
        def create_weights(self, *args, **kwargs):
            pass

        def get_fused_moe_quant_config(self, layer):
            return None

        def apply(self, *args, **kwargs):
            raise NotImplementedError

    return DummyMoEMethod(types.SimpleNamespace())


def _kernel(fused_experts):
    return types.SimpleNamespace(impl=types.SimpleNamespace(fused_experts=fused_experts))


def test_monolithic_kernel_rebuild_keeps_capture_binding():
    """process_weights_after_loading rebuilds moe_kernel on every layerwise reload; the
    routed-experts callback must move to the new experts."""
    method = _moe_method()
    old_experts, _ = _monolithic_experts()
    method.moe_kernel = _kernel(old_experts)
    capture_fn = Mock()
    old_experts.set_capture_fn(capture_fn)  # what bind_routed_experts_capturer does

    new_experts, bound = _monolithic_experts()
    method.moe_kernel = _kernel(new_experts)

    assert bound == [capture_fn]
    assert new_experts.routing_replay_capture_fn is capture_fn


def test_monolithic_kernel_rebuild_keeps_replay_buffer():
    """CUDA graphs captured before the rebuild write into the old replay buffer; the rebuilt
    experts must reuse it, not allocate (and free the old) one."""
    method = _moe_method()
    old_experts, _ = _monolithic_experts()
    method.moe_kernel = _kernel(old_experts)
    capture_fn = Mock()
    old_experts.routing_replay_capture_fn = capture_fn
    old_experts._routing_replay_buffer = buffer = torch.empty(16, 2, dtype=torch.int16)

    new_experts, bound = _monolithic_experts()
    method.moe_kernel = _kernel(new_experts)

    assert bound == []  # no new allocation through set_capture_fn
    assert new_experts.routing_replay_capture_fn is capture_fn
    assert new_experts._routing_replay_buffer is buffer


def test_monolithic_kernel_rebuild_without_capture_binds_nothing():
    method = _moe_method()
    method.moe_kernel = _kernel(_monolithic_experts()[0])

    new_experts, bound = _monolithic_experts()
    method.moe_kernel = _kernel(new_experts)

    assert bound == []


def test_monolithic_kernel_rebuild_rejects_kernel_without_replay_support():
    method = _moe_method()
    old_experts, _ = _monolithic_experts()
    method.moe_kernel = _kernel(old_experts)
    old_experts.set_capture_fn(Mock())

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        method.moe_kernel = _kernel(_monolithic_experts(supports_capture=False)[0])
