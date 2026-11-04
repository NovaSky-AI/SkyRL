"""``patch_hybrid_mhc_release_attn_norm_manager``: the hybrid mHC fast path leaves no ``attn_norm_manager``
(and so no reference to the layer's input and its graph) on the inner attention layer.

CPU, fake layer state.

Run with:
uv run --isolated --extra dev --extra megatron pytest \
    tests/backends/skyrl_train/patches/megatron/test_hybrid_mhc_release_attn_norm_manager.py
"""

import gc
import types
import weakref

import pytest
import torch

pytestmark = pytest.mark.megatron


def _patched(monkeypatch, fake):
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
        HyperConnectionHybridLayer,
    )

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_hybrid_mhc_release_attn_norm_manager as mod,
    )

    monkeypatch.setattr(HyperConnectionHybridLayer, "_call_inner_transformer_layer_without_local_bda", fake)
    monkeypatch.setattr(mod, "_APPLIED", False)
    assert mod.patch_hybrid_mhc_release_attn_norm_manager()
    assert mod.patch_hybrid_mhc_release_attn_norm_manager()  # idempotent: wrapped once
    return HyperConnectionHybridLayer


def test_fast_path_releases_attn_norm_manager_and_the_graph(monkeypatch):
    seen = []

    def fake(self, hidden_states, *args, **kwargs):
        # As megatron-core: the attention branch leaves the offload interface (holding the input) on the layer.
        self.inner_layer.attn_norm_manager = types.SimpleNamespace(tensor=hidden_states)
        seen.append(self.inner_layer.attn_norm_manager is not None)
        return (hidden_states * 2, None), None, 0.0, False

    cls = _patched(monkeypatch, fake)
    layer = object.__new__(cls)
    object.__setattr__(layer, "inner_layer", types.SimpleNamespace(attn_norm_manager=None))

    leaf = torch.ones(4, 3, requires_grad=True)
    aggregated = leaf * 3  # non-leaf with a graph back to ``leaf``, as the mHC aggregate output
    (out, _), *_ = cls._call_inner_transformer_layer_without_local_bda(layer, aggregated, None)
    out.sum().backward()

    assert seen == [True]
    assert layer.inner_layer.attn_norm_manager is None
    ref = weakref.ref(leaf)
    del leaf, aggregated, out
    gc.collect()
    assert ref() is None  # nothing on the layer keeps the input's graph (and the leaf) alive


def test_releases_even_when_the_layer_raises(monkeypatch):
    def fake(self, *args, **kwargs):
        self.inner_layer.attn_norm_manager = object()
        raise RuntimeError("boom")

    cls = _patched(monkeypatch, fake)
    layer = object.__new__(cls)
    object.__setattr__(layer, "inner_layer", types.SimpleNamespace(attn_norm_manager=None))
    with pytest.raises(RuntimeError, match="boom"):
        cls._call_inner_transformer_layer_without_local_bda(layer, torch.zeros(1))
    assert layer.inner_layer.attn_norm_manager is None
