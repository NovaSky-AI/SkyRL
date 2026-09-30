"""Smoke tests for the ``freeze_moe_router`` helper.

The helper walks ``model.decoder.layers`` (or ``model.language_model.decoder.layers``
for multimodal models) and flips ``requires_grad`` on
router weights/biases. These tests build minimal mock modules that mimic Megatron-Core's attribute layout
without importing Megatron.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_freeze_moe_router.py
"""

import pytest
import torch
import torch.nn as nn
from loguru import logger

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    freeze_moe_router,
)


class _SharedExperts(nn.Module):
    def __init__(self, in_features: int = 8, hidden: int = 4, with_bias: bool = True):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.randn(hidden, in_features))
        if with_bias:
            self.gate_bias = nn.Parameter(torch.randn(hidden))


class _MLP(nn.Module):
    def __init__(self, with_shared_experts: bool = True):
        super().__init__()
        # Mimic Megatron's TopKRouter: a module with ``weight`` (and optional ``bias``)
        # Parameters. We use nn.Linear here because it has the right attribute layout.
        self.router = nn.Linear(8, 4, bias=True)
        if with_shared_experts:
            self.shared_experts = _SharedExperts(in_features=8, hidden=4, with_bias=True)
        # Non-router param that must remain trainable.
        self.linear_fc1 = nn.Linear(8, 16)


class _Layer(nn.Module):
    def __init__(self, **mlp_kwargs):
        super().__init__()
        self.mlp = _MLP(**mlp_kwargs)


class _Decoder(nn.Module):
    def __init__(self, n_layers: int = 2, **mlp_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(**mlp_kwargs) for _ in range(n_layers)])


class _Model(nn.Module):
    def __init__(self, n_layers: int = 2, **mlp_kwargs):
        super().__init__()
        self.decoder = _Decoder(n_layers=n_layers, **mlp_kwargs)


@pytest.mark.megatron
def test_freeze_moe_router_hyperconnection_wrapped_layers():
    class HyperConnectionHybridLayer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.mhc_weight = nn.Parameter(torch.ones(4))

    model = _Model()
    model.decoder.layers = nn.ModuleList([HyperConnectionHybridLayer(layer) for layer in model.decoder.layers])
    multimodal = nn.Module()
    multimodal.language_model = model

    freeze_moe_router(multimodal)

    for wrapper in model.decoder.layers:
        assert not wrapper.layer.mlp.router.weight.requires_grad
        assert not wrapper.layer.mlp.router.bias.requires_grad
        assert wrapper.layer.mlp.linear_fc1.weight.requires_grad
        assert wrapper.layer.mlp.shared_experts.gate_weight.requires_grad
        assert wrapper.mhc_weight.requires_grad


@pytest.mark.megatron
def test_freeze_moe_router_freezes_router_params():
    m = _Model()
    # sanity: all params start trainable
    assert all(p.requires_grad for p in m.parameters())

    ret = freeze_moe_router(m)
    assert ret is not None
    assert ret == m

    for layer in m.decoder.layers:
        assert layer.mlp.router.weight.requires_grad is False
        assert layer.mlp.router.bias.requires_grad is False


@pytest.mark.megatron
def test_freeze_moe_router_leaves_other_params_trainable():
    m = _Model()

    freeze_moe_router(m)

    for layer in m.decoder.layers:
        assert layer.mlp.linear_fc1.weight.requires_grad is True
        assert layer.mlp.linear_fc1.bias.requires_grad is True
        assert layer.mlp.shared_experts.gate_weight.requires_grad is True
        assert layer.mlp.shared_experts.gate_bias.requires_grad is True


@pytest.mark.megatron
def test_freeze_moe_router_handles_layer_without_router():
    # PP/VPP stages without MoE layers: layer.mlp has no .router attribute.
    class _NonMoEMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = nn.Linear(8, 16)

    class _NonMoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _NonMoEMLP()

    m = _Model()
    m.decoder.layers = nn.ModuleList([_NonMoELayer(), _NonMoELayer()])

    # Should be a no-op without raising.
    freeze_moe_router(m)

    for layer in m.decoder.layers:
        assert layer.mlp.linear_fc1.weight.requires_grad is True


@pytest.mark.megatron
def test_freeze_moe_router_no_bias():
    class _MoEMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(8, 4, bias=False)  # router.bias is None
            self.shared_experts = nn.Module()
            # no gate_bias attr on shared_experts
            self.shared_experts.gate_weight = nn.Parameter(torch.randn(4, 8))
            self.linear_fc1 = nn.Linear(8, 16)

    class _MoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MoEMLP()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([_MoELayer()])

    m = _Model()
    freeze_moe_router(m)  # should NOT raise
    assert not m.decoder.layers[0].mlp.router.weight.requires_grad
    assert m.decoder.layers[0].mlp.shared_experts.gate_weight.requires_grad
    assert m.decoder.layers[0].mlp.linear_fc1.weight.requires_grad


@pytest.mark.megatron
def test_freeze_moe_router_list():
    m = _Model()
    # sanity: all params start trainable
    assert all(p.requires_grad for p in m.parameters())

    ret = freeze_moe_router([m])
    assert isinstance(ret, list)
    assert len(ret) == 1

    for layer in m.decoder.layers:
        assert layer.mlp.router.weight.requires_grad is False
        assert layer.mlp.router.bias.requires_grad is False


@pytest.mark.megatron
def test_freeze_moe_router_multimodal_language_model_nesting():
    """
    Multimodal models (``LLaVAModel`` and the VLM classes derived from it) hold the
    decoder at ``model.language_model.decoder`` and have no ``decoder`` of their own.
    """

    class _MultimodalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _Model(n_layers=2)
            self.vision_model = nn.Linear(8, 8)

    m = _MultimodalModel()
    assert not hasattr(m, "decoder")

    freeze_moe_router(m)

    for layer in m.language_model.decoder.layers:
        assert layer.mlp.router.weight.requires_grad is False
        assert layer.mlp.router.bias.requires_grad is False
    # The vision tower is untouched.
    assert m.vision_model.weight.requires_grad is True


@pytest.mark.megatron
def test_freeze_moe_router_skips_chunk_without_decoder():
    """A pipeline stage holding no decoder layers -- e.g. a vision-only or
    embedding-only chunk -- is skipped rather than raising ``AttributeError``."""

    class _EmbeddingOnlyChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)

    chunk = _EmbeddingOnlyChunk()
    moe_chunk = _Model()

    freeze_moe_router([chunk, moe_chunk])

    assert chunk.embedding.weight.requires_grad is True
    for layer in moe_chunk.decoder.layers:
        assert layer.mlp.router.weight.requires_grad is False


@pytest.mark.megatron
def test_freeze_moe_router_warns_when_nothing_frozen():
    """A rank holding no MoE router (dense-only layers, or no decoder at all) warns
    that nothing was frozen and still returns its input unchanged."""

    class _DenseLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(8, 8)

    class _DenseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([_DenseLayer(), _DenseLayer()])

    class _EmbeddingOnlyChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)

    models = [_EmbeddingOnlyChunk(), _DenseModel()]
    messages = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        ret = freeze_moe_router(models)
    finally:
        logger.remove(handler_id)

    assert ret is models
    assert all(p.requires_grad for m in models for p in m.parameters())
    assert any("no transformer decoder found on _EmbeddingOnlyChunk" in msg for msg in messages)
    assert any("froze no router parameters" in msg for msg in messages)
