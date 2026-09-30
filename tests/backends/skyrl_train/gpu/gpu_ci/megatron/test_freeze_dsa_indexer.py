"""Smoke tests for the ``freeze_dsa_indexer`` helper.

The helper walks the subtrees of ``model.decoder.layers`` and ``model.mtp.layers`` and
clears ``requires_grad`` on every parameter under ``self_attention.core_attention.indexer``. These tests build
minimal mock modules that mimic Megatron-Core's attribute layout without importing
Megatron.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_freeze_dsa_indexer.py
"""

import pytest
import torch
import torch.nn as nn

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    freeze_dsa_indexer,
)


class _Indexer(nn.Module):
    """The seven parameters a kpool DSA indexer carries per layer."""

    def __init__(self, dim: int = 8, heads: int = 4):
        super().__init__()
        self.index_kpool_compress_ape = nn.Parameter(torch.randn(4, dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.randn(4))
        self.k_norm = nn.LayerNorm(dim)
        self.linear_weights_proj = nn.Linear(dim, heads, bias=False)
        self.linear_wk = nn.Linear(dim, dim, bias=False)
        self.linear_wq_b = nn.Linear(dim, dim, bias=False)


class _SparseAttention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.linear_qkv = nn.Linear(dim, 3 * dim)
        self.linear_proj = nn.Linear(dim, dim)
        self.core_attention = nn.Module()
        self.core_attention.indexer = _Indexer(dim=dim)


class _DenseAttention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.linear_qkv = nn.Linear(dim, 3 * dim)
        self.linear_proj = nn.Linear(dim, dim)
        self.core_attention = nn.Module()


class _Layer(nn.Module):
    def __init__(self, sparse: bool = True, dim: int = 8):
        super().__init__()
        self.self_attention = _SparseAttention(dim) if sparse else _DenseAttention(dim)
        self.mlp = nn.Linear(dim, 2 * dim)


class _Model(nn.Module):
    def __init__(self, n_layers: int = 2, sparse: bool = True):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_Layer(sparse=sparse) for _ in range(n_layers)])


def _indexer_params(model: nn.Module):
    return [p for layer in model.decoder.layers for p in layer.self_attention.core_attention.indexer.parameters()]


@pytest.mark.megatron
def test_freeze_dsa_indexer_hyperconnection_wrapped_layers():
    class HyperConnectionHybridLayer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.mhc_weight = nn.Parameter(torch.ones(4))

    model = _Model()
    indexer_params = _indexer_params(model)
    model.decoder.layers = nn.ModuleList([HyperConnectionHybridLayer(layer) for layer in model.decoder.layers])
    multimodal = nn.Module()
    multimodal.language_model = model

    freeze_dsa_indexer(multimodal)

    assert not any(param.requires_grad for param in indexer_params)
    for wrapper in model.decoder.layers:
        assert wrapper.mhc_weight.requires_grad
        assert wrapper.layer.self_attention.linear_qkv.weight.requires_grad
        assert wrapper.layer.mlp.weight.requires_grad


@pytest.mark.megatron
def test_freeze_dsa_indexer_freezes_indexer_params():
    m = _Model()
    # sanity: all params start trainable
    assert all(p.requires_grad for p in m.parameters())

    ret = freeze_dsa_indexer(m)
    assert ret is m

    frozen = _indexer_params(m)
    # 7 params per layer: 2 bare Parameters, k_norm weight+bias, 3 projection weights.
    assert len(frozen) == 14
    assert not any(p.requires_grad for p in frozen)


@pytest.mark.megatron
def test_freeze_dsa_indexer_leaves_other_params_trainable():
    m = _Model()

    freeze_dsa_indexer(m)

    for layer in m.decoder.layers:
        assert layer.self_attention.linear_qkv.weight.requires_grad is True
        assert layer.self_attention.linear_proj.weight.requires_grad is True
        assert layer.mlp.weight.requires_grad is True


@pytest.mark.megatron
def test_freeze_dsa_indexer_handles_layer_without_indexer():
    # Dense-attention models, and DSA models' dense prefix layers, have no indexer.
    m = _Model(sparse=False)

    # Should be a no-op without raising.
    freeze_dsa_indexer(m)

    assert all(p.requires_grad for p in m.parameters())


@pytest.mark.megatron
def test_freeze_dsa_indexer_handles_layer_without_self_attention():
    # MTP heads and Mamba/linear-attention layers carry no ``self_attention`` at all.
    class _MTPLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(8, 16)

    m = _Model()
    m.decoder.layers = nn.ModuleList([_MTPLayer(), _Layer(sparse=True)])

    freeze_dsa_indexer(m)

    assert m.decoder.layers[0].mlp.weight.requires_grad is True
    assert not any(p.requires_grad for p in m.decoder.layers[1].self_attention.core_attention.indexer.parameters())


@pytest.mark.megatron
def test_freeze_dsa_indexer_mixed_dense_and_sparse_layers():
    m = _Model()
    m.decoder.layers = nn.ModuleList([_Layer(sparse=False), _Layer(sparse=True), _Layer(sparse=False)])

    freeze_dsa_indexer(m)

    assert all(p.requires_grad for p in m.decoder.layers[0].parameters())
    assert not any(p.requires_grad for p in m.decoder.layers[1].self_attention.core_attention.indexer.parameters())
    assert m.decoder.layers[1].self_attention.linear_qkv.weight.requires_grad is True
    assert all(p.requires_grad for p in m.decoder.layers[2].parameters())


@pytest.mark.megatron
def test_freeze_dsa_indexer_list():
    m = _Model()

    ret = freeze_dsa_indexer([m])
    assert isinstance(ret, list)
    assert len(ret) == 1

    assert not any(p.requires_grad for p in _indexer_params(m))


@pytest.mark.megatron
def test_freeze_dsa_indexer_multimodal_language_model_nesting():
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

    freeze_dsa_indexer(m)

    assert not any(p.requires_grad for p in _indexer_params(m.language_model))
    # The vision tower is untouched.
    assert m.vision_model.weight.requires_grad is True


@pytest.mark.megatron
def test_freeze_dsa_indexer_skips_chunk_without_decoder():
    """A pipeline stage holding no decoder layers -- e.g. a vision-only or
    embedding-only chunk -- is skipped rather than raising ``AttributeError``."""

    class _EmbeddingOnlyChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)

    chunk = _EmbeddingOnlyChunk()
    dsa_chunk = _Model()

    freeze_dsa_indexer([chunk, dsa_chunk])

    assert chunk.embedding.weight.requires_grad is True
    assert not any(p.requires_grad for p in _indexer_params(dsa_chunk))


@pytest.mark.megatron
def test_freeze_dsa_indexer_is_idempotent():
    m = _Model()

    freeze_dsa_indexer(m)
    freeze_dsa_indexer(m)

    assert not any(p.requires_grad for p in _indexer_params(m))
    assert m.decoder.layers[0].self_attention.linear_qkv.weight.requires_grad is True


class _MTPLayer(nn.Module):
    def __init__(self, sparse: bool = True):
        super().__init__()
        self.eh_proj = nn.Linear(16, 8)
        self.mtp_model_layer = _Layer(sparse=sparse)


@pytest.mark.megatron
def test_freeze_dsa_indexer_mtp_layers():
    """MTP depths sit at ``model.mtp.layers[i].mtp_model_layer``, beside the decoder."""
    m = _Model()
    m.mtp = nn.Module()
    m.mtp.layers = nn.ModuleList([_MTPLayer(sparse=True), _MTPLayer(sparse=False)])

    freeze_dsa_indexer(m)

    sparse_mtp, dense_mtp = m.mtp.layers
    assert not any(
        p.requires_grad for p in sparse_mtp.mtp_model_layer.self_attention.core_attention.indexer.parameters()
    )
    assert sparse_mtp.mtp_model_layer.self_attention.linear_qkv.weight.requires_grad is True
    assert sparse_mtp.eh_proj.weight.requires_grad is True
    assert all(p.requires_grad for p in dense_mtp.parameters())
    assert not any(p.requires_grad for p in _indexer_params(m))


@pytest.mark.megatron
def test_freeze_dsa_indexer_multimodal_mtp_layers():
    """Multimodal models hold the MTP block on the language tower, as ``language_model.mtp``."""
    language_model = _Model()
    language_model.mtp = nn.Module()
    language_model.mtp.layers = nn.ModuleList([_MTPLayer()])
    multimodal = nn.Module()
    multimodal.language_model = language_model

    freeze_dsa_indexer(multimodal)

    mtp_indexer = language_model.mtp.layers[0].mtp_model_layer.self_attention.core_attention.indexer
    assert not any(p.requires_grad for p in mtp_indexer.parameters())
    assert not any(p.requires_grad for p in _indexer_params(language_model))
