"""CPU-only smoke tests for the ``freeze_moe_router`` helper.

The helper walks ``model.decoder.layers`` and flips ``requires_grad`` on
router weights/biases and the shared-expert gate weight/bias. These tests
build minimal mock modules that mimic Megatron-Core's attribute layout
without importing Megatron.
"""

import torch
import torch.nn as nn

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


def test_freeze_moe_router_freezes_router_params():
    m = _Model()
    # sanity: all params start trainable
    assert all(p.requires_grad for p in m.parameters())

    freeze_moe_router(m)

    for layer in m.decoder.layers:
        assert layer.mlp.router.weight.requires_grad is False
        assert layer.mlp.router.bias.requires_grad is False


def test_freeze_moe_router_freezes_shared_expert_gate():
    m = _Model()

    freeze_moe_router(m)

    for layer in m.decoder.layers:
        assert layer.mlp.shared_experts.gate_weight.requires_grad is False
        assert layer.mlp.shared_experts.gate_bias.requires_grad is False


def test_freeze_moe_router_leaves_other_params_trainable():
    m = _Model()

    freeze_moe_router(m)

    for layer in m.decoder.layers:
        assert layer.mlp.linear_fc1.weight.requires_grad is True
        assert layer.mlp.linear_fc1.bias.requires_grad is True


def test_freeze_moe_router_handles_missing_shared_experts():
    m = _Model(with_shared_experts=False)

    # Should not raise when shared_experts attribute is absent.
    freeze_moe_router(m)

    for layer in m.decoder.layers:
        assert layer.mlp.router.weight.requires_grad is False


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


def test_freeze_moe_router_no_bias():
    class _MoEMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(8, 4, bias=False)  # router.bias is None
            self.shared_experts = nn.Module()
            self.shared_experts.gate_weight = nn.Parameter(torch.randn(4, 8))
            # deliberately no gate_bias attr on shared_experts
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
    assert not m.decoder.layers[0].mlp.shared_experts.gate_weight.requires_grad
    assert m.decoder.layers[0].mlp.linear_fc1.weight.requires_grad


def test_freeze_moe_router_two_level_wrap():
    """
    Under Megatron's ``bf16=True`` path, chunks are wrapped as
    ``DDP(Float16Module(GPTModel))``. This tests whether `freeze_moe_router`
    can handle 2 levels of wrapping.
    """
    inner_model = _Model()

    class FakeFloat16Module(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

    class FakeDDP(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

    wrapped = FakeDDP(FakeFloat16Module(inner_model))

    # Pass the wrapped chunk directly — the helper must unwrap internally.
    freeze_moe_router(wrapped)

    for layer in inner_model.decoder.layers:
        assert not layer.mlp.router.weight.requires_grad
        assert layer.mlp.linear_fc1.weight.requires_grad
