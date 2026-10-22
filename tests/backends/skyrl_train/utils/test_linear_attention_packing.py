"""Unit tests for packed-sequence boundary kwargs handed to Gated DeltaNet layers."""

import inspect
import sys
import types

import pytest
import torch

from skyrl.backends.skyrl_train.utils.linear_attention_packing import (
    check_boundaries_consumed,
    gated_delta_net_layers,
    packed_sequence_kwargs,
)

SEQUENCE_LENGTHS = (7, 13, 5)


class BoundaryAwareGatedDeltaNet(torch.nn.Module):
    """Mirrors transformers >= 5.9: the kwarg reaches the chunked delta rule."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states, kwargs.pop("cu_seq_lens_q", None)


class LeakyGatedDeltaNet(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


def _model(*layers):
    return torch.nn.Sequential(torch.nn.Linear(4, 4), *layers)


def _fla(chunk_gated_delta_rule):
    return types.SimpleNamespace(chunk_gated_delta_rule=chunk_gated_delta_rule)


def _conv(causal_conv1d_fn):
    return types.SimpleNamespace(causal_conv1d_fn=causal_conv1d_fn)


def _varlen_conv(x, weight, bias=None, seq_idx=None, activation=None):
    return None


@pytest.fixture
def fla_with_varlen(monkeypatch):
    def chunk_gated_delta_rule(query, key, value, g, beta, cu_seqlens=None):
        return None

    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", _fla(chunk_gated_delta_rule))
    monkeypatch.setitem(sys.modules, "causal_conv1d", _conv(_varlen_conv))


def test_packed_sequence_kwargs_marks_boundaries_where_position_ids_restart():
    position_ids = torch.cat([torch.arange(length) for length in SEQUENCE_LENGTHS])[None]

    kwargs = packed_sequence_kwargs(position_ids)

    assert kwargs["cu_seq_lens_q"].tolist() == [0, 7, 20, 25]
    assert kwargs["cu_seq_lens_q"].dtype == torch.int32
    assert kwargs["cu_seq_lens_k"] is kwargs["cu_seq_lens_q"]
    assert kwargs["max_length_q"] == kwargs["max_length_k"] == 13
    assert kwargs["seq_idx"].tolist() == [[0] * 7 + [1] * 13 + [2] * 5]
    assert kwargs["seq_idx"].dtype == torch.int32


def test_single_sequence_yields_trivial_boundaries():
    position_ids = torch.arange(9)[None]

    kwargs = packed_sequence_kwargs(position_ids)

    assert kwargs["cu_seq_lens_q"].tolist() == [0, 9]
    assert kwargs["max_length_q"] == 9
    assert kwargs["seq_idx"].tolist() == [[0] * 9]


def test_boundary_dtype_and_device_follow_position_ids(fla_with_varlen):
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    kwargs = packed_sequence_kwargs(position_ids)

    assert kwargs["cu_seq_lens_q"].device == position_ids.device
    assert kwargs["seq_idx"].device == position_ids.device


def test_finds_layers_without_importing_kernels():
    gdn = [BoundaryAwareGatedDeltaNet(), BoundaryAwareGatedDeltaNet()]

    assert gated_delta_net_layers(_model(*gdn)) == gdn


def test_returns_nothing_for_models_without_gated_delta_net():
    assert gated_delta_net_layers(_model()) == []


def test_rejects_forward_that_ignores_boundaries(fla_with_varlen):
    with pytest.raises(RuntimeError, match="does not read cu_seq_lens_q"):
        check_boundaries_consumed(LeakyGatedDeltaNet())


@pytest.mark.parametrize("module", ["fla.ops.gated_delta_rule", "causal_conv1d"])
def test_rejects_missing_kernel_package(monkeypatch, fla_with_varlen, module):
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(RuntimeError, match=f"{module}.* is required"):
        check_boundaries_consumed(BoundaryAwareGatedDeltaNet())


def test_rejects_delta_rule_without_varlen_support(monkeypatch, fla_with_varlen):
    def chunk_gated_delta_rule(query, key, value, g, beta):
        return None

    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", _fla(chunk_gated_delta_rule))

    with pytest.raises(RuntimeError, match="no cu_seqlens support"):
        check_boundaries_consumed(BoundaryAwareGatedDeltaNet())


def test_rejects_conv_without_sequence_index_support(monkeypatch, fla_with_varlen):
    def causal_conv1d_fn(x, weight, bias=None, activation=None):
        return None

    monkeypatch.setitem(sys.modules, "causal_conv1d", _conv(causal_conv1d_fn))

    with pytest.raises(RuntimeError, match="no seq_idx support"):
        check_boundaries_consumed(BoundaryAwareGatedDeltaNet())


def test_accepts_kwarg_read_with_explicit_default(fla_with_varlen):
    class DefaultedGatedDeltaNet(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states, kwargs.pop("cu_seq_lens_q", None)

    check_boundaries_consumed(DefaultedGatedDeltaNet())


def test_boundary_check_does_not_read_source_files(monkeypatch, fla_with_varlen):
    def unavailable(*args, **kwargs):
        raise OSError("source not available")

    monkeypatch.setattr(inspect, "getsource", unavailable)

    # The guard reads the compiled code object, so a stripped-source environment still checks.
    check_boundaries_consumed(BoundaryAwareGatedDeltaNet())


def test_packed_sequence_kwargs_rejects_non_zero_start():
    # position_ids always restart at 0 on the packed path (derived from the attention mask); a
    # non-zero start would produce a malformed cu_seqlens, so fail loudly instead of guessing.
    with pytest.raises(ValueError, match="start at 0"):
        packed_sequence_kwargs(torch.tensor([[10, 11, 12, 0, 1, 2]]))
