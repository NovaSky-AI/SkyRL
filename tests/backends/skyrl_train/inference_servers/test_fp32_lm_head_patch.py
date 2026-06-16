"""Tests for the SkyRL fp32 LM-head vLLM patch.

Probe the matmul that computes logits and assert both operands are fp32 when the flag is on, and
stay in the activation dtype when off.

Runs on GPU-less hosts (CPU matmul), but requires vLLM to be importable so we
can patch the real ``LogitsProcessor`` class.
"""

import pytest
import torch


@pytest.fixture
def patched_logits_processor_cls():
    """Apply the fp32 lm-head patch and return the patched LogitsProcessor class."""
    from skyrl.backends.skyrl_train.patches.vllm.patch_fp32_lm_head import (
        apply_fp32_lm_head_patch,
    )

    apply_fp32_lm_head_patch()

    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    return LogitsProcessor


def _make_processor(cls, enable_fp32, vocab_size):
    """Build a LogitsProcessor instance without running vLLM's __init__.

    The patched ``_get_logits`` only relies on ``_skyrl_fp32_lm_head``,
    ``_gather_logits`` and ``org_vocab_size``, so we set those directly and avoid
    constructing the full vLLM platform/config machinery on a GPU-less host.
    """
    proc = object.__new__(cls)
    proc._skyrl_fp32_lm_head = enable_fp32
    proc.org_vocab_size = vocab_size
    # No TP in tests: gather is identity.
    proc._gather_logits = lambda logits: logits
    return proc


class _LMHeadStub:
    def __init__(self, vocab, hidden, dtype):
        self.weight = torch.randn(vocab, hidden, dtype=dtype)


def _run_case(cls, enable_fp32, activations_dtype, weights_dtype, expected_dtype, monkeypatch):
    batch, hidden, vocab = 2, 64, 128
    hidden_states = torch.randn(batch, hidden, dtype=activations_dtype)
    lm_head = _LMHeadStub(vocab, hidden, weights_dtype)
    proc = _make_processor(cls, enable_fp32, vocab)

    captured = {}
    original_matmul = torch.matmul

    def probe_matmul(a, b, *args, **kwargs):
        captured.setdefault("a", a.dtype)
        captured.setdefault("b", b.dtype)
        return original_matmul(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", probe_matmul)
    logits = proc._get_logits(hidden_states, lm_head, None)

    assert captured["a"] == expected_dtype
    assert captured["b"] == expected_dtype
    if enable_fp32:
        assert logits.dtype == torch.float32
    assert logits.shape == (batch, vocab)


@pytest.mark.vllm
def test_fp32_flag_true_upcasts_bf16(patched_logits_processor_cls, monkeypatch):
    _run_case(
        patched_logits_processor_cls,
        enable_fp32=True,
        activations_dtype=torch.bfloat16,
        weights_dtype=torch.bfloat16,
        expected_dtype=torch.float32,
        monkeypatch=monkeypatch,
    )


@pytest.mark.vllm
def test_fp32_flag_false_keeps_dtype(patched_logits_processor_cls, monkeypatch):
    # Flag off: the patched _get_logits delegates to the original implementation,
    # which calls lm_head.quant_method.apply rather than torch.matmul, so we just
    # assert it does not upcast to fp32 by checking the result dtype path. Build a
    # stub lm_head with a quant_method that performs a model-dtype matmul.
    class _QuantMethod:
        def apply(self, layer, x, bias=None):
            return torch.matmul(x.to(layer.weight.dtype), layer.weight.t())

    batch, hidden, vocab = 2, 64, 128
    hidden_states = torch.randn(batch, hidden, dtype=torch.bfloat16)
    lm_head = _LMHeadStub(vocab, hidden, torch.bfloat16)
    lm_head.quant_method = _QuantMethod()
    proc = _make_processor(patched_logits_processor_cls, enable_fp32=False, vocab_size=vocab)

    logits = proc._get_logits(hidden_states, lm_head, None)
    assert logits.dtype == torch.bfloat16
    assert logits.shape == (batch, vocab)
