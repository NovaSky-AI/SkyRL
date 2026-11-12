import importlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from skyrl.train.fused_lm_head import FusedLmHeadBackend


def _import_model_utils_without_megatron_extensions():
    """Import the pure tensor helpers without loading optional TE binaries."""
    module_names = ("megatron", "megatron.core", "megatron.core.parallel_state")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.__path__ = []
    parallel_state = ModuleType("megatron.core.parallel_state")
    try:
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = core
        sys.modules["megatron.core.parallel_state"] = parallel_state
        return importlib.import_module("skyrl.backends.skyrl_train.distributed.megatron.model_utils")
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


model_utils = _import_model_utils_without_megatron_extensions()


def _local_entropy(logits):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def test_vocab_entropy_chunking_matches_unchunked_output_and_gradient(monkeypatch):
    monkeypatch.setattr(model_utils._VocabParallelEntropy, "apply", _local_entropy)
    torch.manual_seed(3)
    unchunked_logits = torch.randn(2, 11, 17, dtype=torch.float64, requires_grad=True)
    chunked_logits = unchunked_logits.detach().clone().requires_grad_(True)

    unchunked = model_utils.vocab_parallel_entropy(unchunked_logits, chunk_size=None)
    chunked = model_utils.vocab_parallel_entropy(chunked_logits, chunk_size=3)
    unchunked.sum().backward()
    chunked.sum().backward()

    torch.testing.assert_close(chunked, unchunked)
    torch.testing.assert_close(chunked_logits.grad, unchunked_logits.grad)


def test_vocab_entropy_weighted_sum_chunking_matches_unchunked(monkeypatch):
    monkeypatch.setattr(model_utils._VocabParallelEntropy, "apply", _local_entropy)
    logits = torch.randn(1, 9, 13, dtype=torch.float64)
    weights = torch.tensor([1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.0, 0.25, 0.0], dtype=torch.float64)

    expected = model_utils.vocab_parallel_entropy_weighted_sum(logits, weights, chunk_size=None)
    actual = model_utils.vocab_parallel_entropy_weighted_sum(logits, weights, chunk_size=2)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("logits_shape", [(5, 7), (2, 5, 7)])
def test_vocab_entropy_weighted_sum_supports_2d_and_batched_logits(monkeypatch, logits_shape):
    monkeypatch.setattr(model_utils._VocabParallelEntropy, "apply", _local_entropy)
    logits = torch.randn(*logits_shape, dtype=torch.float64)
    weights = torch.linspace(0.25, 1.25, logits_shape[-2], dtype=torch.float64)
    expected = (_local_entropy(logits) * weights).sum()

    for chunk_size in (None, 1, 3):
        actual = model_utils.vocab_parallel_entropy_weighted_sum(logits, weights, chunk_size=chunk_size)
        torch.testing.assert_close(actual, expected)


def test_vocab_entropy_weighted_sum_all_masked_keeps_zero_gradient(monkeypatch):
    monkeypatch.setattr(model_utils._VocabParallelEntropy, "apply", _local_entropy)
    logits = torch.randn(1, 7, 11, dtype=torch.float64, requires_grad=True)
    weights = torch.zeros(7, dtype=torch.float64)

    result = model_utils.vocab_parallel_entropy_weighted_sum(logits, weights, chunk_size=2)
    result.backward()

    assert result.item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_vocab_entropy_auto_chunk_respects_memory_budget():
    logits = torch.empty(1, 10, 65536, dtype=torch.bfloat16)

    assert model_utils._resolve_vocab_entropy_chunk_size(logits, 0, 1) == 2


def test_vocab_entropy_auto_chunk_accounts_for_leading_dimensions():
    logits = torch.empty(2, 2, 10, 65536, dtype=torch.bfloat16)

    assert model_utils._resolve_vocab_entropy_chunk_size(logits, 0, 4) == 2


@pytest.mark.parametrize(("chunk_size", "memory_mb"), [(-1, 1), (0, 0)])
def test_vocab_entropy_chunk_resolver_rejects_invalid_values(chunk_size, memory_mb):
    with pytest.raises(ValueError):
        model_utils._resolve_vocab_entropy_chunk_size(torch.empty(1, 4, 8), chunk_size, memory_mb)


@pytest.mark.parametrize("backend", [FusedLmHeadBackend.TRITON, FusedLmHeadBackend.TRITON_BLOCK_SPARSE])
def test_fused_lm_head_dispatch_propagates_triton_errors(monkeypatch, backend):
    module_name = "skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton"
    args = (torch.randn(1, 2, 3), torch.randn(5, 3), torch.tensor([[1, 2]]), 0, 5, 2, None, False)
    monkeypatch.setitem(sys.modules, module_name, None)
    with pytest.raises(ModuleNotFoundError, match="fused_linear_logprob_triton"):
        model_utils._fused_lm_head_logprob_apply(backend, *args)

    module = ModuleType(module_name)
    module.FusedLinearLogprobTriton = Mock()
    module.FusedLinearLogprobTriton.apply.side_effect = RuntimeError("unsupported target")
    monkeypatch.setitem(sys.modules, module_name, module)
    with pytest.raises(RuntimeError, match="unsupported target"):
        model_utils._fused_lm_head_logprob_apply(backend, *args)


def test_torch_lm_head_dispatch_does_not_import_triton(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton", None
    )
    expected = torch.randn(1, 2)
    apply = Mock(return_value=expected)
    monkeypatch.setattr(model_utils.FusedLinearChunkedDistributedLogprob, "apply", apply)
    result = model_utils._fused_lm_head_logprob_apply(
        FusedLmHeadBackend.TORCH,
        torch.randn(1, 2, 3),
        torch.randn(5, 3),
        torch.tensor([[1, 2]]),
        0,
        5,
        2,
        None,
        False,
    )
    assert result is expected
    apply.assert_called_once()
