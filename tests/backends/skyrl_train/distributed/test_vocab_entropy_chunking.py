import importlib
import sys
from types import ModuleType

import pytest
import torch


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


@pytest.mark.parametrize("temperature", [0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("padding", [0, 2])
def test_vocab_entropy_masked_logits_match_cropped_vocabulary(monkeypatch, temperature, dtype, padding):
    monkeypatch.setattr(model_utils.mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(model_utils.dist, "all_reduce", lambda tensor, **kwargs: None)
    logits = torch.tensor([[0.2, -1.0, 2.0], [0.7, 0.3, -0.5]], dtype=dtype)
    padded = torch.cat((logits, torch.full((2, padding), -torch.inf, dtype=dtype)), dim=-1).requires_grad_()
    reference = logits.clone().requires_grad_()
    scaled = padded / temperature
    before = scaled.detach().clone()

    actual = model_utils._VocabParallelEntropy.apply(scaled)
    expected = _local_entropy(reference / temperature)
    actual.sum().backward()
    expected.sum().backward()

    assert torch.isfinite(actual).all()
    assert torch.isfinite(padded.grad).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(padded.grad[:, :3], reference.grad)
    assert torch.count_nonzero(padded.grad[:, 3:]) == 0
    assert torch.count_nonzero(scaled.softmax(-1)[:, 3:]) == 0
    torch.testing.assert_close(scaled, before)


@pytest.mark.parametrize("values", [[0.0, torch.nan], [0.0, torch.inf], [-torch.inf, -torch.inf]])
def test_vocab_entropy_preserves_invalid_distribution(monkeypatch, values):
    monkeypatch.setattr(model_utils.mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(model_utils.dist, "all_reduce", lambda tensor, **kwargs: None)
    entropy = model_utils._VocabParallelEntropy.apply(torch.tensor([values]))
    assert torch.isnan(entropy).all()


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
