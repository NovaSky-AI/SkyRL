"""CUDA entropy values, gradients, and temporary allocation bounds."""

import pytest
import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.utils.torch_utils import chunked_entropy_from_logits

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.usefixtures("module_scoped_ray_init_fixture"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_chunked_entropy_preserves_values_inputs_and_gradients(dtype, strided, autocast, with_mask, requires_grad):
    """Chunking preserves categorical entropy, including a partial final chunk."""
    torch.manual_seed(187)
    storage = torch.randn(2, 11, 514 if strided else 257, device="cuda", dtype=dtype)
    logits = (storage[..., ::2] if strided else storage).requires_grad_()
    original = logits.detach().clone()
    mask = torch.randint(0, 2, logits.shape[:2], device="cuda") if with_mask else None
    original_mask = mask.clone() if with_mask else None

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logprobs = F.log_softmax(logits, dim=-1)
        expected = -(logprobs.exp() * logprobs).sum(-1)
        if mask is not None:
            expected = expected * mask
        expected = expected.to(logits.dtype)
        actual = chunked_entropy_from_logits(logits, requires_grad=requires_grad, attention_mask=mask, chunk_size=4)

    assert actual.dtype == logits.dtype
    assert actual.shape == logits.shape[:2]
    assert actual.requires_grad == requires_grad
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    if mask is not None:
        torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)

    if requires_grad:
        upstream = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, logits, upstream)[0]
        expected_grad = torch.autograd.grad(expected, logits, upstream)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(logits, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_chunked_entropy_bounds_temporary_cuda_memory(dtype, strided, autocast):
    """Metric-only entropy needs fewer than three full chunk-sized buffers."""
    chunk_size, vocab_size = 128, 32768
    storage = torch.randn(1, 2 * chunk_size + 1, vocab_size * (2 if strided else 1), device="cuda", dtype=dtype)
    logits = storage[..., ::2] if strided else storage
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        temporary_element_size = F.log_softmax(logits[:, :1], dim=-1).element_size()
        warmup = chunked_entropy_from_logits(logits, chunk_size=chunk_size)
        del warmup
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        entropy = chunked_entropy_from_logits(logits, chunk_size=chunk_size)
    torch.cuda.synchronize()
    temporary_peak = torch.cuda.max_memory_allocated() - allocated_before

    chunk_bytes = chunk_size * vocab_size * temporary_element_size
    assert entropy.shape == logits.shape[:2]
    # Leave half a chunk of headroom for output, reductions, and allocator rounding.
    # A separate probability-times-logprob tensor exceeds this bound.
    assert temporary_peak < 2.5 * chunk_bytes
