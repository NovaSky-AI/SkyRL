"""CPU contracts for the fused-kernel boundary; real TE coverage is in the GPU lane."""

import sys
from types import ModuleType

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_grad_norm_mixed_dtype import (
    make_dtype_grouped_applier,
    patch_grad_norm_mixed_dtype,
)


@pytest.fixture
def kernel():
    calls = []
    op = object()

    def applier(operation, overflow, tensor_lists, per_tensor=False):
        grads = tensor_lists[0]
        assert operation is op
        assert len({g.dtype for g in grads}) == 1
        assert all(g.is_contiguous() for g in grads)
        calls.append((overflow, tensor_lists))
        norms = torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])
        return torch.linalg.vector_norm(norms).reshape(1), norms if per_tensor else torch.empty(0)

    return op, calls, applier, make_dtype_grouped_applier(applier, op)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_uniform_list_uses_original_tensors_and_list(kernel, dtype):
    op, calls, _, grouped = kernel
    lists = [[torch.randn(32, dtype=dtype), torch.randn(8, dtype=dtype)]]
    grouped(op, None, lists, False)
    assert len(calls) == 1
    assert calls[0][1] is lists


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_norm_preserves_values_and_groups_without_copies(kernel, reverse):
    op, calls, _, grouped = kernel
    grads = [
        torch.zeros(32, dtype=torch.float16),
        torch.tensor([100000.123], dtype=torch.float32),
        torch.tensor([1.125], dtype=torch.bfloat16),
        torch.tensor([0.00391], dtype=torch.float32),
    ]
    if reverse:
        grads.reverse()
    snapshots = [g.clone() for g in grads]
    overflow = object()
    result, _ = grouped(op, overflow, [grads], False)
    reference = torch.linalg.vector_norm(torch.cat([g.float() for g in grads])).reshape(1)
    torch.testing.assert_close(result, reference, rtol=1e-6, atol=0)
    assert torch.isfinite(result).all()
    assert len(calls) == 3
    assert all(buf is overflow for buf, _ in calls)
    assert {id(g) for _, lists in calls for g in lists[0]} == {id(g) for g in grads}
    for g, before in zip(grads, snapshots, strict=True):
        torch.testing.assert_close(g, before, rtol=0, atol=0)


def test_fp32_contribution_is_not_rounded_to_bfloat16(kernel):
    op, _, _, grouped = kernel
    grads = [torch.zeros(1024, dtype=torch.bfloat16), torch.tensor([1.003], dtype=torch.float32)]
    result, _ = grouped(op, None, [grads], False)
    torch.testing.assert_close(result, grads[1], rtol=0, atol=0)


def test_per_tensor_norms_keep_input_order(kernel):
    op, _, _, grouped = kernel
    grads = [torch.tensor([3.0], dtype=torch.bfloat16), torch.tensor([4.0]), torch.tensor([12.0], dtype=torch.bfloat16)]
    norm, each = grouped(op, None, [grads], True)
    torch.testing.assert_close(norm, torch.tensor([13.0]))
    torch.testing.assert_close(each, torch.tensor([3.0, 4.0, 12.0]))


def test_strided_input_is_copied_without_changing_values(kernel):
    op, calls, _, grouped = kernel
    grad = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    result, _ = grouped(op, None, [[grad]], False)
    torch.testing.assert_close(result, torch.linalg.vector_norm(grad).reshape(1))
    torch.testing.assert_close(calls[0][1][0][0], grad)
    assert calls[0][1][0][0].is_contiguous()


def test_other_kernels_and_empty_lists_are_passed_through():
    norm_op, scale_op, sentinel = object(), object(), object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    grouped = make_dtype_grouped_applier(original, norm_op)
    tensors = [[torch.ones(1)], [torch.ones(1)]]
    assert grouped(scale_op, None, tensors, 0.5) is sentinel
    assert calls[-1][0] == (scale_op, None, tensors, 0.5)
    assert grouped(norm_op, None, [[]], False) is sentinel


def test_patch_reaches_previously_bound_norm_function_and_is_idempotent(monkeypatch, kernel):
    op, calls, original, _ = kernel
    clip = ModuleType("megatron.core.optimizer.clip_grads")
    clip.multi_tensor_applier, clip.l2_norm_impl = original, op
    exec("def norm(grads):\n    return multi_tensor_applier(l2_norm_impl, None, [grads], False)[0]", clip.__dict__)
    already_imported_norm = clip.norm
    optimizer = ModuleType("megatron.core.optimizer")
    optimizer.clip_grads = clip
    monkeypatch.setitem(sys.modules, optimizer.__name__, optimizer)
    patch_grad_norm_mixed_dtype()
    installed = clip.multi_tensor_applier
    patch_grad_norm_mixed_dtype()
    assert clip.multi_tensor_applier is installed
    result = already_imported_norm([torch.tensor([3.0], dtype=torch.bfloat16), torch.tensor([4.0])])
    torch.testing.assert_close(result, torch.tensor([5.0]))
    assert len(calls) == 2
