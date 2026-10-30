"""``patch_clip_grads_mixed_dtype``: mixed fp32/bf16 grad lists are split per dtype, keeping parallel lists aligned,
and the installed ``multi_tensor_applier`` wrapper runs the kernel once per dtype and recombines the results."""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_clip_grads_mixed_dtype import (
    split_by_dtype,
)


def test_split_by_dtype_keeps_parallel_lists_aligned():
    a = [torch.zeros(2), torch.zeros(3, dtype=torch.bfloat16), torch.zeros(4), torch.zeros(5, dtype=torch.bfloat16)]
    b = [t.clone() for t in a]
    groups = split_by_dtype([a, b])
    assert set(groups) == {torch.float32, torch.bfloat16}
    assert [t.numel() for t in groups[torch.float32][0]] == [2, 4]
    assert [t.numel() for t in groups[torch.bfloat16][1]] == [3, 5]
    assert groups[torch.float32][1][1] is b[2]


def test_split_by_dtype_uniform_list_is_one_group():
    a = [torch.zeros(2), torch.zeros(3)]
    assert list(split_by_dtype([a])) == [torch.float32]


class _FakeKernels:
    """Stands in for the multi-tensor kernels: records each call and, like them, requires one dtype per call."""

    def __init__(self):
        self.calls = []
        self.l2_norm = object()
        self.scale = object()

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        assert len({t.dtype for t in tensor_lists[0]}) == 1, "a kernel call mixed dtypes"
        self.calls.append((op, tensor_lists))
        if op is self.l2_norm:
            norm = torch.sqrt(sum(t.float().square().sum() for t in tensor_lists[0]))
            return norm, None
        if op is self.scale:
            for src, dst in zip(*tensor_lists):
                dst.copy_(src * args[0])
            return None
        raise AssertionError("unexpected op")


@pytest.fixture
def patched(monkeypatch):
    clip_grads = pytest.importorskip("megatron.core.optimizer.clip_grads")
    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_clip_grads_mixed_dtype,
    )

    kernels = _FakeKernels()
    # Restored by monkeypatch, together with the wrapper installed below.
    monkeypatch.setattr(clip_grads, "multi_tensor_applier", kernels)
    monkeypatch.setattr(clip_grads, "l2_norm_impl", kernels.l2_norm, raising=False)
    monkeypatch.setattr(patch_clip_grads_mixed_dtype, "_APPLIED", False)
    assert patch_clip_grads_mixed_dtype.apply_clip_grads_mixed_dtype_patch()
    return clip_grads, kernels, patch_clip_grads_mixed_dtype


def _mixed_grads():
    return [
        torch.tensor([3.0, 4.0]),
        torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        torch.tensor([2.0]),
        torch.tensor([2.0], dtype=torch.bfloat16),
    ]


@pytest.mark.megatron
def test_mixed_l2_norm_combines_per_dtype_norms(patched):
    clip_grads, kernels, _ = patched
    grads = _mixed_grads()
    norm, per_tensor = clip_grads.multi_tensor_applier(kernels.l2_norm, None, [grads], False)
    assert per_tensor is None
    torch.testing.assert_close(norm, torch.sqrt(torch.tensor(9.0 + 16 + 1 + 4 + 4 + 4)))
    assert len(kernels.calls) == 2


@pytest.mark.megatron
def test_mixed_scale_clips_both_dtype_groups(patched):
    clip_grads, kernels, _ = patched
    grads = _mixed_grads()
    expected = [g * 0.5 for g in grads]
    clip_grads.multi_tensor_applier(kernels.scale, None, [grads, grads], 0.5)
    for got, want in zip(grads, expected):
        torch.testing.assert_close(got, want)
    assert len(kernels.calls) == 2


@pytest.mark.megatron
def test_uniform_list_passes_through_unchanged(patched):
    clip_grads, kernels, _ = patched
    grads = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    lists = [grads]
    norm, _ = clip_grads.multi_tensor_applier(kernels.l2_norm, None, lists, False)
    torch.testing.assert_close(norm, torch.tensor(5.0))
    assert len(kernels.calls) == 1 and kernels.calls[0][1] is lists


@pytest.mark.megatron
def test_applying_twice_wraps_once(patched):
    clip_grads, _, patch = patched
    wrapper = clip_grads.multi_tensor_applier
    assert patch.apply_clip_grads_mixed_dtype_patch()
    assert clip_grads.multi_tensor_applier is wrapper
