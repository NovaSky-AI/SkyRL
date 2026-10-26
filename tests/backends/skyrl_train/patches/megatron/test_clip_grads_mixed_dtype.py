"""``split_by_dtype``: mixed fp32/bf16 grad lists are split per dtype, keeping parallel lists aligned."""

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
