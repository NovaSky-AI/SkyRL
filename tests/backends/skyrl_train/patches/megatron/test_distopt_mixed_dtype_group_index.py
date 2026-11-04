"""CPU test for patch_distopt_mixed_dtype_group_index.rebuild_model_param_group_index_map."""

from skyrl.backends.skyrl_train.patches.megatron.patch_distopt_mixed_dtype_group_index import (
    rebuild_model_param_group_index_map,
)


def test_mixed_group_indexes_fp32_params_first():
    # Group 0 mixes dtypes: megatron-core orders its main params [fp32..., bf16-derived...].
    bf16_a, bf16_b, fp32_a, fp32_b = object(), object(), object(), object()
    # Group 1 is bf16-only.
    bf16_c = object()
    index = rebuild_model_param_group_index_map(
        model_fp32_groups=[[fp32_a, fp32_b], []],
        model_float16_groups=[[bf16_a, bf16_b], [bf16_c]],
    )
    assert index[fp32_a] == (0, 0)
    assert index[fp32_b] == (0, 1)
    assert index[bf16_a] == (0, 2)
    assert index[bf16_b] == (0, 3)
    assert index[bf16_c] == (1, 0)


def test_single_dtype_groups_keep_buffer_order():
    a, b, c = object(), object(), object()
    index = rebuild_model_param_group_index_map(model_fp32_groups=[[]], model_float16_groups=[[a, b, c]])
    assert [index[p] for p in (a, b, c)] == [(0, 0), (0, 1), (0, 2)]
