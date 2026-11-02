"""patch_grouped_expert_lora_fast_path: copy-free per-expert grouped LoRA forward matches the original.

Run with:

    uv run --isolated --extra dev --extra megatron pytest -q -s \
      tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_grouped_expert_lora_fast_path.py
"""

from __future__ import annotations

import os

import pytest
import torch

from skyrl.train.utils.utils import get_free_port

pytestmark = pytest.mark.megatron


@pytest.fixture(scope="module")
def model_parallel_state():
    """Single-rank model-parallel state (TP=EP=ETP=1) for building expert adapters."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    yield
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_adapter(num_experts: int, in_features: int, out_features: int, dim: int, input_is_parallel: bool):
    from megatron.bridge.peft.utils import GroupedExpertLinearAdapter
    from megatron.core.model_parallel_config import ModelParallelConfig

    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    adapter = GroupedExpertLinearAdapter(
        in_features,
        out_features,
        dim,
        num_local_experts=num_experts,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
        activation="identity",
        column_init_method="xavier",
        row_init_method="xavier",  # non-zero B so the output and both grads are informative
        input_is_parallel=input_is_parallel,
        model_parallel_config=config,
        alpha=2 * dim,
        params_device=torch.device("cuda"),
        params_dtype=torch.bfloat16,
    )
    return adapter


def _run(adapter, x, splits):
    x = x.detach().clone().requires_grad_(True)
    adapter.zero_grad(set_to_none=True)
    out = adapter(x, splits)
    out.backward(torch.linspace(-1, 1, out.numel(), device=out.device).reshape(out.shape).to(out.dtype))
    return (
        out.detach(),
        x.grad.detach(),
        adapter.linear_in.weight.grad.detach(),
        adapter.linear_out.weight.grad.detach(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("input_is_parallel", [False, True])
@pytest.mark.parametrize(
    "splits",
    [
        [5, 3, 8, 4, 2, 7, 1, 6],  # every expert active
        [0, 9, 0, 0, 11, 0, 3, 13],  # zero-token experts (empty groups)
        [36, 0, 0, 0, 0, 0, 0, 0],  # a single active expert
    ],
)
def test_fast_path_matches_original_forward(model_parallel_state, input_is_parallel, splits):
    from megatron.bridge.peft import utils

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_grouped_expert_lora_fast_path as patch,
    )

    original_forward = patch._ORIGINAL_FORWARD or utils.GroupedExpertLinearAdapter.forward
    torch.manual_seed(0)
    num_experts, in_features, out_features, dim = 8, 64, 48, 16
    adapter = _make_adapter(num_experts, in_features, out_features, dim, input_is_parallel)
    x = torch.randn(sum(splits), in_features, device="cuda", dtype=torch.bfloat16)

    patch.apply_grouped_expert_lora_fast_path_patch()
    assert utils.GroupedExpertLinearAdapter.forward is not original_forward
    fast = _run(adapter, x, splits)

    utils.GroupedExpertLinearAdapter.forward = original_forward
    try:
        reference = _run(adapter, x, splits)
    finally:
        utils.GroupedExpertLinearAdapter.forward = patch._PATCHED_FORWARD

    for got, want in zip(fast, reference):
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
    # Inactive experts' adapter weights receive an exact zero gradient on both paths.
    for expert_idx, split in enumerate(splits):
        if split == 0:
            assert fast[2][expert_idx].abs().max() == 0
            assert fast[3][expert_idx].abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fast_path_empty_input_falls_back(model_parallel_state):
    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_grouped_expert_lora_fast_path as patch,
    )

    patch.apply_grouped_expert_lora_fast_path_patch()
    adapter = _make_adapter(4, 32, 16, 8, input_is_parallel=False)
    x = torch.zeros(0, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = adapter(x, [0, 0, 0, 0])
    assert out.shape == (0, 16)
    out.sum().backward()  # the original zero-token path keeps the adapter in the graph
    assert adapter.linear_in.weight.grad is not None
