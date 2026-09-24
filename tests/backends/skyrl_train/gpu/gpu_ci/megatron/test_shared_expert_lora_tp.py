"""Two-rank regression for the Megatron-Bridge #6089 runtime backport.

Run on a GPU worker with:

    uv run --env-file .env.mc --isolated --extra dev --extra megatron \
      --with ray==2.56.0 torchrun --standalone --nproc_per_node=2 -m pytest -q -s \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_shared_expert_lora_tp.py
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection

from skyrl.backends.skyrl_train.patches.megatron.patch_shared_expert_lora_tp import (
    _ScaleForward,
    apply_shared_expert_lora_tp_patch,
)

_TP_SIZE = 2


def test_scale_forward_preserves_gradient() -> None:
    """#6089 divides the replicated forward value, not its gradient."""
    value = torch.tensor([2.0, -6.0], requires_grad=True)
    _ScaleForward.apply(value, 0.5).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


@contextmanager
def _distributed_tp() -> Iterator[ProcessGroupCollection]:
    """Initialize the minimum two-rank TP topology used by this regression."""
    import megatron.core.parallel_state as parallel_state
    import torch.distributed as dist
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if owns_process_group:
        dist.init_process_group(backend="nccl")
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=_TP_SIZE,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)

    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _make_fc2_adapter(
    pg_collection: ProcessGroupCollection,
    *,
    overlap: bool,
    sequence_parallel: bool,
    base_linear_name: str = "decoder.layers.0.mlp.shared_experts.linear_fc2",
):
    """Construct the row-parallel adapter used by shared-expert FC2."""
    from megatron.bridge.peft.utils import ParallelLinearAdapter
    from megatron.core.model_parallel_config import ModelParallelConfig

    config = ModelParallelConfig(
        tensor_model_parallel_size=_TP_SIZE,
        sequence_parallel=sequence_parallel,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )
    return ParallelLinearAdapter(
        in_features=8,
        out_features=8,
        dim=4,
        base_linear_name=base_linear_name,
        activation="identity",
        input_is_parallel=True,
        model_parallel_config=config,
        alpha=4,
        disable_tensor_parallel_comm=overlap,
        disable_sequence_parallel_comm=overlap,
        pg_collection=pg_collection,
    )


def _set_nonzero_weights(adapter) -> None:
    """Set deterministic nonzero weights on each TP shard."""
    import torch.distributed as dist

    rank = dist.get_rank()
    with torch.no_grad():
        for index, parameter in enumerate(adapter.parameters(), start=1):
            values = torch.arange(
                1,
                parameter.numel() + 1,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter) * (0.01 * index) + rank)


@pytest.mark.megatron
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_shared_expert_fc2_overlap_matches_standard_tp_forward_and_backward(sequence_parallel: bool) -> None:
    """External overlap reduction preserves standard LoRA values and gradients."""
    import torch.distributed as dist
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    apply_shared_expert_lora_tp_patch()
    with _distributed_tp() as pg_collection:
        standard = _make_fc2_adapter(pg_collection, overlap=False, sequence_parallel=sequence_parallel)
        overlap = _make_fc2_adapter(pg_collection, overlap=True, sequence_parallel=sequence_parallel)
        routed_expert = _make_fc2_adapter(
            pg_collection,
            overlap=True,
            sequence_parallel=sequence_parallel,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
        )
        assert overlap._skyrl_external_tp_reduce_scale == 1.0 / _TP_SIZE
        assert routed_expert._skyrl_external_tp_reduce_scale == 1.0
        _set_nonzero_weights(standard)
        overlap.load_state_dict(standard.state_dict())

        rank = dist.get_rank()
        values = torch.arange(1, 33, device="cuda", dtype=torch.float32).reshape(4, 2, 4)
        standard_input = (values + rank).requires_grad_(True)
        overlap_input = standard_input.detach().clone().requires_grad_(True)

        standard_output = standard(standard_input)
        overlap_output = overlap(overlap_input)
        if sequence_parallel:
            overlap_output = reduce_scatter_to_sequence_parallel_region(overlap_output, group=pg_collection.tp)
        else:
            overlap_output = reduce_from_tensor_model_parallel_region(overlap_output, group=pg_collection.tp)
        torch.testing.assert_close(overlap_output, standard_output, rtol=1e-6, atol=1e-6)

        output_grad = torch.arange(
            1,
            standard_output.numel() + 1,
            device=standard_output.device,
            dtype=standard_output.dtype,
        ).reshape_as(standard_output)
        standard_output.backward(output_grad)
        overlap_output.backward(output_grad)

        torch.testing.assert_close(overlap_input.grad, standard_input.grad, rtol=1e-6, atol=1e-6)
        assert dict(overlap.named_parameters()).keys() == dict(standard.named_parameters()).keys()
        for name, parameter in standard.named_parameters():
            overlap_parameter = dict(overlap.named_parameters())[name]
            assert parameter.grad is not None
            assert overlap_parameter.grad is not None
            torch.testing.assert_close(overlap_parameter.grad, parameter.grad, rtol=1e-6, atol=1e-6)
