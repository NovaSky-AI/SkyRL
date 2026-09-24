"""Two-rank regression for the Megatron-Bridge #6089 runtime backport.

Run with:

    uv run --isolated --extra dev --extra megatron pytest -q -s \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_shared_expert_lora_tp.py
"""

from __future__ import annotations

import pytest
import ray
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.patches.megatron.patch_shared_expert_lora_tp import (
    _ScaleForward,
    apply_shared_expert_lora_tp_patch,
)
from skyrl.train.utils import get_ray_pg_ready_with_timeout

_TP_SIZE = 2
pytestmark = pytest.mark.megatron


@ray.remote(num_gpus=1)
class _SharedExpertLoRATpWorker:
    def endpoint(self) -> tuple[str, int]:
        import socket

        from ray.util import get_node_ip_address

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return get_node_ip_address(), sock.getsockname()[1]

    def run(self, rank: int, master_addr: str, master_port: int, sequence_parallel: bool) -> None:
        import megatron.core.parallel_state as parallel_state
        import torch.distributed as dist
        from megatron.core.tensor_parallel.mappings import (
            reduce_from_tensor_model_parallel_region,
            reduce_scatter_to_sequence_parallel_region,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=_TP_SIZE,
        )
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=_TP_SIZE,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
            )
            model_parallel_cuda_manual_seed(2026, force_reset_rng=True)
            apply_shared_expert_lora_tp_patch()

            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
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
        finally:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def test_scale_forward_preserves_gradient() -> None:
    """#6089 divides the replicated forward value, not its gradient."""
    value = torch.tensor([2.0, -6.0], requires_grad=True)
    _ScaleForward.apply(value, 0.5).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


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


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_shared_expert_fc2_overlap_matches_standard_tp_forward_and_backward(
    ray_init_fixture, sequence_parallel: bool
) -> None:
    """External overlap reduction preserves standard LoRA values and gradients."""
    pg = placement_group([{"GPU": _TP_SIZE, "CPU": _TP_SIZE}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    workers = [
        _SharedExpertLoRATpWorker.options(scheduling_strategy=scheduling_strategy).remote() for _ in range(_TP_SIZE)
    ]
    master_addr, master_port = ray.get(workers[0].endpoint.remote())
    ray.get(
        [worker.run.remote(rank, master_addr, master_port, sequence_parallel) for rank, worker in enumerate(workers)]
    )
