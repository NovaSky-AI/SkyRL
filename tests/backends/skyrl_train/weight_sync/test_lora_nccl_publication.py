from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_nccl import (
    LoRANcclPublicationPlanner,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (
    extract_lora_bridge_sources,
)


def _record(tp_rank, values=None):
    return SimpleNamespace(
        global_param_name="model.proj.adapter.linear_out.weight",
        weight=torch.tensor(
            values or [[1.0, 2.0]],
            dtype=torch.float32,
        ),
        hf_param_names=("model.proj.lora_B.weight",),
        component="linear_out",
        transform="identity",
        tensor_parallel_axis=0,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=2,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=(),
    )


def _global_sources():
    return [
        extract_lora_bridge_sources([_record(0)], 0)[1],
        extract_lora_bridge_sources([_record(1)], 1)[1],
    ]


def test_publication_planner_freezes_layout_and_advances_each_attempt():
    planner = LoRANcclPublicationPlanner("adapter", 0)

    first = planner.plan([_record(0)], _global_sources())
    second = planner.plan([_record(0, [[3.0, 4.0]])])

    assert first.request.generation == 0
    assert second.request.generation == 1
    assert first.request.layout_digest == second.request.layout_digest
    assert first.local_tensors["model.proj.adapter.linear_out.weight"].tolist() == [
        [1.0, 2.0]
    ]
    assert second.local_tensors["model.proj.adapter.linear_out.weight"].tolist() == [
        [3.0, 4.0]
    ]


def test_publication_planner_rejects_changed_or_reinitialized_layout():
    planner = LoRANcclPublicationPlanner("adapter", 0)
    planner.plan([_record(0)], _global_sources())

    with pytest.raises(ValueError, match="changed its fixed local source layout"):
        planner.plan([_record(0, [[1.0, 2.0], [3.0, 4.0]])])
    with pytest.raises(ValueError, match="already initialized"):
        planner.plan([_record(0)], _global_sources())
