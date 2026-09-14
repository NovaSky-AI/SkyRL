from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (
    LoRABridgeSourceLayout,
    extract_lora_bridge_sources,
    reconstruct_lora_bridge_tensors,
    validate_lora_bridge_source_layout,
)


def _record(**overrides):
    fields = {
        "global_param_name": "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight",
        "hf_param_names": ("base_model.model.layers.0.mlp.gate_proj.lora_A.weight",),
        "component": "linear_in",
        "transform": "identity",
        "weight": torch.ones((2, 4), dtype=torch.float32),
        "tensor_parallel_axis": 1,
        "tensor_parallel_rank": 0,
        "tensor_parallel_size": 2,
        "expert_parallel_axis": None,
        "expert_parallel_rank": 0,
        "expert_parallel_size": 1,
        "transform_config": (),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_extract_lora_bridge_sources_separates_fp32_storage_from_stable_metadata():
    tensors, sources = extract_lora_bridge_sources([_record()])

    assert tensors.keys() == {sources[0].key}
    assert tensors[sources[0].key].dtype is torch.float32
    assert sources[0].shape == (2, 4)
    assert sources[0].hf_param_names == ("base_model.model.layers.0.mlp.gate_proj.lora_A.weight",)
    assert validate_lora_bridge_source_layout(sources)[(sources[0].key, 0, 0)] == sources[0]


def test_extract_lora_bridge_sources_rejects_duplicate_or_non_fp32_sources():
    record = _record()
    with pytest.raises(ValueError, match="duplicate"):
        extract_lora_bridge_sources([record, record])
    with pytest.raises(ValueError, match="float32"):
        extract_lora_bridge_sources([_record(weight=torch.ones((2, 4), dtype=torch.bfloat16))])


def test_reconstruct_lora_bridge_tensors_assembles_tp_and_ep_shards():
    source = _record(
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
        hf_param_names=("base_model.model.layers.0.mlp.down_proj.lora_B.weight",),
        component="linear_out",
        tensor_parallel_axis=1,
        tensor_parallel_size=2,
        expert_parallel_axis=0,
        expert_parallel_size=2,
        weight=torch.ones((1, 2), dtype=torch.float32),
    )
    records = []
    tensors = {}
    for ep_rank in range(2):
        for tp_rank in range(2):
            record = _record(
                **{
                    **source.__dict__,
                    "tensor_parallel_rank": tp_rank,
                    "expert_parallel_rank": ep_rank,
                }
            )
            _, sources = extract_lora_bridge_sources([record])
            records.extend(sources)
            tensors[(sources[0].key, tp_rank, ep_rank)] = torch.full(
                (1, 2), ep_rank * 10 + tp_rank, dtype=torch.float32
            )

    result = reconstruct_lora_bridge_tensors(records, tensors)

    assert torch.equal(
        result["base_model.model.layers.0.mlp.down_proj.lora_B.weight"],
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]]),
    )


def test_reconstruct_lora_bridge_tensors_replicates_and_splits_gated_sources():
    replicated = _record(
        hf_param_names=("q.lora_A.weight", "k.lora_A.weight", "v.lora_A.weight"),
        transform="replicate",
        tensor_parallel_size=1,
    )
    gated = _record(
        global_param_name="decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight",
        hf_param_names=("gate.lora_B.weight", "up.lora_B.weight"),
        component="linear_out",
        transform="split_gated_mlp",
        weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        tensor_parallel_size=1,
    )
    tensors_a, sources_a = extract_lora_bridge_sources([replicated])
    tensors_b, sources_b = extract_lora_bridge_sources([gated])
    tensors = {
        (sources_a[0].key, 0, 0): tensors_a[sources_a[0].key],
        (sources_b[0].key, 0, 0): tensors_b[sources_b[0].key],
    }

    result = reconstruct_lora_bridge_tensors((*sources_a, *sources_b), tensors)

    assert result["q.lora_A.weight"] is result["k.lora_A.weight"]
    assert torch.equal(result["gate.lora_B.weight"], torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert torch.equal(result["up.lora_B.weight"], torch.tensor([[4.0, 5.0], [6.0, 7.0]]))


def test_reconstruct_lora_bridge_tensors_splits_qkv_with_bridge_layout_config():
    source_tensor = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    qkv = _record(
        global_param_name="decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        hf_param_names=("q.lora_B.weight", "k.lora_B.weight", "v.lora_B.weight"),
        component="linear_out",
        transform="split_qkv",
        weight=source_tensor,
        tensor_parallel_size=1,
        transform_config=(
            ("num_attention_heads", 4),
            ("num_query_groups", 2),
            ("kv_channels", 2),
            ("hidden_size", 8),
            ("attention_output_gate", False),
        ),
    )
    tensors, sources = extract_lora_bridge_sources([qkv])

    result = reconstruct_lora_bridge_tensors(
        sources,
        {(sources[0].key, 0, 0): tensors[sources[0].key]},
    )

    assert torch.equal(
        result["q.lora_B.weight"],
        torch.cat([source_tensor[:4], source_tensor[8:12]], dim=0),
    )
    assert torch.equal(
        result["k.lora_B.weight"],
        torch.cat([source_tensor[4:6], source_tensor[12:14]], dim=0),
    )
    assert torch.equal(
        result["v.lora_B.weight"],
        torch.cat([source_tensor[6:8], source_tensor[14:16]], dim=0),
    )


def test_reconstruct_lora_bridge_tensors_splits_gdn_with_bridge_layout_config():
    source = _record(
        global_param_name="decoder.layers.0.linear_attn.in_proj.adapter.linear_out.weight",
        hf_param_names=(
            "qkv.lora_B.weight",
            "z.lora_B.weight",
            "b.lora_B.weight",
            "a.lora_B.weight",
        ),
        component="linear_out",
        transform="split_gdn_in_proj",
        weight=torch.ones((6, 2), dtype=torch.float32),
        tensor_parallel_axis=0,
        tensor_parallel_size=2,
        transform_config=(
            ("linear_key_head_dim", 1),
            ("linear_value_head_dim", 1),
            ("linear_num_key_heads", 2),
            ("linear_num_value_heads", 2),
        ),
    )
    records = []
    tensors = {}
    for tp_rank in range(2):
        record = _record(**{**source.__dict__, "tensor_parallel_rank": tp_rank})
        _, sources = extract_lora_bridge_sources([record])
        records.extend(sources)
        tensors[(sources[0].key, tp_rank, 0)] = torch.arange(
            tp_rank * 12, (tp_rank + 1) * 12, dtype=torch.float32
        ).reshape(6, 2)

    result = reconstruct_lora_bridge_tensors(records, tensors)

    assert torch.equal(
        result["qkv.lora_B.weight"],
        torch.tensor(
            [
                [0.0, 1.0],
                [12.0, 13.0],
                [2.0, 3.0],
                [14.0, 15.0],
                [4.0, 5.0],
                [16.0, 17.0],
            ]
        ),
    )
    assert torch.equal(result["z.lora_B.weight"], torch.tensor([[6.0, 7.0], [18.0, 19.0]]))
    assert torch.equal(result["b.lora_B.weight"], torch.tensor([[8.0, 9.0], [20.0, 21.0]]))
    assert torch.equal(result["a.lora_B.weight"], torch.tensor([[10.0, 11.0], [22.0, 23.0]]))


def test_bridge_source_layout_digest_tracks_rank_ownership_and_rejects_duplicates():
    _, first_sources = extract_lora_bridge_sources([_record()], source_rank=0)
    _, second_sources = extract_lora_bridge_sources([_record(tensor_parallel_rank=1)], source_rank=1)
    layout = LoRABridgeSourceLayout("adapter", (*first_sources, *second_sources))

    _, remapped_sources = extract_lora_bridge_sources([_record(tensor_parallel_rank=1)], source_rank=2)
    assert layout.layout_digest != LoRABridgeSourceLayout("adapter", (*first_sources, *remapped_sources)).layout_digest
    with pytest.raises(ValueError, match="ownership"):
        LoRABridgeSourceLayout(
            "adapter",
            (*first_sources, *first_sources),
        )
    with pytest.raises(ValueError, match="missing shard"):
        LoRABridgeSourceLayout("adapter", first_sources)


def test_bridge_source_layout_round_trips_over_the_control_plane_and_verifies_digest():
    _, first_sources = extract_lora_bridge_sources([_record()], source_rank=0)
    _, second_sources = extract_lora_bridge_sources([_record(tensor_parallel_rank=1)], source_rank=1)
    layout = LoRABridgeSourceLayout("adapter", (*first_sources, *second_sources))
    payload = layout.to_json_dict()

    assert LoRABridgeSourceLayout.from_json_dict(payload) == layout

    payload["sources"][0]["source_rank"] = 7
    with pytest.raises(ValueError, match="digest"):
        LoRABridgeSourceLayout.from_json_dict(payload)
