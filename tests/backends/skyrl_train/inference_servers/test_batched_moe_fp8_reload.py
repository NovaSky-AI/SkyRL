from types import SimpleNamespace

import torch

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    _decode_serialized_name,
    _load_serialized_moe_tensor,
)
from skyrl.backends.skyrl_train.quantization import (
    SERIALIZED_BLOCKWISE_FP8,
    SERIALIZED_MXFP8,
    SERIALIZED_NVFP4,
    SERIALIZED_WEIGHT_PREFIX,
)


def _load(model, params_dict, name, weight):
    decoded_name = _decode_serialized_name(name)
    assert decoded_name is not None
    mode, _ = decoded_name
    model_type = "qwen3_5_moe_text" if mode == SERIALIZED_BLOCKWISE_FP8 else "qwen3_moe"
    return _load_serialized_moe_tensor(model, params_dict, name, decoded_name, model_type, weight)


def test_decode_serialized_name():
    checkpoint_name = "model.layers.0.mlp.experts.gate_proj.weight"
    assert _decode_serialized_name(f"__skyrl_serialized__:serialized_mxfp8:{checkpoint_name}") == (
        "serialized_mxfp8",
        checkpoint_name,
    )
    assert _decode_serialized_name(checkpoint_name) is None


def test_batched_moe_tensor_uses_one_full_expert_loader_call():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((param, loaded_weight, weight_name, shard_id, expert_id, return_success))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(3, 8, 4), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "model.layers.0.mlp.experts.w13_weight"
    loaded_weight = torch.randn(3, 4, 4)
    serialized_name = (
        f"{SERIALIZED_WEIGHT_PREFIX}{SERIALIZED_MXFP8}:model.layers.0.mlp.experts.gate_proj.weight"
    )

    loaded = _load(
        SimpleNamespace(),
        {target_name: param},
        serialized_name,
        loaded_weight,
    )

    assert loaded
    assert len(calls) == 1
    assert calls[0][1] is loaded_weight
    assert calls[0][2:] == (target_name, "w1", 0, True)


def test_batched_moe_scale_maps_to_fused_scale_parameter():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, tuple(loaded_weight.shape)))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(2, 6, 3), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "language_model.model.layers.2.mlp.experts.w13_weight_scale_inv"
    loaded_weight = torch.randn(2, 3, 3)
    mapper = SimpleNamespace(
        apply_list=lambda names: [names[0].replace("model.language_model.", "language_model.model.", 1)]
    )
    model = SimpleNamespace(hf_to_vllm_mapper=mapper)
    serialized_name = (
        f"{SERIALIZED_WEIGHT_PREFIX}{SERIALIZED_BLOCKWISE_FP8}:"
        "model.language_model.layers.2.mlp.experts.up_proj.weight_scale_inv"
    )

    assert _load(model, {target_name: param}, serialized_name, loaded_weight)
    assert calls == [(target_name, "w3", (2, 3, 3))]


def test_batched_moe_mxfp8_scale_maps_to_modelopt_scale_parameter():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, loaded_weight.dtype))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(2, 6, 3, dtype=torch.uint8), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "model.layers.2.mlp.experts.w2_weight_scale"
    loaded_weight = torch.zeros(2, 3, 3, dtype=torch.uint8)
    serialized_name = (
        f"{SERIALIZED_WEIGHT_PREFIX}{SERIALIZED_MXFP8}:"
        "model.layers.2.mlp.experts.down_proj.weight_scale"
    )

    assert _load(SimpleNamespace(), {target_name: param}, serialized_name, loaded_weight)
    assert calls == [(target_name, "w2", torch.uint8)]


def test_batched_moe_nvfp4_global_scale_maps_to_modelopt_parameter():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, expert_id, tuple(loaded_weight.shape), loaded_weight.dtype))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(3, 2, dtype=torch.float32), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "model.layers.2.mlp.experts.w13_weight_scale_2"
    loaded_weight = torch.ones(3, dtype=torch.float32)
    serialized_name = (
        f"{SERIALIZED_WEIGHT_PREFIX}{SERIALIZED_NVFP4}:"
        "model.layers.2.mlp.experts.gate_proj.weight_scale_2"
    )

    assert _load(SimpleNamespace(), {target_name: param}, serialized_name, loaded_weight)
    assert calls == [
        (target_name, "w1", 0, (), torch.float32),
        (target_name, "w1", 1, (), torch.float32),
        (target_name, "w1", 2, (), torch.float32),
    ]


def test_batched_moe_nvfp4_input_scale_maps_to_modelopt_parameter():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, expert_id, loaded_weight.item()))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(2, dtype=torch.float32), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "model.layers.2.mlp.experts.w2_input_scale"
    loaded_weight = torch.ones(2, dtype=torch.float32)
    serialized_name = (
        f"{SERIALIZED_WEIGHT_PREFIX}{SERIALIZED_NVFP4}:"
        "model.layers.2.mlp.experts.down_proj.input_scale"
    )

    assert _load(SimpleNamespace(), {target_name: param}, serialized_name, loaded_weight)
    assert calls == [
        (target_name, "w2", 0, 1.0),
        (target_name, "w2", 1, 1.0),
    ]
