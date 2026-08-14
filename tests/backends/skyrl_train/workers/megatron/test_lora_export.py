from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import safetensors.torch
import torch

pytest.importorskip("megatron")
from megatron.bridge.models.conversion import peft_bridge

import skyrl.backends.skyrl_train.workers.megatron.megatron_worker as worker_module
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


def test_all_linear_targets_hybrid_mamba_projections(monkeypatch):
    captured = {}

    class FakeLoRA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(worker_module, "LoRA", FakeLoRA)
    worker = worker_module.MegatronPolicyWorkerBase.__new__(worker_module.MegatronPolicyWorkerBase)
    worker.cfg = SimpleNamespace(bf16=True)
    config = SimpleNamespace(
        target_modules="all-linear",
        rank=32,
        alpha=32,
        dropout=0.0,
        init_method="xavier",
        exclude_modules=None,
    )

    worker.configure_lora(config)

    assert captured["target_modules"] == [
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "in_proj",
        "out_proj",
    ]


@pytest.mark.asyncio
async def test_conditional_generation_lora_export_matches_vllm_names(monkeypatch, tmp_path):
    exported = torch.ones((2, 2), dtype=torch.bfloat16)
    worker = worker_module.MegatronPolicyWorkerBase.__new__(worker_module.MegatronPolicyWorkerBase)
    worker.actor_module = []
    worker.bridge = SimpleNamespace(
        export_adapter_weights=lambda *args, **kwargs: iter(
            [("model.language_model.layers.0.self_attn.q_proj.lora_A.weight", exported)]
        ),
        hf_pretrained=SimpleNamespace(model_name_or_path="unused"),
    )
    worker.lora_cls = object()
    worker._logical_model_path = "base-model"

    saved_state = {}
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "_convert_moe_experts_lora_to_vllm", lambda state: state)
    monkeypatch.setattr(
        peft_bridge,
        "infer_target_modules_from_adapter_weights",
        lambda keys: {"q_proj"},
    )
    monkeypatch.setattr(peft_bridge, "build_adapter_config_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(safetensors.torch, "save_file", lambda state, path: saved_state.update(state))

    client = RemoteInferenceClient(
        proxy_url="http://router",
        server_urls=["http://server"],
        data_parallel_size=1,
    )
    client.load_lora_adapter = AsyncMock()

    await worker._save_lora_adapters_and_sync(tmp_path, client)

    key = "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert set(saved_state) == {key}
    assert saved_state[key].dtype == torch.bfloat16
    client.load_lora_adapter.assert_awaited_once()
