from types import SimpleNamespace
from unittest.mock import AsyncMock

import megatron.bridge.models.conversion.peft_bridge as peft_bridge
import pytest
import safetensors.torch
import torch

import skyrl.backends.skyrl_train.workers.megatron.megatron_worker as worker_module
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


@pytest.mark.asyncio
async def test_megatron_lora_export_preserves_training_dtype(monkeypatch, tmp_path):
    exported = torch.ones((2, 2), dtype=torch.bfloat16)
    worker = worker_module.MegatronPolicyWorkerBase.__new__(worker_module.MegatronPolicyWorkerBase)
    worker.actor_module = []
    worker.bridge = SimpleNamespace(
        export_adapter_weights=lambda *args, **kwargs: iter([("layer.lora_A.weight", exported)]),
        hf_pretrained=SimpleNamespace(model_name_or_path="unused"),
    )
    worker.lora_cls = object()
    worker._logical_model_path = "base-model"

    saved_state = {}
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "_convert_moe_experts_lora_to_vllm", lambda state: state)
    monkeypatch.setattr(peft_bridge, "infer_target_modules_from_adapter_weights", lambda keys: {"layer"})
    monkeypatch.setattr(peft_bridge, "build_adapter_config_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(safetensors.torch, "save_file", lambda state, path: saved_state.update(state))

    client = RemoteInferenceClient(
        proxy_url="http://router",
        server_urls=["http://server"],
        data_parallel_size=1,
    )
    client.load_lora_adapter = AsyncMock()

    await worker._save_lora_adapters_and_sync(tmp_path, client)

    assert saved_state["base_model.model.layer.lora_A.weight"].dtype == torch.bfloat16
    client.load_lora_adapter.assert_awaited_once()
