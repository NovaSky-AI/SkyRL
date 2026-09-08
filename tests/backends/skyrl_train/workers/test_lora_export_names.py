"""LoRA exports must name the modules used by the inference model."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

# Collection imports this module before marker selection, including in CPU-only CI.
pytest.importorskip("vllm", reason="LoRA export tests use vLLM's adapter-name parser and model mapper")
pytestmark = pytest.mark.vllm

from vllm.lora.utils import parse_fine_tuned_lora_name  # noqa: E402
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402

from skyrl.backends.skyrl_train.distributed import fsdp_utils  # noqa: E402
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (  # noqa: E402
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase  # noqa: E402
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    # This test invokes the export method directly; no Ray actors are needed.
    yield


@pytest.mark.parametrize(
    "is_multimodal_lm_only,source_module,expected_inference_module",
    [
        (True, "model.layers.0.self_attn.q_proj", "language_model.model.layers.0.self_attn.q_proj"),
        (False, "model.language_model.layers.0.self_attn.q_proj", "language_model.model.layers.0.self_attn.q_proj"),
        (False, "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"),
        (True, "model.layers.0.linear_attn.in_proj_a", "language_model.model.layers.0.linear_attn.in_proj_a"),
        (True, "lm_head", "language_model.lm_head"),
    ],
    ids=["language-only-vlm", "full-vlm", "text-model", "hybrid-attention", "output-head"],
)
def test_lora_export_targets_inference_modules(
    tmp_path, monkeypatch, is_multimodal_lm_only, source_module, expected_inference_module
):
    peft_wrapper_prefix = "base_model.model."
    # A text-only backbone lacks the enclosing VLM's language_model namespace.
    source_params = {
        peft_wrapper_prefix + source_module + ".lora_A.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        peft_wrapper_prefix + source_module + ".lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }
    peft_model = SimpleNamespace(
        peft_config={
            "default": LoraConfig(r=2, target_modules=[source_module.rsplit(".", 1)[-1]], task_type=TaskType.CAUSAL_LM)
        }
    )
    worker = SimpleNamespace(model=SimpleNamespace(model=peft_model), _is_multimodal_lm_only=is_multimodal_lm_only)
    monkeypatch.setattr(fsdp_utils, "collect_lora_params", lambda module: source_params)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = Mock(spec=RemoteInferenceClient)
    exported = {}

    async def load_adapter(name, path):
        # Inspect the actual on-disk payload at the inference handoff.
        exported.update(load_file(str(Path(path) / "adapter_model.safetensors")))

    client.load_lora_adapter = AsyncMock(side_effect=load_adapter)
    asyncio.run(
        FSDPPolicyWorkerBase._save_lora_adapters_and_sync(worker, peft_model, str(tmp_path), client, "test-adapter")
    )
    client.load_lora_adapter.assert_awaited_once_with("test-adapter", str(tmp_path))
    mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
    inference_modules = {parse_fine_tuned_lora_name(key, mapper)[0] for key in exported}
    assert inference_modules == {expected_inference_module}
    export_module = f"language_model.{source_module}" if is_multimodal_lm_only else source_module
    expected_params = {
        f"{peft_wrapper_prefix}{export_module}.{adapter}.weight": source_params[
            f"{peft_wrapper_prefix}{source_module}.{adapter}.weight"
        ]
        for adapter in ("lora_A", "lora_B")
    }
    assert exported.keys() == expected_params.keys()
    for name, tensor in expected_params.items():
        assert torch.equal(exported[name], tensor)
        assert exported[name].dtype == tensor.dtype


@pytest.mark.parametrize("is_multimodal_lm_only,expected_prefix", [(True, "language_model."), (False, "")])
def test_full_weight_sync_uses_inference_namespace(monkeypatch, is_multimodal_lm_only, expected_prefix):
    source_name = "model.layers.0.self_attn.q_proj.weight"
    source_tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    model = SimpleNamespace(state_dict=lambda: {source_name: source_tensor})
    # Exercise the real setup method without initializing distributed workers.
    worker = object.__new__(FSDPPolicyWorkerBase)
    worker.model = SimpleNamespace(model=model)
    worker.cfg = SimpleNamespace(placement=SimpleNamespace(colocate_all=False))
    worker._is_multimodal_lm_only = is_multimodal_lm_only
    engine_config = SimpleNamespace(weight_sync_backend="nccl")
    from skyrl.backends.skyrl_train import weight_sync

    monkeypatch.setattr(weight_sync, "get_transfer_strategy_cls", lambda **kwargs: object)
    parent_init = AsyncMock()
    monkeypatch.setattr(PolicyWorkerBase, "init_weight_sync_state", parent_init)

    asyncio.run(worker.init_weight_sync_state(None, engine_config))
    parent_init.assert_awaited_once_with(None, engine_config)
    extractor = worker.weight_extractor
    monkeypatch.setattr(extractor, "_gather_tensor", lambda tensor: tensor)
    chunks = list(extractor.extract_weights(torch.float32))
    assert len(chunks) == 1
    assert chunks[0].names == [expected_prefix + source_name]
    assert torch.equal(chunks[0].tensors[0], source_tensor)
    assert extractor.get_weight_metadata(torch.float32)["names"] == chunks[0].names


@pytest.mark.parametrize("unrecognized_prefix", ["", "base_model", "base_model.model_extra."])
@pytest.mark.parametrize("is_multimodal_lm_only", [True, False])
def test_lora_export_preserves_unrecognized_prefix(
    tmp_path, monkeypatch, unrecognized_prefix, is_multimodal_lm_only
):
    original_name = f"{unrecognized_prefix}model.layers.0.self_attn.q_proj.lora_B.weight"
    recognized_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    source_params = {
        recognized_name: torch.zeros(2, 4),
        original_name: torch.ones(4, 2),
    }
    peft_model = SimpleNamespace(
        peft_config={"default": LoraConfig(r=2, target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM)}
    )
    worker = SimpleNamespace(model=SimpleNamespace(model=peft_model), _is_multimodal_lm_only=is_multimodal_lm_only)
    monkeypatch.setattr(fsdp_utils, "collect_lora_params", lambda module: source_params)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = Mock(spec=RemoteInferenceClient)
    client.load_lora_adapter = AsyncMock()
    export_dir = tmp_path / "adapter"

    asyncio.run(
        FSDPPolicyWorkerBase._save_lora_adapters_and_sync(worker, peft_model, str(export_dir), client, "test-adapter")
    )
    exported = load_file(str(export_dir / "adapter_model.safetensors"))
    expected_name = (
        "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        if is_multimodal_lm_only
        else recognized_name
    )
    assert exported.keys() == {expected_name, original_name}
    assert torch.equal(exported[original_name], source_params[original_name])
    assert torch.equal(exported[expected_name], source_params[recognized_name])
    client.load_lora_adapter.assert_awaited_once_with("test-adapter", str(export_dir))
