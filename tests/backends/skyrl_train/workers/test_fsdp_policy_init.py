from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.train.config import TrainerConfig


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    # These tests call the worker directly and do not need Ray actors.
    yield


@pytest.mark.parametrize(
    "lora_rank,quantized,trainer_bf16,inference_only,expected_bf16",
    [
        (0, False, True, False, False),
        (8, False, True, False, True),
        (8, False, False, False, False),
        (8, True, False, False, True),
        (0, False, False, True, True),
    ],
)
def test_policy_model_loading_precision(monkeypatch, lora_rank, quantized, trainer_bf16, inference_only, expected_bf16):
    cfg = TrainerConfig()
    cfg.bf16 = trainer_bf16
    cfg.gradient_checkpointing = False
    cfg.policy.inference_only_init = inference_only
    cfg.policy.model.lora.rank = lora_rank
    cfg.policy.model.bitsandbytes_4bit.enabled = quantized

    wrapped_model = SimpleNamespace(model=object())
    strategy = Mock()
    strategy.device_mesh = object()
    strategy.prepare.return_value = (
        (wrapped_model, None, None) if inference_only else (wrapped_model, object(), object())
    )
    model_loader = Mock(return_value=wrapped_model)
    monkeypatch.setattr(fsdp_worker, "FSDPStrategy", Mock(return_value=strategy))
    monkeypatch.setattr(fsdp_worker, "HFModelWrapper", model_loader)
    monkeypatch.setattr(fsdp_worker, "should_use_meta_init", Mock(return_value=False))
    monkeypatch.setattr(fsdp_worker, "build_profiler_from_policy_cfg", Mock())
    monkeypatch.setattr(
        fsdp_worker.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace(tie_word_embeddings=True))
    )
    worker = SimpleNamespace(cfg=cfg, _seq_parallel_monkey_patch=Mock(), _set_expandable_segments=Mock())

    fsdp_worker.FSDPPolicyWorkerBase.init_model(worker, "test/model")

    assert model_loader.call_args.kwargs["bf16"] is expected_bf16
    assert model_loader.call_args.kwargs["load_in_4bit"] is quantized
    if quantized:
        assert model_loader.call_args.kwargs["meta_init"] is False
