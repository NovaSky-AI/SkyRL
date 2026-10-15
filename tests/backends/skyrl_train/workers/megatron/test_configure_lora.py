"""Tests that ``MegatronWorker.configure_lora`` forwards ``MegatronLoraConfig`` fields to Megatron-Bridge's
``LoRA`` transform.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_configure_lora.py
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.bridge", reason="requires the megatron extra")

from megatron.bridge.peft.lora import LoRA  # noqa: E402

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (  # noqa: E402
    MegatronWorker,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig  # noqa: E402

pytestmark = pytest.mark.megatron


@pytest.mark.parametrize("regather", [False, True])
def test_configure_lora_forwards_sequence_parallel_input_regather(regather: bool) -> None:
    trainer_cfg = SkyRLTrainConfig().trainer
    trainer_cfg.policy.megatron_config.lora_config.sequence_parallel_input_regather = regather
    worker = SimpleNamespace(cfg=trainer_cfg, provider=None, bridge=None)

    lora_config = SkyRLLoraConfig(rank=8, alpha=16, target_modules=["linear_qkv", "linear_fc1"])
    MegatronWorker.configure_lora(worker, lora_config, lora_type="lora")

    assert isinstance(worker.lora_cls, LoRA)
    assert worker.lora_cls.sequence_parallel_input_regather is regather
