import json

import pytest
from peft import LoraConfig, TaskType

from skyrl.backends.skyrl_train.distributed import fsdp_utils


@pytest.mark.parametrize("target_modules", [["q_a_proj"], "all-linear"])
def test_serialize_peft_config_round_trip(tmp_path, target_modules):
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=target_modules,
        exclude_modules=["indexer"],
        task_type=TaskType.CAUSAL_LM,
    )

    serialized = fsdp_utils.serialize_peft_config(config)

    assert "runtime_config" not in serialized
    (tmp_path / "adapter_config.json").write_text(json.dumps(serialized))
    restored = LoraConfig.from_pretrained(tmp_path)
    assert restored.target_modules == config.target_modules
    assert restored.exclude_modules == config.exclude_modules
    assert restored.task_type == config.task_type
    assert restored.peft_type == config.peft_type
