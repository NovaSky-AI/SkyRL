import json

from peft import LoraConfig, TaskType

from skyrl.backends.skyrl_train.distributed import fsdp_utils


def test_serialize_peft_config_handles_sets_and_enums():
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_a_proj"],
        exclude_modules=["indexer"],
        task_type=TaskType.CAUSAL_LM,
    )

    serialized = fsdp_utils.serialize_peft_config(config)

    assert serialized["target_modules"] == ["q_a_proj"]
    assert serialized["exclude_modules"] == ["indexer"]
    assert serialized["task_type"] == "CAUSAL_LM"
    assert serialized["peft_type"] == "LORA"
    json.dumps(serialized)
