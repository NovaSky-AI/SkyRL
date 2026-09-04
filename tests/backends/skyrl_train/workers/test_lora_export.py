import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.lora_export import (
    build_lora_adapter_state,
)
from skyrl.train.config import SkyRLTrainConfig


def test_preserve_export_dtype_cli_override_is_typed() -> None:
    config = SkyRLTrainConfig.from_cli_overrides(["trainer.policy.model.lora.preserve_export_dtype=true"])

    assert config.trainer.policy.model.lora.preserve_export_dtype is True


@pytest.mark.parametrize(
    ("source_dtype", "preserve_dtype", "expected_dtype"),
    [
        (torch.bfloat16, False, torch.float32),
        (torch.bfloat16, True, torch.bfloat16),
        (torch.float32, False, torch.float32),
    ],
)
def test_build_lora_adapter_state_preserves_values_and_peft_names(source_dtype, preserve_dtype, expected_dtype):
    source = torch.tensor([1.25, -2.5], dtype=source_dtype)
    adapter_state = build_lora_adapter_state(
        [("decoder.layers.0.linear.adapter.weight", source)],
        preserve_dtype=preserve_dtype,
    )

    exported = adapter_state["base_model.model.decoder.layers.0.linear.adapter.weight"]
    assert exported.dtype == expected_dtype
    assert exported.data_ptr() != source.data_ptr()
    torch.testing.assert_close(exported.float(), source.float())
