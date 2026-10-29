"""CPU test: the PEFT-format adapter written into FSDP checkpoints holds every LoRA tensor.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/distributed/test_layered_summon_lora.py
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch import nn

from skyrl.backends.skyrl_train.distributed.fsdp_utils import layered_summon_lora_params


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()


class _Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


class _TextLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Text()


class _Nested(nn.Module):
    """language_model.model.layers.N (VLM-style nesting, as ML4) or model.layers.N."""

    def __init__(self, vlm: bool):
        super().__init__()
        if vlm:
            self.language_model = _TextLM()
        else:
            self.model = _Text()


@pytest.mark.parametrize("vlm", [True, False], ids=["language_model-nesting", "model-nesting"])
def test_layered_summon_matches_peft_state_dict(vlm):
    torch.manual_seed(0)
    model = get_peft_model(_Nested(vlm), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    for n, p in model.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p)
    expected = get_peft_model_state_dict(model)
    got = layered_summon_lora_params(model)
    assert len(expected) == 4
    assert set(got) == set(expected)
    for k in expected:
        assert torch.equal(got[k], expected[k])
