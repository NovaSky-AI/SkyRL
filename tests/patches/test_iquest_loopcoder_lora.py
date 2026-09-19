import sys
from types import ModuleType
from typing import ClassVar

import pytest

from skyrl.backends.skyrl_train.patches.vllm_iquest_loopcoder_lora import (
    apply_iquest_loopcoder_lora_patch,
)


def test_iquest_loopcoder_lora_patch(monkeypatch):
    class FakeLoopCoder:
        pass

    module = ModuleType("vllm.model_executor.models.iquest_loopcoder")
    module.IQuestLoopCoderForCausalLM = FakeLoopCoder
    monkeypatch.setitem(sys.modules, module.__name__, module)

    apply_iquest_loopcoder_lora_patch()

    assert FakeLoopCoder.supports_lora
    assert FakeLoopCoder.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    assert FakeLoopCoder.embedding_modules == {}
    assert FakeLoopCoder.lora_skip_prefixes == []
    assert FakeLoopCoder.lora_manager is None


def test_iquest_loopcoder_lora_patch_preserves_native_support(monkeypatch):
    class NativeLoopCoder:
        supports_lora = True
        packed_modules_mapping: ClassVar = {"native": ["mapping"]}

    module = ModuleType("vllm.model_executor.models.iquest_loopcoder")
    module.IQuestLoopCoderForCausalLM = NativeLoopCoder
    monkeypatch.setitem(sys.modules, module.__name__, module)

    apply_iquest_loopcoder_lora_patch()

    assert NativeLoopCoder.packed_modules_mapping == {"native": ["mapping"]}


@pytest.mark.vllm
def test_iquest_loopcoder_satisfies_vllm_lora_protocol(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.model_executor.models.interfaces import SupportsLoRA, supports_lora
    from vllm.model_executor.models.iquest_loopcoder import IQuestLoopCoderForCausalLM

    monkeypatch.delattr(IQuestLoopCoderForCausalLM, "supports_lora", raising=False)
    apply_iquest_loopcoder_lora_patch()

    assert supports_lora(IQuestLoopCoderForCausalLM)
    instance = object.__new__(IQuestLoopCoderForCausalLM)
    assert isinstance(instance, SupportsLoRA)
    assert supports_lora(instance)
