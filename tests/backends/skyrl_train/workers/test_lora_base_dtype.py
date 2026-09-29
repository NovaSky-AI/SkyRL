"""Parameter dtypes of an FSDP LoRA policy under ``trainer.policy.model.lora.base_dtype``.

``FSDPPolicyWorkerBase`` builds the policy with ``HFModelWrapper(bf16=...)``; ``base_dtype="bfloat16"`` sets
``bf16=True``. These tests load a tiny model through ``HFModelWrapper`` on CPU and check what is stored in which
dtype.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    # HFModelWrapper is built directly; no Ray actors are needed.
    yield


@pytest.fixture(scope="module")
def tiny_bf16_checkpoint(tmp_path_factory):
    """A tiny Qwen3 causal LM saved in bf16, as released checkpoints usually are."""
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny_qwen3_bf16")
    model.save_pretrained(path)
    return str(path)


def _wrap(path: str, bf16: bool) -> HFModelWrapper:
    return HFModelWrapper(path, bf16=bf16, use_flash_attention_2=False, lora_rank=4, lora_alpha=8)


def _dtypes(wrapper: HFModelWrapper) -> tuple[set, set]:
    """(dtypes of the LoRA adapter parameters, dtypes of every other parameter)."""
    params = list(wrapper.model.named_parameters())
    lora = {p.dtype for name, p in params if "lora_" in name}
    base = {p.dtype for name, p in params if "lora_" not in name}
    return lora, base


def test_bf16_base_keeps_fp32_adapters(tiny_bf16_checkpoint):
    wrapper = _wrap(tiny_bf16_checkpoint, bf16=True)
    lora, base = _dtypes(wrapper)

    assert base == {torch.bfloat16}
    assert lora == {torch.float32}
    trainable = [name for name, p in wrapper.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)


def test_default_stores_everything_in_fp32(tiny_bf16_checkpoint):
    lora, base = _dtypes(_wrap(tiny_bf16_checkpoint, bf16=False))

    assert base == {torch.float32}
    assert lora == {torch.float32}


def test_bf16_base_holds_the_checkpoint_values(tiny_bf16_checkpoint):
    """The bf16 checkpoint upcast to fp32 is exact, so the two storages hold the same base values."""
    fp32_base = {n: p for n, p in _wrap(tiny_bf16_checkpoint, bf16=False).model.named_parameters() if "lora_" not in n}
    bf16_base = {n: p for n, p in _wrap(tiny_bf16_checkpoint, bf16=True).model.named_parameters() if "lora_" not in n}

    assert fp32_base.keys() == bf16_base.keys()
    for name, value in bf16_base.items():
        torch.testing.assert_close(value.float(), fp32_base[name], rtol=0, atol=0, msg=name)
