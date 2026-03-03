import os

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl.tx.models.configs import Qwen3_5Config
from skyrl.tx.models.qwen3_5 import Qwen3_5ForCausalLM

from tests.tx.models.conftest import load_model


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3_5(tp: int):
    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", attn_implementation="eager", use_safetensors=True, torch_dtype=torch.float32
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )
    del hf_model

    _, model = load_model(
        "Qwen/Qwen3.5-0.8B",
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        ("fsdp", "tp"),
        mesh_shape=(1, tp),
        max_lora_adapters=32,
        max_lora_rank=32,
    )

    outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
    assert outputs.hidden_states is not None
    assert np.allclose(hf_outputs.hidden_states[0].numpy(), outputs.hidden_states[0], rtol=1e-6)
    assert np.allclose(hf_outputs.hidden_states[1].numpy(), outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
    assert np.allclose(hf_outputs.hidden_states[-1].numpy(), outputs.hidden_states[-1], rtol=1e-3, atol=1e-3)
