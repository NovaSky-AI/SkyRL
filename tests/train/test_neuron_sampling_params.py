"""
uv run --isolated --extra dev pytest -s tests/train/test_neuron_sampling_params.py
"""

import pytest

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_neuron_sampling_params,
    get_sampling_params_for_backend,
)
from skyrl.train.config import SamplingParams


def test_neuron_sampling_params_maps_generate_kwargs():
    params = SamplingParams(
        max_generate_length=64,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        min_p=0.05,
        repetition_penalty=1.2,
        stop=["<|im_end|>"],
    )
    kwargs = get_neuron_sampling_params(params)
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 50
    assert kwargs["min_p"] == 0.05
    assert kwargs["repetition_penalty"] == 1.2
    assert kwargs["stop_strings"] == ["<|im_end|>"]
    # `logprobs` is a vLLM field with no `generate` equivalent; it must not leak.
    assert "logprobs" not in kwargs


def test_neuron_sampling_params_skips_disabled_knobs():
    params = SamplingParams()  # defaults: top_k=-1, min_p=0.0, repetition_penalty=1.0
    kwargs = get_neuron_sampling_params(params)
    assert "top_k" not in kwargs
    assert "min_p" not in kwargs
    assert kwargs["stop_strings"] is None


def test_get_sampling_params_for_backend_dispatch():
    params = SamplingParams(max_generate_length=32)
    assert get_sampling_params_for_backend("neuron", params)["max_new_tokens"] == 32
    assert get_sampling_params_for_backend("vllm", params)["max_tokens"] == 32
    with pytest.raises(ValueError):
        get_sampling_params_for_backend("bogus", params)
