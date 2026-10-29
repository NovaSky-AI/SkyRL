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
    # ``repetition_penalty`` is NOT forwarded by the typed-params path (matching vLLM);
    # it flows only through ``additional_kwargs`` per the SamplingParams docstring.
    assert "repetition_penalty" not in kwargs
    assert kwargs["stop_strings"] == ["<|im_end|>"]
    # `logprobs` is a vLLM field with no `generate` equivalent; it must not leak.
    assert "logprobs" not in kwargs


def test_neuron_sampling_params_skips_disabled_knobs():
    params = SamplingParams()  # defaults: top_k=-1, min_p=0.0, repetition_penalty=1.0
    kwargs = get_neuron_sampling_params(params)
    assert "top_k" not in kwargs
    assert "min_p" not in kwargs
    # No stop strings: the key is left unset rather than emitted as a dead ``None``.
    assert "stop_strings" not in kwargs
    # temperature=1.0 must sample (vLLM semantics), not fall back to greedy.
    assert kwargs["do_sample"] is True


def test_neuron_sampling_params_forwards_repetition_penalty_via_additional_kwargs():
    # The only supported path for repetition_penalty (mirrors vLLM): extra kwargs.
    params = SamplingParams(repetition_penalty=1.0, additional_kwargs={"repetition_penalty": 1.3})
    kwargs = get_neuron_sampling_params(params)
    assert kwargs["repetition_penalty"] == 1.3


def test_neuron_sampling_params_accepts_plain_dict():
    # Generators pass ``asdict(SamplingParams)`` -- a plain dict, not the typed config.
    params = {
        "max_generate_length": 96,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "logprobs": 1,
        "stop": ["<|im_end|>"],
        "additional_kwargs": None,
    }
    kwargs = get_neuron_sampling_params(params)
    assert kwargs["max_new_tokens"] == 96
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 40
    assert kwargs["stop_strings"] == ["<|im_end|>"]
    assert kwargs["do_sample"] is True
    assert "min_p" not in kwargs
    assert "logprobs" not in kwargs
    assert "stop" not in kwargs


def test_get_sampling_params_for_backend_dispatch():
    params = SamplingParams(max_generate_length=32)
    # The native neuron engine maps SkyRL params to ``generate`` kwargs *inside*
    # ``NeuronInferenceEngine.generate`` (via ``get_neuron_sampling_params``), so the
    # dispatch layer hands it the raw SkyRL params here -- ``max_generate_length``,
    # ``logprobs`` and ``stop`` preserved -- rather than the already-mapped
    # ``max_new_tokens`` kwargs. Re-mapping at this layer would be double-mapped in the
    # engine and would drop the ``logprobs``/``stop`` keys the generators read first.
    neuron_params = get_sampling_params_for_backend("neuron", params)
    assert neuron_params["max_generate_length"] == 32
    assert "logprobs" in neuron_params
    assert get_sampling_params_for_backend("vllm", params)["max_tokens"] == 32
    with pytest.raises(ValueError):
        get_sampling_params_for_backend("bogus", params)
