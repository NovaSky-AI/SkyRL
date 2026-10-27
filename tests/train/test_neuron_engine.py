"""Tests for the native TorchNeuron inference engine (text path).

Model loading and device execution are not exercised here (no neuron device on CI);
these tests pin the interface contract and the pure sampling-logprob math.
"""

import torch

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.backends.skyrl_train.inference_servers.neuron_engine import NeuronInferenceEngine


def test_engine_fully_implements_interface():
    assert issubclass(NeuronInferenceEngine, InferenceEngineInterface)
    assert not NeuronInferenceEngine.__abstractmethods__


def test_sampled_token_logprobs_matches_softmax():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([2]), temperature=1.0, top_k=None, top_p=None, min_p=None
    )
    assert torch.allclose(out, logits.log_softmax(-1)[0, 2])


def test_sampled_token_logprobs_applies_temperature():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([2]), temperature=0.5, top_k=None, top_p=None, min_p=None
    )
    assert torch.allclose(out, (logits / 0.5).log_softmax(-1)[0, 2])


def test_sampled_token_logprobs_top_k_masks_out_of_rank():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([0]), temperature=1.0, top_k=1, top_p=None, min_p=None
    )
    assert torch.isneginf(out).all()


def test_sampled_token_logprobs_top_p_keeps_only_nucleus():
    # probs = [0.090, 0.244, 0.665]; top_p=0.5 keeps only the argmax (idx 2),
    # whose logprob under the renormalized singleton distribution is 0.0.
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([2]), temperature=1.0, top_k=None, top_p=0.5, min_p=None
    )
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6)
