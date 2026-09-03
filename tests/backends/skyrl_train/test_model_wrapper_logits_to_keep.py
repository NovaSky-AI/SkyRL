from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from skyrl.backends.skyrl_train.workers import model_wrapper
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    _build_action_logits_selection,
    _can_use_action_logits_selection,
    _model_supports_logits_to_keep,
)


def _cpu_logprobs(logits, labels, inplace_backward=True):
    del inplace_backward
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@pytest.fixture(autouse=True)
def use_cpu_logprobs(monkeypatch):
    monkeypatch.setattr(model_wrapper, "logprobs_from_logits", _cpu_logprobs)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=23, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, image_token_id=vocab_size - 1)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.trunk = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.calls = 0
        self.last_logits_to_keep = None
        self.last_lm_head_tokens = None

    def _forward_impl(self, input_ids, logits_to_keep):
        self.calls += 1
        self.last_logits_to_keep = logits_to_keep
        hidden = torch.tanh(self.trunk(self.embed(input_ids)))
        selected = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        selected_hidden = hidden[:, selected, :]
        self.last_lm_head_tokens = selected_hidden.shape[1]
        return {"logits": self.lm_head(selected_hidden), "preserved_field": hidden.mean()}

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        del attention_mask, position_ids, kwargs
        return self._forward_impl(input_ids, logits_to_keep)


class UnsupportedTinyCausalLM(TinyCausalLM):
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return self._forward_impl(input_ids, 0)


def _wrapper(model, *, packed=False):
    return HFModelWrapper(
        model,
        bf16=False,
        use_flash_attention_2=packed,
        sequence_parallel_size=1,
        remove_microbatch_padding=packed,
        logprobs_chunk_size=2,
    )


def test_standard_path_passes_integer_and_aligns_targets_and_entropy(monkeypatch):
    model = TinyCausalLM()
    wrapper = _wrapper(model)
    sequences = torch.tensor([[0, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    captured = {}

    def capture_logprobs(logits, labels, inplace_backward=True):
        captured["logits_shape"] = logits.shape
        captured["labels"] = labels.detach().clone()
        return _cpu_logprobs(logits, labels, inplace_backward)

    original_entropy = wrapper.chunked_entropy_from_logits_fn

    def capture_entropy(logits, **kwargs):
        captured["entropy_mask"] = kwargs["attention_mask"].detach().clone()
        return original_entropy(logits, **kwargs)

    monkeypatch.setattr(model_wrapper, "logprobs_from_logits", capture_logprobs)
    wrapper.chunked_entropy_from_logits_fn = capture_entropy

    action_log_probs, output = wrapper(
        sequences,
        3,
        attention_mask=attention_mask,
        compute_entropy=True,
        return_output=True,
    )

    assert model.calls == 1
    assert model.last_logits_to_keep == 4
    assert model.last_lm_head_tokens == 4
    assert captured["logits_shape"][:2] == (2, 4)
    assert torch.equal(captured["labels"], torch.roll(sequences, -1, dims=1)[:, -4:])
    assert torch.equal(captured["entropy_mask"], attention_mask[:, -4:])
    assert action_log_probs.shape == (2, 3)
    assert output["logits"].shape[:2] == (2, 4)
    assert output["entropy"].shape == (2, 4)
    assert "preserved_field" in output


def test_standard_outputs_entropy_and_gradients_match_full_logits():
    torch.manual_seed(1858)
    optimized_model = TinyCausalLM()
    baseline_model = deepcopy(optimized_model)
    optimized = _wrapper(optimized_model)
    baseline = _wrapper(baseline_model)
    baseline._model_supports_logits_to_keep = False

    sequences = torch.tensor([[0, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    num_actions = 3

    baseline_actions, baseline_output = baseline(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        temperature=0.7,
        compute_entropy=True,
        entropy_requires_grad=True,
        return_output=True,
    )
    optimized_actions, optimized_output = optimized(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        temperature=0.7,
        compute_entropy=True,
        entropy_requires_grad=True,
        return_output=True,
    )
    baseline_action_entropy = baseline_output["entropy"][:, -num_actions - 1 : -1]
    optimized_action_entropy = optimized_output["entropy"][:, -num_actions - 1 : -1]

    torch.testing.assert_close(optimized_actions, baseline_actions, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(optimized_action_entropy, baseline_action_entropy, rtol=1e-6, atol=1e-6)

    (-baseline_actions.mean() - 0.03 * baseline_action_entropy.mean()).backward()
    (-optimized_actions.mean() - 0.03 * optimized_action_entropy.mean()).backward()
    torch.testing.assert_close(
        optimized_model.lm_head.weight.grad,
        baseline_model.lm_head.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        optimized_model.trunk.weight.grad,
        baseline_model.trunk.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_packed_variable_lengths_match_outputs_and_gradients_with_tensor_selector():
    torch.manual_seed(1858)
    optimized_model = TinyCausalLM()
    baseline_model = deepcopy(optimized_model)
    optimized = _wrapper(optimized_model, packed=True)
    baseline = _wrapper(baseline_model, packed=True)
    baseline._model_supports_logits_to_keep = False

    sequences = torch.tensor([[0, 0, 2, 3, 4, 5, 6], [0, 7, 8, 9, 10, 11, 12]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
    num_actions = 2

    baseline_actions, baseline_output = baseline(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        temperature=0.7,
        compute_entropy=True,
        entropy_requires_grad=True,
        return_output=True,
    )
    optimized_actions, optimized_output = optimized(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        temperature=0.7,
        compute_entropy=True,
        entropy_requires_grad=True,
        return_output=True,
    )
    baseline_action_entropy = baseline_output["entropy"][:, -num_actions - 1 : -1]
    optimized_action_entropy = optimized_output["entropy"][:, -num_actions - 1 : -1]

    assert torch.equal(optimized_model.last_logits_to_keep, torch.tensor([2, 3, 4, 8, 9, 10]))
    assert optimized_model.last_lm_head_tokens == 6
    assert optimized_output["logits"].shape[:2] == (1, 6)
    assert optimized_output["entropy"].shape == (2, 3)
    assert optimized_actions.shape == (2, 2)
    torch.testing.assert_close(optimized_actions, baseline_actions, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        optimized_action_entropy,
        baseline_action_entropy,
        rtol=1e-6,
        atol=1e-6,
    )

    selection = _build_action_logits_selection(
        num_actions,
        padded_sequence_length=7,
        nnz_indices=torch.tensor([2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]),
    )
    assert torch.equal(selection.restore_indices, torch.tensor([4, 5, 6, 11, 12, 13]))

    position_weights = baseline_actions.new_tensor([[0.1, 0.2], [0.3, 0.4]])
    baseline_loss = -((baseline_actions + 0.03 * baseline_action_entropy) * position_weights).sum()
    optimized_loss = -((optimized_actions + 0.03 * optimized_action_entropy) * position_weights).sum()
    baseline_loss.backward()
    optimized_loss.backward()

    baseline_parameters = dict(baseline_model.named_parameters())
    optimized_parameters = dict(optimized_model.named_parameters())
    assert list(optimized_parameters) == list(baseline_parameters)
    for name, optimized_parameter in optimized_parameters.items():
        baseline_parameter = baseline_parameters[name]
        assert optimized_parameter.grad is not None, name
        assert baseline_parameter.grad is not None, name
        torch.testing.assert_close(
            optimized_parameter.grad,
            baseline_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=f"gradient mismatch for {name}",
        )


def test_unsupported_model_falls_back_and_runs_forward_once():
    model = UnsupportedTinyCausalLM()
    wrapper = _wrapper(model)
    sequences = torch.tensor([[2, 3, 4, 5]])
    attention_mask = torch.ones_like(sequences)

    _, output = wrapper(sequences, 2, attention_mask=attention_mask, return_output=True)

    assert not wrapper._model_supports_logits_to_keep
    assert model.calls == 1
    assert model.last_lm_head_tokens == sequences.shape[1]
    assert output["logits"].shape[1] == sequences.shape[1]


def test_vlm_and_sequence_parallel_fallback_selection():
    assert not _can_use_action_logits_selection(
        model_supports_logits_to_keep=True,
        is_vlm=True,
        has_image_inputs=False,
        sequence_parallel_size=1,
    )
    assert not _can_use_action_logits_selection(
        model_supports_logits_to_keep=True,
        is_vlm=False,
        has_image_inputs=False,
        sequence_parallel_size=2,
    )

    model = TinyCausalLM()
    wrapper = _wrapper(model)
    wrapper.is_vlm = True
    sequences = torch.tensor([[2, 3, 4, 5]])
    attention_mask = torch.ones_like(sequences)
    _, output = wrapper(sequences, 2, attention_mask=attention_mask, return_output=True)
    assert model.calls == 1
    assert model.last_logits_to_keep == 0
    assert output["logits"].shape[1] == sequences.shape[1]


def test_single_action_uses_smallest_causal_window_and_runs_once():
    model = TinyCausalLM()
    wrapper = _wrapper(model)
    sequences = torch.tensor([[2, 3]])
    attention_mask = torch.ones_like(sequences)

    action_log_probs = wrapper(sequences, 1, attention_mask=attention_mask)

    assert model.calls == 1
    assert model.last_logits_to_keep == 2
    assert action_log_probs.shape == (1, 1)


def test_ambiguous_action_layout_does_not_build_selector():
    assert _build_action_logits_selection([2, 3], padded_sequence_length=8) is None
    assert _build_action_logits_selection(8, padded_sequence_length=8) is None


def test_peft_capability_check_uses_base_model_signature():
    config = GPT2Config(vocab_size=23, n_layer=1, n_head=1, n_embd=8, n_positions=8)
    base_model = GPT2LMHeadModel(config)
    peft_model = get_peft_model(
        base_model,
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["c_attn"]),
    )
    assert _model_supports_logits_to_keep(peft_model)
