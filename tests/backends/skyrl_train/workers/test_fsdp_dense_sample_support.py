import importlib
import importlib.machinery
import sys
import types

import pytest
import torch
from torch import nn


@pytest.fixture
def model_wrapper(monkeypatch):
    def unpad_input(tensor, attention_mask):
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        unpadded = tensor.flatten(0, 1).index_select(0, indices)
        return unpadded, indices, None, None, None

    def pad_input(tensor, indices, batch, seqlen):
        padded = tensor.new_zeros((batch * seqlen, *tensor.shape[1:]))
        return padded.index_copy(0, indices, tensor).reshape(batch, seqlen, *tensor.shape[1:])

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.pad_input = pad_input
    bert_padding.unpad_input = unpad_input
    flash_attn.bert_padding = bert_padding

    peft = types.ModuleType("peft")
    peft.LoraConfig = object
    peft.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft.get_peft_model = lambda model, config: model
    peft_tuners = types.ModuleType("peft.tuners")
    peft_lora = types.ModuleType("peft.tuners.lora")
    peft_lora.LoraLayer = nn.Module
    peft_tuners.lora = peft_lora
    peft.tuners = peft_tuners

    ulysses = types.ModuleType("skyrl.backends.skyrl_train.distributed.ulysses")
    ulysses_utils = types.ModuleType("skyrl.backends.skyrl_train.distributed.ulysses.utils")
    ulysses_utils.gather_outputs_and_unpad = None
    ulysses_utils.ulysses_pad_and_slice_inputs = None
    ulysses.utils = ulysses_utils

    stubs = {
        "flash_attn": flash_attn,
        "flash_attn.bert_padding": bert_padding,
        "peft": peft,
        "peft.tuners": peft_tuners,
        "peft.tuners.lora": peft_lora,
        "skyrl.backends.skyrl_train.distributed.ulysses": ulysses,
        "skyrl.backends.skyrl_train.distributed.ulysses.utils": ulysses_utils,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "skyrl.backends.skyrl_train.workers.model_wrapper"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


class _FakeCausalLM(nn.Module):
    def __init__(self, sequence_length: int, vocab_size: int):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(sequence_length, vocab_size, dtype=torch.float64))

    def forward(self, input_ids, **kwargs):
        return {"logits": (self.logits * 1.0).unsqueeze(0).expand(input_ids.shape[0], -1, -1)}


def _reference(logits, sequences, support):
    values = []
    for position in range(sequences.shape[1] - 1):
        sampled = sequences[0, position + 1]
        members = support[0, position + 1]
        members = members[members >= 0].long()
        values.append(logits[position, sampled] - torch.logsumexp(logits[position, members], dim=0))
    return torch.stack(values).unsqueeze(0)


def test_fsdp_forward_matches_dense_reference_values_and_gradients(model_wrapper):
    sequences = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(sequences)
    support = torch.tensor(
        [[[-1, -1, -1], [2, 5, -1], [3, -1, -1], [4, 0, 6]]],
        dtype=torch.int32,
    )
    model = _FakeCausalLM(sequence_length=4, vocab_size=7)
    wrapper = model_wrapper.HFModelWrapper(model, bf16=False)

    actual = wrapper(
        sequences,
        num_actions=3,
        attention_mask=attention_mask,
        sample_support_ids=support,
        loss_mask=torch.ones((1, 3), dtype=torch.bool),
        enable_sample_support_replay=True,
    )
    actual.sum().backward()
    actual_grad = model.logits.grad.clone()

    reference_logits = model.logits.detach().clone().requires_grad_(True)
    expected = _reference(reference_logits, sequences, support)
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, reference_logits.grad)


def test_fsdp_dense_support_runs_optimizer_step(model_wrapper):
    sequences = torch.tensor([[1, 2, 3]])
    support = torch.tensor([[[-1, -1], [2, 4], [3, 1]]], dtype=torch.int32)
    model = _FakeCausalLM(sequence_length=3, vocab_size=5)
    wrapper = model_wrapper.HFModelWrapper(model, bf16=False)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)
    before = model.logits.detach().clone()

    loss = -wrapper(
        sequences,
        num_actions=2,
        attention_mask=torch.ones_like(sequences),
        sample_support_ids=support,
        loss_mask=torch.ones((1, 2), dtype=torch.bool),
        enable_sample_support_replay=True,
    ).mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(model.logits, before)


def test_fsdp_dense_support_skips_full_sequence_vocabulary_logprobs(monkeypatch, model_wrapper):
    # The fixed-capacity EOS fallback may score B rows; the ordinary B*S path must stay disabled.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("full-sequence vocabulary logprobs should not run during support replay")

    monkeypatch.setattr(model_wrapper, "logprobs_from_logits", fail_if_called)
    sequences = torch.tensor([[1, 2, 3]])
    support = torch.tensor([[[-1, -1], [2, 4], [3, 1]]], dtype=torch.int32)
    wrapper = model_wrapper.HFModelWrapper(_FakeCausalLM(sequence_length=3, vocab_size=5), bf16=False)

    actual = wrapper(
        sequences,
        num_actions=2,
        attention_mask=torch.ones_like(sequences),
        sample_support_ids=support,
        loss_mask=torch.ones((1, 2), dtype=torch.bool),
        enable_sample_support_replay=True,
    )

    assert torch.isfinite(actual).all()


def test_fsdp_synthetic_eos_uses_full_vocabulary_logprob(model_wrapper):
    sequences = torch.tensor([[1, 2, 3]])
    support = torch.tensor([[[-1, -1], [2, 4], [-1, -1]]], dtype=torch.int32)
    model = _FakeCausalLM(sequence_length=3, vocab_size=5)
    wrapper = model_wrapper.HFModelWrapper(model, bf16=False)

    actual = wrapper(
        sequences,
        num_actions=2,
        attention_mask=torch.ones_like(sequences),
        sample_support_ids=support,
        loss_mask=torch.ones((1, 2), dtype=torch.bool),
        enable_sample_support_replay=True,
    )
    actual.sum().backward()
    actual_grad = model.logits.grad.clone()

    reference_logits = model.logits.detach().clone().requires_grad_(True)
    supported = reference_logits[0, 2] - torch.logsumexp(reference_logits[0, [2, 4]], dim=0)
    eos = reference_logits[1, 3] - torch.logsumexp(reference_logits[1], dim=0)
    expected = torch.stack((supported, eos)).unsqueeze(0)
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, reference_logits.grad)


def test_fsdp_packed_microbatch_matches_dense_reference(model_wrapper):
    sequences = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    support = torch.full((2, 4, 2), -1, dtype=torch.int32)
    support[0, 2:] = torch.tensor([[3, 8], [4, 0]], dtype=torch.int32)
    support[1, 2:] = torch.tensor([[6, 1], [7, 2]], dtype=torch.int32)
    model = _FakeCausalLM(sequence_length=7, vocab_size=9)
    wrapper = model_wrapper.HFModelWrapper(
        model,
        use_flash_attention_2=True,
        bf16=False,
        remove_microbatch_padding=True,
    )

    actual = wrapper(
        sequences,
        num_actions=2,
        attention_mask=attention_mask,
        sample_support_ids=support,
        loss_mask=torch.ones((2, 2), dtype=torch.bool),
        enable_sample_support_replay=True,
    )
    actual.sum().backward()
    actual_grad = model.logits.grad.clone()

    reference_logits = model.logits.detach().clone().requires_grad_(True)
    expected = torch.stack(
        (
            torch.stack(
                (
                    reference_logits[1, 3] - torch.logsumexp(reference_logits[1, [3, 8]], dim=0),
                    reference_logits[2, 4] - torch.logsumexp(reference_logits[2, [4, 0]], dim=0),
                )
            ),
            torch.stack(
                (
                    reference_logits[4, 6] - torch.logsumexp(reference_logits[4, [6, 1]], dim=0),
                    reference_logits[5, 7] - torch.logsumexp(reference_logits[5, [7, 2]], dim=0),
                )
            ),
        )
    )
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, reference_logits.grad)


def test_fsdp_packed_microbatch_supports_one_synthetic_eos_per_trajectory(model_wrapper):
    sequences = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    support = torch.full((2, 4, 2), -1, dtype=torch.int32)
    support[0, 2] = torch.tensor([3, 8], dtype=torch.int32)
    support[1, 2] = torch.tensor([6, 1], dtype=torch.int32)
    model = _FakeCausalLM(sequence_length=7, vocab_size=9)
    wrapper = model_wrapper.HFModelWrapper(
        model,
        use_flash_attention_2=True,
        bf16=False,
        remove_microbatch_padding=True,
    )

    actual = wrapper(
        sequences,
        num_actions=2,
        attention_mask=attention_mask,
        sample_support_ids=support,
        loss_mask=torch.ones((2, 2), dtype=torch.bool),
        enable_sample_support_replay=True,
    )
    actual.sum().backward()
    actual_grad = model.logits.grad.clone()

    reference_logits = model.logits.detach().clone().requires_grad_(True)
    expected = torch.stack(
        (
            torch.stack(
                (
                    reference_logits[1, 3] - torch.logsumexp(reference_logits[1, [3, 8]], dim=0),
                    reference_logits[2, 4] - torch.logsumexp(reference_logits[2], dim=0),
                )
            ),
            torch.stack(
                (
                    reference_logits[4, 6] - torch.logsumexp(reference_logits[4, [6, 1]], dim=0),
                    reference_logits[5, 7] - torch.logsumexp(reference_logits[5], dim=0),
                )
            ),
        )
    )
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, reference_logits.grad)


def test_fsdp_capture_only_matches_feature_disabled(model_wrapper):
    sequences = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(sequences)
    support = torch.tensor([[[-1, -1], [2, 4], [3, 1]]], dtype=torch.int32)
    wrapper = model_wrapper.HFModelWrapper(_FakeCausalLM(sequence_length=3, vocab_size=5), bf16=False)

    expected = wrapper(sequences, num_actions=2, attention_mask=attention_mask)
    actual = wrapper(
        sequences,
        num_actions=2,
        attention_mask=attention_mask,
        sample_support_ids=support,
        enable_sample_support_replay=False,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_mask": torch.ones((1, 2), dtype=torch.bool)}, "no recorded support"),
        ({"sample_support_ids": torch.full((1, 3, 2), -1, dtype=torch.int32)}, "no loss mask"),
    ],
)
def test_fsdp_replay_requires_support_and_loss_mask(model_wrapper, kwargs, message):
    wrapper = model_wrapper.HFModelWrapper(_FakeCausalLM(sequence_length=3, vocab_size=5), bf16=False)

    with pytest.raises(ValueError, match=message):
        wrapper(
            torch.tensor([[1, 2, 3]]),
            num_actions=2,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            enable_sample_support_replay=True,
            **kwargs,
        )
