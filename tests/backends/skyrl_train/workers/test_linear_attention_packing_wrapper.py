"""HFModelWrapper must hand packed-sequence boundaries to Gated DeltaNet layers.

These use a fake model so they exercise only the wrapper's wiring, not the transformer kernels.
"""

import sys
import types

import pytest
import torch

from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    _get_critic_model,
)


class BoundaryAwareGatedDeltaNet(torch.nn.Module):
    """Mirrors transformers >= 5.9: the kwarg reaches the chunked delta rule."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states, kwargs.pop("cu_seq_lens_q", None)


class RecordingModel(torch.nn.Module):
    """Records the kwargs of the last forward and returns logits of the right shape."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.received_kwargs = None

    def forward(self, sequences, **kwargs):
        self.received_kwargs = kwargs
        batch, seqlen = sequences.shape
        return {"logits": torch.zeros(batch, seqlen, 16)}


@pytest.fixture
def kernels(monkeypatch):
    def chunk_gated_delta_rule(query, key, value, g, beta, cu_seqlens=None):
        return None

    def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None):
        return None

    monkeypatch.setitem(
        sys.modules,
        "fla.ops.gated_delta_rule",
        types.SimpleNamespace(chunk_gated_delta_rule=chunk_gated_delta_rule),
    )
    monkeypatch.setitem(sys.modules, "causal_conv1d", types.SimpleNamespace(causal_conv1d_fn=causal_conv1d_fn))


@pytest.fixture
def logprobs(monkeypatch):
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.workers.model_wrapper.logprobs_from_logits",
        lambda logits, sequences, **kwargs: torch.zeros(logits.shape[0], logits.shape[1]),
    )


def _wrapper(model, *, remove_microbatch_padding=True, sequence_parallel_size=1):
    return HFModelWrapper(
        pretrain_or_model=model,
        use_flash_attention_2=True,
        bf16=False,
        sequence_parallel_size=sequence_parallel_size,
        remove_microbatch_padding=remove_microbatch_padding,
    )


def test_packed_wrapper_rejects_gated_delta_net_under_sequence_parallelism(kernels):
    model = RecordingModel(BoundaryAwareGatedDeltaNet())

    with pytest.raises(ValueError, match="1 Gated DeltaNet layers cannot be split across"):
        _wrapper(model, sequence_parallel_size=2)


def test_wrapper_rejects_gated_delta_net_under_sequence_parallelism_without_packing():
    # The recurrence cannot be sharded across ranks regardless of packing, and the guard must not
    # need the kernels to notice (gated_delta_net_layers does not import them).
    model = RecordingModel(BoundaryAwareGatedDeltaNet())

    with pytest.raises(ValueError, match="1 Gated DeltaNet layers cannot be split across"):
        _wrapper(model, remove_microbatch_padding=False, sequence_parallel_size=2)


def test_packed_wrapper_accepts_gated_delta_net_without_sequence_parallelism(kernels):
    _wrapper(RecordingModel(BoundaryAwareGatedDeltaNet()))


def test_packed_forward_passes_boundaries_to_gated_delta_net(kernels, logprobs):
    model = RecordingModel(BoundaryAwareGatedDeltaNet())
    wrapper = _wrapper(model)

    # Two rows of three tokens: packing unpads them into one row of six, so each restarts at 0.
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    attention_mask = torch.ones_like(input_ids)
    wrapper(input_ids, 2, attention_mask)

    assert model.received_kwargs["cu_seq_lens_q"].tolist() == [0, 3, 6]
    assert model.received_kwargs["max_length_q"] == 3
    assert model.received_kwargs["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 1]]


def test_unpacked_forward_does_not_pass_boundaries(kernels, logprobs):
    model = RecordingModel(BoundaryAwareGatedDeltaNet())
    wrapper = _wrapper(model, remove_microbatch_padding=False)

    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    attention_mask = torch.ones_like(input_ids)
    wrapper(input_ids, 2, attention_mask)

    assert "cu_seq_lens_q" not in model.received_kwargs


def test_pure_attention_model_does_not_receive_boundaries(kernels, logprobs):
    model = RecordingModel(torch.nn.Linear(4, 4))
    wrapper = _wrapper(model)

    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    attention_mask = torch.ones_like(input_ids)
    wrapper(input_ids, 2, attention_mask)

    assert "cu_seq_lens_q" not in model.received_kwargs


class FakeCriticBackbone(torch.nn.Module):
    """Stands in for the critic's base model: records kwargs, returns hidden states."""

    def __init__(self):
        super().__init__()
        self.gdn = BoundaryAwareGatedDeltaNet()
        self.received_kwargs = None

    def forward(self, input_ids, **kwargs):
        self.received_kwargs = kwargs
        batch, seqlen = input_ids.shape
        return {"last_hidden_state": torch.zeros(batch, seqlen, 8)}


class FakeCriticConfig:
    _attn_implementation = "flash_attention_2"
    hidden_size = 8


class FakeCriticBase(torch.nn.Module):
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__()

    def post_init(self):
        pass


def _critic(*, sequence_parallel_size=1, remove_microbatch_padding=True):
    critic_cls = _get_critic_model(
        FakeCriticBase,
        lambda config: FakeCriticBackbone(),
        sequence_parallel_size=sequence_parallel_size,
        remove_microbatch_padding=remove_microbatch_padding,
    )
    return critic_cls(FakeCriticConfig())


def test_packed_critic_forwards_boundaries(kernels):
    critic = _critic()
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    attention_mask = torch.ones_like(input_ids)

    critic(input_ids, 2, attention_mask)

    kwargs = critic.model.received_kwargs
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 3, 6]
    assert kwargs["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 1]]


def test_critic_rejects_gated_delta_net_under_sequence_parallelism(kernels):
    with pytest.raises(ValueError, match="1 Gated DeltaNet layers cannot be split across"):
        _critic(sequence_parallel_size=2)
