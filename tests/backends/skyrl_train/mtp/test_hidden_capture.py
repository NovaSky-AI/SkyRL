"""CPU unit tests for the MTP hidden-state capture / decoupled replay plumbing.

These use a fake MTP block (no Megatron) to verify the capture records the block's
arguments and that the decoupled replay detaches the trunk hidden states so the
draft gradient never reaches the policy backbone. The capture is model-agnostic
(GPTModel for DeepSeek/GLM/Qwen3-Next, MambaModel for Qwen3.5/NemotronH); the fakes
below stand in for either base class.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_hidden_capture.py
"""

import sys
import types

import torch
import torch.nn as nn

# Stub out megatron.core.utils.unwrap_model so hidden_capture imports on CPU.
# (hidden_capture also tries megatron.core.transformer.multi_token_prediction.get_mtp_layer_offset,
# which is absent here, so it falls back to offset 0 — the single-stage / no-PP case.)
_fake_mcore_utils = types.ModuleType("megatron.core.utils")
_fake_mcore_utils.unwrap_model = lambda m: m
sys.modules.setdefault("megatron", types.ModuleType("megatron"))
sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
sys.modules["megatron.core.utils"] = _fake_mcore_utils

from skyrl.backends.skyrl_train.mtp.adapter import (  # noqa: E402
    project_mtp_hidden_to_logits,
)
from skyrl.backends.skyrl_train.mtp.hidden_capture import MTPHiddenCapture  # noqa: E402


class _FakeMTPBlock(nn.Module):
    """Mimics MultiTokenPredictionBlock: returns cat([trunk; mtp_0; ...; mtp_{k-1}], dim=0)."""

    def __init__(self, hidden, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        # One distinct weight per MTP depth so each depth's output is separable.
        self.w = nn.ParameterList([nn.Parameter(torch.ones(hidden) * (i + 1)) for i in range(num_layers)])

    def forward(self, hidden_states, **kwargs):
        # depends on params AND trunk hidden, like the real block.
        chunks = [hidden_states] + [hidden_states * self.w[i] for i in range(self.num_layers)]
        return torch.cat(chunks, dim=0)


class _FakeGPT(nn.Module):
    """Stands in for any Megatron model exposing a native ``.mtp`` block + ``.config``."""

    def __init__(self, hidden=4, mtp_num_layers=1):
        super().__init__()
        self.mtp = _FakeMTPBlock(hidden, num_layers=mtp_num_layers)
        self.config = types.SimpleNamespace(mtp_num_layers=mtp_num_layers)


class _FakeMamba(_FakeGPT):
    """Stands in for MambaModel: same MTP surface plus the shared-output-layer surface the
    adapter's projection uses (``output_layer`` + tied ``shared_embedding_or_output_weight``)."""

    def __init__(self, hidden=4, vocab=5, mtp_num_layers=1):
        super().__init__(hidden=hidden, mtp_num_layers=mtp_num_layers)
        self.share_embeddings_and_output_weights = True
        self._out_weight = nn.Parameter(torch.randn(vocab, hidden))

    def shared_embedding_or_output_weight(self):
        return self._out_weight

    def output_layer(self, hidden, weight=None):
        w = weight if weight is not None else self._out_weight
        # [seq, batch, hidden] @ [hidden, vocab] -> [seq, batch, vocab]; mirror Megatron's (logits, bias).
        return hidden @ w.t(), None


def _run(detach_trunk, model_cls=_FakeGPT, mtp_num_layers=1):
    model = model_cls(mtp_num_layers=mtp_num_layers)
    capture = MTPHiddenCapture(model, detach_trunk=detach_trunk)
    s, b, h = 3, 2, 4
    trunk = torch.randn(s, b, h, requires_grad=True)
    with capture.capture():
        # Simulate the model's in-forward MTP call (records kwargs via the pre-hook).
        _ = model.mtp(hidden_states=trunk, position_ids=torch.zeros(b, s))
    student = capture.compute_student_hidden_states()
    return model, trunk, student


def test_capture_returns_one_chunk_per_mtp_depth():
    _, _, student = _run(detach_trunk=True)
    assert student is not None and len(student) == 1
    assert student[0].shape == (3, 2, 4)


def test_capture_returns_one_chunk_per_depth_multilayer():
    # chunk[-num_layers:] must return exactly this stage's MTP-depth outputs.
    _, _, student = _run(detach_trunk=True, mtp_num_layers=2)
    assert student is not None and len(student) == 2
    assert all(chunk.shape == (3, 2, 4) for chunk in student)


def test_replay_detaches_trunk_when_detach_trunk_true():
    model, trunk, student = _run(detach_trunk=True)
    student[0].sum().backward()
    # The MTP-head parameter receives gradient...
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    # ...but the trunk hidden states do NOT (decoupled).
    assert trunk.grad is None


def test_replay_couples_trunk_when_detach_trunk_false():
    model, trunk, student = _run(detach_trunk=False)
    student[0].sum().backward()
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    # With detach disabled, the gradient flows back into the trunk hidden states.
    assert trunk.grad is not None and trunk.grad.abs().sum() > 0


def test_capture_is_model_agnostic_mambamodel():
    # The capture must work identically for a MambaModel-like base (Qwen3.5), exposing the same
    # .mtp surface. capture.model is the unwrapped model (renamed from the old GPTModel-specific attr).
    model, trunk, student = _run(detach_trunk=True, model_cls=_FakeMamba)
    assert isinstance(MTPHiddenCapture(model).model, _FakeMamba)
    assert student is not None and len(student) == 1
    assert student[0].shape == (3, 2, 4)


def test_project_to_logits_decoupled_end_to_end():
    # End-to-end through the adapter: capture (detached) -> shared output layer -> student logits.
    # The draft gradient reaches the MTP head + shared output weight, never the trunk.
    model, trunk, student = _run(detach_trunk=True, model_cls=_FakeMamba)
    logits = project_mtp_hidden_to_logits(student, model)
    assert len(logits) == 1
    # [seq, batch, vocab] -> transposed to [batch, seq, vocab].
    assert logits[0].shape == (2, 3, 5)
    logits[0].sum().backward()
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    assert model._out_weight.grad is not None and model._out_weight.grad.abs().sum() > 0
    assert trunk.grad is None


def test_project_to_logits_detach_shared_output():
    # detach_output_weight=True isolates the shared embedding/output weight from the draft gradient.
    model, trunk, student = _run(detach_trunk=True, model_cls=_FakeMamba)
    logits = project_mtp_hidden_to_logits(student, model, detach_output_weight=True)
    logits[0].sum().backward()
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    assert model._out_weight.grad is None


class _FakeEmbedding(nn.Module):
    """Stands in for the shared LanguageModelEmbedding the MTP block re-embeds rolled ids with."""

    def __init__(self, vocab=5, hidden=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, input_ids=None, position_ids=None):
        return self.weight[input_ids]


class _FakeMTPBlockWithEmbedding(_FakeMTPBlock):
    """Like the real block: re-embeds rolled input ids via the ``embedding`` kwarg and mixes the
    result into each depth's output, creating the second gradient path into the shared weight."""

    def forward(self, hidden_states, input_ids=None, embedding=None, **kwargs):
        emb = embedding(input_ids=input_ids).transpose(0, 1)  # [b, s, h] -> [s, b, h]
        chunks = [hidden_states] + [(hidden_states + emb) * self.w[i] for i in range(self.num_layers)]
        return torch.cat(chunks, dim=0)


def _run_with_embedding(detach_shared_embedding):
    model = _FakeGPT()
    model.mtp = _FakeMTPBlockWithEmbedding(hidden=4, num_layers=1)
    embedding = _FakeEmbedding(vocab=5, hidden=4)
    capture = MTPHiddenCapture(model, detach_trunk=True, detach_shared_embedding=detach_shared_embedding)
    s, b = 3, 2
    trunk = torch.randn(s, b, 4, requires_grad=True)
    ids = torch.randint(0, 5, (b, s))
    with capture.capture():
        _ = model.mtp(hidden_states=trunk, input_ids=ids, embedding=embedding)
    student = capture.compute_student_hidden_states()
    student[0].sum().backward()
    return model, embedding


def test_replay_detaches_shared_embedding_when_requested():
    # mtp_detach_shared_output must also sever the re-embedding path (the second route into the
    # tied embedding/lm_head, next to the output projection) so only MTP-head params train.
    model, embedding = _run_with_embedding(detach_shared_embedding=True)
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    assert embedding.weight.grad is None


def test_replay_trains_shared_embedding_by_default():
    model, embedding = _run_with_embedding(detach_shared_embedding=False)
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    assert embedding.weight.grad is not None and embedding.weight.grad.abs().sum() > 0


def test_replay_asserts_on_positional_hidden_states():
    # The trunk detach patches kwargs only; if a future megatron passes hidden_states positionally
    # the capture must fail loudly instead of silently coupling the draft loss into the trunk.
    import pytest

    model = _FakeGPT()
    trunk = torch.randn(3, 2, 4, requires_grad=True)
    capture = MTPHiddenCapture(model, detach_trunk=True)
    with capture.capture():
        _ = model.mtp(trunk, position_ids=torch.zeros(2, 3))  # positional hidden_states
    with pytest.raises(AssertionError, match="keyword argument"):
        capture.compute_student_hidden_states()


class _FakeUntiedOutputLayer(nn.Module):
    def __init__(self, vocab=5, hidden=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, hidden, weight=None):
        w = weight if weight is not None else self.weight
        return hidden @ w.t(), None


class _FakeUntied(_FakeGPT):
    """Untied embeddings: no shared weight; the output layer owns its own parameter."""

    def __init__(self, hidden=4, vocab=5, mtp_num_layers=1):
        super().__init__(hidden=hidden, mtp_num_layers=mtp_num_layers)
        self.share_embeddings_and_output_weights = False
        self.output_layer = _FakeUntiedOutputLayer(vocab=vocab, hidden=hidden)


def test_project_to_logits_detach_output_weight_untied_model():
    # detach_output_weight must also isolate an UNTIED model's own output-layer weight (passed
    # explicitly as a detached tensor), not just the tied/shared weight.
    model, trunk, student = _run(detach_trunk=True, model_cls=_FakeUntied)
    logits = project_mtp_hidden_to_logits(student, model, detach_output_weight=True)
    logits[0].sum().backward()
    assert model.mtp.w[0].grad is not None and model.mtp.w[0].grad.abs().sum() > 0
    assert model.output_layer.weight.grad is None
    assert trunk.grad is None


class _FakeVL(nn.Module):
    """Stands in for a vision-language wrapper (e.g. Qwen3.5-VL ``Qwen3VLModel``): the text backbone
    and its MTP head are nested at ``.language_model``, and the top-level model has no ``.mtp``."""

    def __init__(self, hidden=4, vocab=5, mtp_num_layers=1):
        super().__init__()
        self.language_model = _FakeMamba(hidden=hidden, vocab=vocab, mtp_num_layers=mtp_num_layers)
        # Deliberately NO top-level .mtp / .config — mirrors the real VL wrapper.


def test_capture_descends_into_language_model_for_vl_wrapper():
    # Regression: Qwen3.5-VL nests the GPTModel + MTP head at model.language_model.mtp. The capture
    # must resolve that host instead of looking for a (nonexistent) top-level .mtp, otherwise no
    # student hidden states are produced and the draft loss is silently never computed.
    from skyrl.backends.skyrl_train.mtp.hidden_capture import _resolve_mtp_host

    model = _FakeVL(mtp_num_layers=1)
    assert getattr(model, "mtp", None) is None  # top-level has no MTP block...
    host = _resolve_mtp_host(model)
    assert host is model.language_model  # ...capture resolves the nested language model.

    capture = MTPHiddenCapture(model, detach_trunk=True)
    assert capture.model is model.language_model
    assert capture.mtp_num_layers == 1

    s, b, h = 3, 2, 4
    trunk = torch.randn(s, b, h, requires_grad=True)
    with capture.capture():
        _ = model.language_model.mtp(hidden_states=trunk, position_ids=torch.zeros(b, s))
    student = capture.compute_student_hidden_states()
    assert student is not None and len(student) == 1
    assert student[0].shape == (3, 2, 4)
    # End-to-end projection through the nested language model's shared output layer.
    logits = project_mtp_hidden_to_logits(student, capture.model)
    assert logits[0].shape == (2, 3, 5)


def test_resolve_mtp_host_returns_model_when_no_mtp_anywhere():
    from skyrl.backends.skyrl_train.mtp.hidden_capture import _resolve_mtp_host

    bare = _FakeGPT()
    bare.mtp = None
    # No .mtp on the model and no .language_model nesting -> return the model unchanged.
    assert _resolve_mtp_host(bare) is bare


def test_no_capture_when_block_absent():
    model = _FakeGPT()
    model.mtp = None
    capture = MTPHiddenCapture(model, detach_trunk=True)
    with capture.capture():
        pass
    assert capture.compute_student_hidden_states() is None
