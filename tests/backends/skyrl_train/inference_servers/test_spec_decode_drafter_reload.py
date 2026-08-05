"""Unit tests for the spec-decode drafter weight-sync helper.

Regression guard for the MTP speculative-decoding bug: vLLM's main-model weight
reload does NOT update the separate spec-decode drafter (``model_runner.drafter.model``),
so after a colocate sleep(level=2) the drafter ran on garbage and draft-acceptance
collapsed to ~0. ``_reload_spec_decode_drafter`` re-loads the drafter from the same
synced weights. These tests use lightweight stand-ins (no vLLM / GPU required).
"""

import torch

from skyrl.backends.skyrl_train.inference_servers.spec_decode_utils import (
    _reload_spec_decode_drafter,
)


class _RecordingDrafterModel:
    def __init__(self):
        self.loaded = None

    def load_weights(self, weights):
        self.loaded = list(weights)
        return {n for n, _ in self.loaded}


class _Drafter:
    def __init__(self, model):
        self.model = model


class _ModelRunner:
    def __init__(self, drafter=None):
        self.model = object()  # the main model; irrelevant to this helper
        if drafter is not None:
            self.drafter = drafter


def _weights():
    return [("mtp.fc.weight", torch.zeros(2, 2)), ("model.embed_tokens.weight", torch.zeros(2, 2))]


def test_reloads_drafter_when_present():
    drafter_model = _RecordingDrafterModel()
    mr = _ModelRunner(drafter=_Drafter(drafter_model))
    weights = _weights()

    assert _reload_spec_decode_drafter(mr, weights) is True
    # The full weight list is forwarded; the drafter's own load_weights filters it.
    assert [n for n, _ in drafter_model.loaded] == [n for n, _ in weights]


def test_noop_without_drafter():
    mr = _ModelRunner(drafter=None)  # no spec decoding -> no .drafter attr
    assert _reload_spec_decode_drafter(mr, _weights()) is False


def test_noop_when_drafter_has_no_model():
    mr = _ModelRunner(drafter=_Drafter(model=None))  # e.g. ngram proposer
    assert _reload_spec_decode_drafter(mr, _weights()) is False


def test_noop_when_drafter_model_not_loadable():
    class _NoLoad:
        pass

    mr = _ModelRunner(drafter=_Drafter(model=_NoLoad()))
    assert _reload_spec_decode_drafter(mr, _weights()) is False
