"""CPU tests for the Megatron VLM freeze helpers (``trainable_params``).

The model mocks mimic the layout of Megatron-Bridge's ``Qwen3VLModel``
(``language_model``, ``vision_model`` with a ``merger`` and
``decoder.deepstack_merger_list``) and its ``freeze()``. No Megatron import is needed.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/workers/megatron/test_vlm_freeze.py
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from skyrl.backends.skyrl_train.distributed.megatron.trainable_params import (
    VLM_FREEZE_FIELDS,
    apply_vlm_freeze_to_provider,
    summarize_trainable_parameters,
)


class _VisionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


class _VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _VisionDecoder()
        self.merger = nn.Linear(4, 4)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.output_layer = nn.Linear(4, 8)


class _Qwen3VLLike(nn.Module):
    """``freeze()`` follows Megatron-Bridge's Qwen3VLModel.freeze."""

    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.vision_model = _VisionModel()

    def projection_modules(self):
        return [self.vision_model.merger, self.vision_model.decoder.deepstack_merger_list]

    def freeze(self, freeze_language_model, freeze_vision_model, freeze_vision_projection):
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_model:
            modules.append(self.vision_model)
        if freeze_vision_projection:
            modules.extend(self.projection_modules())
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        if freeze_vision_model and not freeze_vision_projection:
            for module in self.projection_modules():
                for param in module.parameters():
                    param.requires_grad = True


def _build(freeze_language_model, freeze_vision_model, freeze_vision_projection):
    chunk = _Qwen3VLLike()
    chunk.freeze(freeze_language_model, freeze_vision_model, freeze_vision_projection)
    return chunk


def _trainable(chunk):
    return {name for name, p in chunk.named_parameters() if p.requires_grad}


def _projection_names(chunk):
    return {
        n for n, _ in chunk.named_parameters() if n.startswith(("vision_model.merger", "vision_model.decoder.deep"))
    }


# ---------------------------------------------------------------------------
# apply_vlm_freeze_to_provider
# ---------------------------------------------------------------------------


def _megatron_cfg(**flags):
    return SimpleNamespace(**{name: flags.get(name, False) for name in VLM_FREEZE_FIELDS})


def _vlm_provider(**defaults):
    return SimpleNamespace(**{name: defaults.get(name, False) for name in VLM_FREEZE_FIELDS})


def test_apply_leaves_a_text_only_provider_alone_when_no_flag_is_set():
    provider = SimpleNamespace()  # no freeze fields at all: a text-only provider
    assert apply_vlm_freeze_to_provider(provider, _megatron_cfg()) is None
    assert not any(hasattr(provider, name) for name in VLM_FREEZE_FIELDS)


def test_apply_writes_all_three_flags():
    provider = _vlm_provider()
    summary = apply_vlm_freeze_to_provider(
        provider, _megatron_cfg(freeze_language_model=True, freeze_vision_model=True)
    )
    assert (provider.freeze_language_model, provider.freeze_vision_model, provider.freeze_vision_projection) == (
        True,
        True,
        False,
    )
    assert (
        summary
        == "freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False (provider defaults overridden: freeze_language_model=False, freeze_vision_model=False)"
    )


def test_apply_overrides_provider_defaults_when_no_flag_is_set():
    # Bridge's Qwen3VLMoEModelProvider defaults freeze_language_model / freeze_vision_model to True;
    # with every flag unset the config says "train everything", and that must be what the provider gets.
    provider = _vlm_provider(freeze_language_model=True, freeze_vision_model=True)
    summary = apply_vlm_freeze_to_provider(provider, _megatron_cfg())
    assert not any(getattr(provider, name) for name in VLM_FREEZE_FIELDS)
    assert "provider defaults overridden: freeze_language_model=True, freeze_vision_model=True" in summary


def test_apply_reports_no_override_when_provider_defaults_match():
    summary = apply_vlm_freeze_to_provider(_vlm_provider(), _megatron_cfg())
    assert summary == "freeze_language_model=False, freeze_vision_model=False, freeze_vision_projection=False"


def test_apply_rejects_a_provider_without_freeze_support():
    provider = SimpleNamespace(freeze_language_model=False)
    with pytest.raises(ValueError, match="has no .*freeze_vision_model"):
        apply_vlm_freeze_to_provider(provider, _megatron_cfg(freeze_vision_model=True))


# ---------------------------------------------------------------------------
# selection and summary
# ---------------------------------------------------------------------------


def test_projection_only_selection_matches_the_projection():
    chunk = _build(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False)
    assert _trainable(chunk) == _projection_names(chunk)


def test_summary_counts_and_states():
    chunk = _build(True, True, False)
    wrapped = nn.Module()
    wrapped.module = nn.Module()
    wrapped.module.module = chunk  # DDP(Float16Module(model)) naming
    lines = summarize_trainable_parameters([wrapped])

    total = sum(p.numel() for p in chunk.parameters())
    projection = sum(p.numel() for n, p in chunk.named_parameters() if n in _projection_names(chunk))
    assert lines[0].startswith(f"trainable parameters: {projection} of {total} (")
    by_prefix = {line.strip().split(":")[0]: line for line in lines[1:]}
    assert "frozen" in by_prefix["language_model.decoder"]
    assert by_prefix["vision_model.merger"].endswith("trainable")
    assert "partly trainable" in by_prefix["vision_model.decoder"]  # blocks frozen, DeepStack mergers trainable
