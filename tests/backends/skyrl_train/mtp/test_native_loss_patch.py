# CPU tests for mtp/native_loss_patch.py: the sys.modules-wide rebind of megatron's
# process_mtp_loss and the MTPLossAutoScaler fail-loud backstop. Megatron is faked via
# monkeypatched sys.modules entries (restored after each test), so no GPU/megatron install
# is needed and real megatron -- if present -- is untouched.
#
# Run: uv run --extra dev pytest tests/backends/skyrl_train/mtp/test_native_loss_patch.py

import sys
import types

import pytest
import torch

from skyrl.backends.skyrl_train.mtp.native_loss_patch import (
    _skyrl_skip_native_mtp_loss,
    disable_native_mtp_loss,
)


def _install_fake_megatron(monkeypatch, with_process_fn=True):
    """Install a fake megatron.core.transformer.multi_token_prediction into sys.modules.

    Returns (mtp_module, original_process_fn, caller_module) where caller_module simulates a
    megatron model module that did `from ...multi_token_prediction import process_mtp_loss`
    BEFORE the patch ran.
    """

    def fake_process_mtp_loss(hidden_states, *args, **kwargs):
        return "NATIVE_LOSS_RAN"

    class FakeMTPLossAutoScaler:
        @staticmethod
        def apply(hidden_states, loss):
            return hidden_states

        main_loss_backward_scale = 1.0

    mtp_mod = types.ModuleType("megatron.core.transformer.multi_token_prediction")
    if with_process_fn:
        mtp_mod.process_mtp_loss = fake_process_mtp_loss
    mtp_mod.MTPLossAutoScaler = FakeMTPLossAutoScaler

    caller = types.ModuleType("megatron.core.models.fake.fake_model_for_test")
    if with_process_fn:
        caller.process_mtp_loss = fake_process_mtp_loss  # pre-patch from-import binding

    root = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    transformer = types.ModuleType("megatron.core.transformer")
    root.core = core
    core.transformer = transformer
    transformer.multi_token_prediction = mtp_mod

    for name, mod in [
        ("megatron", root),
        ("megatron.core", core),
        ("megatron.core.transformer", transformer),
        ("megatron.core.transformer.multi_token_prediction", mtp_mod),
        ("megatron.core.models.fake.fake_model_for_test", caller),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return mtp_mod, fake_process_mtp_loss, caller


def test_patches_definition_and_existing_from_import_bindings(monkeypatch):
    mtp_mod, original, caller = _install_fake_megatron(monkeypatch)

    disable_native_mtp_loss()

    # Definition module (future imports) and the pre-existing caller binding are both rebound.
    assert mtp_mod.process_mtp_loss is _skyrl_skip_native_mtp_loss
    assert caller.process_mtp_loss is _skyrl_skip_native_mtp_loss
    assert caller.process_mtp_loss is not original


def test_unrelated_process_mtp_loss_binding_untouched(monkeypatch):
    _install_fake_megatron(monkeypatch)

    def unrelated():
        return "mine"

    other = types.ModuleType("some.unrelated.module_for_test")
    other.process_mtp_loss = unrelated  # same name, different object
    monkeypatch.setitem(sys.modules, "some.unrelated.module_for_test", other)

    disable_native_mtp_loss()
    assert other.process_mtp_loss is unrelated


def test_autoscaler_apply_raises_after_patch(monkeypatch):
    mtp_mod, _, _ = _install_fake_megatron(monkeypatch)

    disable_native_mtp_loss()
    with pytest.raises(RuntimeError, match="MTPLossAutoScaler"):
        mtp_mod.MTPLossAutoScaler.apply(torch.zeros(2), torch.zeros(1))
    # set_loss_scale-style class attribute access is unaffected.
    assert mtp_mod.MTPLossAutoScaler.main_loss_backward_scale == 1.0


def test_idempotent(monkeypatch):
    mtp_mod, _, caller = _install_fake_megatron(monkeypatch)

    disable_native_mtp_loss()
    disable_native_mtp_loss()
    assert mtp_mod.process_mtp_loss is _skyrl_skip_native_mtp_loss
    assert caller.process_mtp_loss is _skyrl_skip_native_mtp_loss


def test_raises_when_process_mtp_loss_missing(monkeypatch):
    _install_fake_megatron(monkeypatch, with_process_fn=False)

    with pytest.raises(RuntimeError, match="process_mtp_loss"):
        disable_native_mtp_loss()


def test_noop_returns_first_chunk():
    class Cfg:
        mtp_num_layers = 2

    trunk = torch.arange(6.0).reshape(3, 2)
    mtp1 = trunk + 100
    mtp2 = trunk + 200
    hidden = torch.cat([trunk, mtp1, mtp2], dim=0)

    out = _skyrl_skip_native_mtp_loss(hidden, config=Cfg())
    assert torch.equal(out, trunk)

    # kwargs-style call as at the real call sites
    out = _skyrl_skip_native_mtp_loss(hidden_states=hidden, config=Cfg())
    assert torch.equal(out, trunk)

    # No MTP layers resolved -> passthrough.
    class NoMTPCfg:
        mtp_num_layers = None

    assert _skyrl_skip_native_mtp_loss(hidden, config=NoMTPCfg()) is hidden
