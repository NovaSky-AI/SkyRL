"""CPU test for ``patch_offload_checkpoint_inputs``.

``checkpointed_forward`` is imported by name into both ``transformer_block`` and ``hybrid_block``,
so wrapping only one leaves the other's models silently without offload. Fake functions stand in
for megatron-core's and record whether they ran inside ``save_on_cpu``.
"""

import contextlib

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

# Listed here rather than read from the patch, so a patch that forgets one still fails.
IMPORTERS = (
    "megatron.core.transformer.transformer_block",
    "megatron.core.models.hybrid.hybrid_block",
)


@pytest.fixture
def offload(monkeypatch):
    import importlib

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_offload_checkpoint_inputs as patch,
    )

    calls = {}
    active = {"save_on_cpu": False}

    @contextlib.contextmanager
    def fake_save_on_cpu(pin_memory=False):
        active["save_on_cpu"] = True
        try:
            yield
        finally:
            active["save_on_cpu"] = False

    modules = {}
    for name in IMPORTERS:
        module = importlib.import_module(name)
        short = name.rsplit(".", 1)[-1]

        def fake(*args, _short=short, **kwargs):
            calls[_short] = active["save_on_cpu"]
            return args

        monkeypatch.setattr(module, "checkpointed_forward", fake)
        modules[short] = module
    monkeypatch.setattr(patch, "_APPLIED", False)
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", fake_save_on_cpu)
    return patch, modules, calls


def test_wraps_every_importer(offload):
    patch, modules, calls = offload
    originals = {short: module.checkpointed_forward for short, module in modules.items()}

    assert patch.patch_offload_checkpoint_inputs() is True
    assert set(modules) == {"transformer_block", "hybrid_block"}
    for short, module in modules.items():
        assert module.checkpointed_forward is not originals[short]
        assert module.checkpointed_forward.__wrapped__ is originals[short]
        assert module.checkpointed_forward("x") == ("x",)
        assert calls[short] is True, f"{short}.checkpointed_forward ran outside save_on_cpu"


def test_idempotent(offload):
    patch, modules, _ = offload
    assert patch.patch_offload_checkpoint_inputs() is True
    first = {short: module.checkpointed_forward for short, module in modules.items()}
    assert patch.patch_offload_checkpoint_inputs() is True
    for short, module in modules.items():
        assert module.checkpointed_forward is first[short]
