"""CPU test for ``patch_dsa_hybrid_indexer``: hook resolution only.

Fake backend modules stand in for ``dsa_tilelang_kernels`` / ``dsa_cudnn_kernels`` so neither
TileLang nor cuDNN/FlashMLA has to import. The kernels themselves are covered by megatron-core.
"""

import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

HOOKS = ("run_fused_qk_topk", "run_fused_qk_topk_with_loss", "run_fused_absorbed_sparse_attention")


@pytest.fixture
def dsa_kernels(monkeypatch):
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    from skyrl.backends.skyrl_train.patches.megatron import patch_dsa_hybrid_indexer

    names = {}
    for backend in ("tilelang", "cudnn"):
        name = f"_fake_dsa_{backend}_kernels"
        module = types.ModuleType(name)
        for hook in HOOKS:
            setattr(module, hook, f"{backend}.{hook}")
        monkeypatch.setitem(sys.modules, name, module)
        names[backend] = name
    monkeypatch.setattr(dsa_kernels, "_BACKEND_MODULE_NAME_BY_BACKEND", names)
    monkeypatch.setattr(dsa_kernels, "_BACKEND", None)
    monkeypatch.setattr(dsa_kernels, "_BACKEND_SELECTION", None)
    # Restored after the test, so the patch's rebinding does not leak.
    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", dsa_kernels._resolve_fused_hook)
    monkeypatch.setattr(patch_dsa_hybrid_indexer, "_APPLIED", False)
    return dsa_kernels


def _apply(monkeypatch, value):
    from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_hybrid_indexer import (
        apply_dsa_hybrid_indexer_patch,
    )

    if value is None:
        monkeypatch.delenv("SKYRL_DSA_INDEXER_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SKYRL_DSA_INDEXER_BACKEND", value)
    apply_dsa_hybrid_indexer_patch()


def _resolved(dsa_kernels, backend):
    config = SimpleNamespace(dsa_kernel_backend=backend)
    return {hook: dsa_kernels._resolve_fused_hook(config, hook) for hook in HOOKS}


def test_cudnn_takes_only_the_indexer_topk_from_tilelang(dsa_kernels, monkeypatch):
    _apply(monkeypatch, "tilelang")
    assert _resolved(dsa_kernels, "cudnn") == {
        "run_fused_qk_topk": "tilelang.run_fused_qk_topk",
        "run_fused_qk_topk_with_loss": "cudnn.run_fused_qk_topk_with_loss",
        "run_fused_absorbed_sparse_attention": "cudnn.run_fused_absorbed_sparse_attention",
    }


def test_other_backends_are_unchanged(dsa_kernels, monkeypatch):
    _apply(monkeypatch, "tilelang")
    assert set(_resolved(dsa_kernels, "tilelang").values()) == {f"tilelang.{hook}" for hook in HOOKS}
    assert set(_resolved(dsa_kernels, "none").values()) == {None}


def test_unset_env_leaves_cudnn_whole(dsa_kernels, monkeypatch):
    _apply(monkeypatch, None)
    assert set(_resolved(dsa_kernels, "cudnn").values()) == {f"cudnn.{hook}" for hook in HOOKS}


def test_rejects_unknown_indexer_backend(dsa_kernels, monkeypatch):
    with pytest.raises(ValueError, match="SKYRL_DSA_INDEXER_BACKEND"):
        _apply(monkeypatch, "cudnn")
