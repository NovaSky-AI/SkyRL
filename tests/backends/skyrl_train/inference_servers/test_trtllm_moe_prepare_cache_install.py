"""Install-time contract for the TRT-LLM MoE prepare cache.

The cache is a speed optimization layered onto a private vLLM helper, so its
design rule is that correctness never depends on it: anything unexpected must
degrade to the original function rather than fail a weight sync.
"""

import sys
import types

import pytest

from skyrl.backends.skyrl_train.inference_servers import (
    trtllm_moe_prepare_cache as cache_mod,
)


@pytest.fixture()
def fake_flashinfer_utils(monkeypatch):
    """Stand in for vllm's private flashinfer_utils module."""

    module = types.ModuleType("flashinfer_utils")
    package = types.ModuleType("vllm.model_executor.layers.quantization.utils")
    package.flashinfer_utils = module
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.quantization.utils", package)
    monkeypatch.setattr(cache_mod, "_INSTALLED", False)
    monkeypatch.setattr(cache_mod, "_ORIGINAL", None)
    monkeypatch.setattr(cache_mod, "_CACHE", {})
    yield module
    cache_mod._INSTALLED = False


def test_install_is_a_no_op_when_the_symbol_is_absent(fake_flashinfer_utils):
    assert cache_mod.install() is False


def test_drifted_call_shape_delegates_to_the_original(fake_flashinfer_utils, monkeypatch):
    """A future vLLM signature change must cost only the speedup, never the sync."""

    calls = []

    def _shuffle_mxfp8_moe_weights(w13, w2, w13_scale, w2_scale, is_gated, layout="new-arg"):
        calls.append(layout)
        return w13, w2, w13_scale, w2_scale

    fake_flashinfer_utils._shuffle_mxfp8_moe_weights = _shuffle_mxfp8_moe_weights
    monkeypatch.delenv("SKYRL_TRTLLM_MOE_PREPARE_CACHE", raising=False)

    assert cache_mod.install() is True
    patched = fake_flashinfer_utils._shuffle_mxfp8_moe_weights
    assert patched is not _shuffle_mxfp8_moe_weights

    # The drifted call: one more positional argument than the cache understands.
    out = patched("w13", "w2", "s13", "s2", True, "swizzled")

    assert calls == ["swizzled"]
    assert out == ("w13", "w2", "s13", "s2")


def test_env_opt_out_leaves_the_original_installed(fake_flashinfer_utils, monkeypatch):
    fake_flashinfer_utils._shuffle_mxfp8_moe_weights = lambda *a: a
    monkeypatch.setenv("SKYRL_TRTLLM_MOE_PREPARE_CACHE", "0")

    assert cache_mod.install() is False
