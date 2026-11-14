"""CPU test for ``patch_nvfp4_per_token_e4m3_256``: applied iff the FlashInfer env asks for the 256 bound.

vLLM is replaced by stub modules in ``sys.modules`` so the env gating, the rebinding of by-name copies and the
missing-target error are tested without the wheel.
"""

import sys
import types

import pytest

from skyrl.backends.skyrl_train.patches.vllm import (
    patch_nvfp4_per_token_e4m3_256 as patch,
)

TARGET = patch.TARGET_MODULE
COPY = "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe"


@pytest.fixture
def fake_vllm(monkeypatch):
    mods = {}
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", TARGET, COPY):
        mods[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mods[name])
    mods[TARGET].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE = 1.0 / (448.0 * 6.0)
    mods[COPY].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE = 1.0 / (448.0 * 6.0)  # imported by name at its import
    monkeypatch.setattr(patch, "_APPLIED", False)
    return mods


def _env(monkeypatch, four6, use256):
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", four6)
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", use256)


@pytest.mark.parametrize("four6,use256", [("0", "0"), ("1", "0"), ("0", "1")])
def test_noop_unless_both_set(fake_vllm, monkeypatch, four6, use256):
    _env(monkeypatch, four6, use256)
    assert patch.apply_nvfp4_per_token_e4m3_256_patch() is False
    assert fake_vllm[TARGET].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / (448.0 * 6.0)


def test_applies_and_rebinds_copies(fake_vllm, monkeypatch):
    _env(monkeypatch, "1", "1")
    logs = []
    monkeypatch.setattr(patch.logger, "warning", lambda msg, *a: logs.append(msg.format(*a)))
    assert patch.apply_nvfp4_per_token_e4m3_256_patch() is True
    assert fake_vllm[TARGET].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / (256.0 * 6.0)
    assert fake_vllm[COPY].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / (256.0 * 6.0)
    assert len(logs) == 1 and logs[0].startswith("[skyrl nvfp4] per-token activation base global scale = 1/(256*6)")
    assert "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1" in logs[0] and "pid=" in logs[0]
    # Idempotent, logs once.
    assert patch.apply_nvfp4_per_token_e4m3_256_patch() is True and len(logs) == 1


def test_raises_when_target_missing(fake_vllm, monkeypatch):
    _env(monkeypatch, "1", "1")
    del fake_vllm[TARGET].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE
    with pytest.raises(RuntimeError, match="lacks"):
        patch.apply_nvfp4_per_token_e4m3_256_patch()


def test_raises_when_target_module_missing(fake_vllm, monkeypatch):
    _env(monkeypatch, "1", "1")
    monkeypatch.delitem(sys.modules, TARGET)
    with pytest.raises(RuntimeError, match="lacks"):
        patch.apply_nvfp4_per_token_e4m3_256_patch()


def test_raises_on_unexpected_value(fake_vllm, monkeypatch):
    _env(monkeypatch, "1", "1")
    fake_vllm[TARGET].NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE = 0.5
    with pytest.raises(RuntimeError, match="Unexpected"):
        patch.apply_nvfp4_per_token_e4m3_256_patch()


def test_wants_e4m3_256():
    assert patch.wants_e4m3_256({"FLASHINFER_NVFP4_4OVER6": "1", "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1"})
    assert not patch.wants_e4m3_256({"FLASHINFER_NVFP4_4OVER6": "1"})
    assert not patch.wants_e4m3_256({})


@pytest.mark.vllm
def test_real_vllm_target(monkeypatch):
    """The pinned vLLM has the constant at 1/(448*6), and the patch moves it to 1/(256*6)."""
    pytest.importorskip("vllm")
    import importlib

    importlib.import_module("vllm.model_executor.layers.fused_moe")  # break the flashinfer_utils import cycle
    module = importlib.import_module(TARGET)
    assert module.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / (448.0 * 6.0)
    monkeypatch.setattr(module, "NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE", module.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE)
    monkeypatch.setattr(patch, "_APPLIED", False)
    _env(monkeypatch, "1", "1")
    assert patch.apply_nvfp4_per_token_e4m3_256_patch()
    assert module.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / (256.0 * 6.0)


@pytest.mark.vllm
def test_real_vllm_patch_from_a_cold_import(monkeypatch):
    """Applied before anything imported vLLM's MoE modules (as at worker-extension import): no import cycle."""
    pytest.importorskip("vllm")
    import subprocess

    code = (
        "import os, sys\n"
        "os.environ['FLASHINFER_NVFP4_4OVER6'] = '1'\n"
        "os.environ['FLASHINFER_NVFP4_4OVER6_E4M3_USE_256'] = '1'\n"
        "from skyrl.backends.skyrl_train.patches.vllm import patch_nvfp4_per_token_e4m3_256 as p\n"
        "assert p.apply_nvfp4_per_token_e4m3_256_patch()\n"
        f"import {TARGET} as m\n"
        "from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutedsl_moe as c\n"
        "assert m.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / 1536.0, m.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE\n"
        "assert c.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == 1.0 / 1536.0\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-3000:]
    assert "[skyrl nvfp4] per-token activation base global scale = 1/(256*6)" in out.stderr
