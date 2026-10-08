"""CPU tests for pinning TE's NVRTC to the pip CUDA whose headers it compiles against.

TE 2.19 loads ``/usr/local/cuda``'s NVRTC ahead of the pip wheel but points its
JIT include dir at the pip ``nvidia/cu{major}`` headers, so a box with a
CUDA 12 toolkit next to a cu13 env fails MXFP8 RMSNorm with
``NVRTC_ERROR_COMPILATION``. A fake ``nvidia`` tree stands in for the wheels.
"""

import importlib.metadata
import importlib.util
import os
import sys
import types

import pytest

from skyrl.backends.skyrl_train.patches.te import pin_nvrtc


def _fake_nvidia_tree(root, major=13, with_nvrtc=True):
    lib = root / "nvidia" / f"cu{major}" / "lib"
    lib.mkdir(parents=True)
    if with_nvrtc:
        (lib / f"libnvrtc.so.{major}").touch()
    return root / "nvidia" / f"cu{major}"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Clean env, no TE imported, cu13 TE core installed, fake nvidia tree on the path."""
    for var in ("NVRTC_HOME", "NVTE_CUDA_INCLUDE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delitem(sys.modules, "transformer_engine", raising=False)

    def fake_distribution(name):
        if name == "transformer-engine-cu13":
            return object()
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(pin_nvrtc.importlib.metadata, "distribution", fake_distribution)

    def fake_find_spec(name):
        assert name == "nvidia"
        return types.SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia")])

    monkeypatch.setattr(pin_nvrtc.importlib.util, "find_spec", fake_find_spec)
    return tmp_path


def test_sets_nvrtc_home_to_pip_cuda(env):
    cu13 = _fake_nvidia_tree(env)
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() == str(cu13)
    assert os.environ["NVRTC_HOME"] == str(cu13)


def test_respects_user_nvrtc_home_with_a_library(env, monkeypatch):
    _fake_nvidia_tree(env)
    user = _fake_nvidia_tree(env / "user")
    monkeypatch.setenv("NVRTC_HOME", str(user))
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() is None
    assert os.environ["NVRTC_HOME"] == str(user)


def test_replaces_nvrtc_home_without_a_library(env, monkeypatch):
    # e.g. inherited from a driver whose ephemeral env is gone: TE would skip it
    # and fall back to /usr/local/cuda, the exact mismatch being avoided.
    cu13 = _fake_nvidia_tree(env)
    monkeypatch.setenv("NVRTC_HOME", str(env / "gone"))
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() == str(cu13)


def test_leaves_user_include_dir_choice_alone(env, monkeypatch):
    _fake_nvidia_tree(env)
    monkeypatch.setenv("NVTE_CUDA_INCLUDE_DIR", "/usr/local/cuda/include")
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() is None
    assert "NVRTC_HOME" not in os.environ


def test_noop_once_te_is_imported(env, monkeypatch):
    _fake_nvidia_tree(env)
    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() is None
    assert "NVRTC_HOME" not in os.environ


def test_noop_without_te_core_wheel(env, monkeypatch):
    _fake_nvidia_tree(env)

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(pin_nvrtc.importlib.metadata, "distribution", missing)
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() is None


def test_noop_without_pip_nvrtc(env):
    _fake_nvidia_tree(env, with_nvrtc=False)
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() is None
    assert "NVRTC_HOME" not in os.environ


def test_matches_te_core_major_not_other_pip_cudas(env, monkeypatch):
    _fake_nvidia_tree(env, major=12)
    cu13 = _fake_nvidia_tree(env, major=13)
    assert pin_nvrtc.pin_te_nvrtc_to_pip_cuda() == str(cu13)


def test_installed_te_still_prefers_system_nvrtc():
    # Tripwire: if TE starts preferring its pip NVRTC, this patch is dead weight
    # and should be deleted (see the module docstring).
    import inspect

    pytest.importorskip("torch")
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None:
        pytest.skip("transformer_engine not installed")
    import transformer_engine.common as te_common

    source = inspect.getsource(te_common._load_cuda_library)
    assert source.index("_load_cuda_library_from_system") < source.index("_load_cuda_library_from_python")
