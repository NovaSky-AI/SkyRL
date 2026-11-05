"""CPU tests for preloading cuDNN's sublibraries from the ``nvidia-cudnn`` wheel.

A fake wheel directory stands in for ``nvidia/cudnn/lib``, and ``ctypes.CDLL`` is replaced by a recorder,
so the tests check which libraries would be loaded, in which order, and when nothing is loaded.
"""

import ctypes

import pytest

from skyrl.backends.skyrl_train.patches.cudnn import preload


@pytest.fixture
def wheel_dir(tmp_path, monkeypatch):
    lib = tmp_path / "nvidia" / "cudnn" / "lib"
    lib.mkdir(parents=True)
    for name in (
        "libcudnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_ext.so.9",
    ):
        (lib / name).touch()
    monkeypatch.setattr(preload.sys, "platform", "linux")
    monkeypatch.setattr(preload, "_wheel_cudnn_lib_dir", lambda: str(lib))
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: set())
    return lib


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def fake_cdll(path, mode=0):
        calls.append((path, mode))
        return object()

    monkeypatch.setattr(preload.ctypes, "CDLL", fake_cdll)
    return calls


def test_loads_every_library_globally_in_dependency_order(wheel_dir, loads):
    loaded = preload.preload_cudnn()

    names = [p.split("/")[-1] for p, _ in loads]
    assert names[:3] == ["libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_engines_precompiled.so.9"]
    assert sorted(names) == sorted(p.name for p in wheel_dir.iterdir())
    assert all(mode == ctypes.RTLD_GLOBAL for _, mode in loads)
    assert loaded == [p for p, _ in loads]


def test_skips_when_cudnn_is_already_loaded_from_another_directory(wheel_dir, loads, monkeypatch):
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: {"/opt/cudnn/lib"})

    assert preload.preload_cudnn() == []
    assert loads == []


def test_proceeds_when_cudnn_is_already_loaded_from_the_wheel(wheel_dir, loads, monkeypatch):
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: {str(wheel_dir)})

    assert len(preload.preload_cudnn()) == len(list(wheel_dir.iterdir()))


def test_no_op_without_the_wheel(monkeypatch, loads):
    monkeypatch.setattr(preload.sys, "platform", "linux")
    monkeypatch.setattr(preload, "_wheel_cudnn_lib_dir", lambda: None)

    assert preload.preload_cudnn() == []
    assert loads == []


def test_no_op_off_linux(wheel_dir, loads, monkeypatch):
    monkeypatch.setattr(preload.sys, "platform", "darwin")

    assert preload.preload_cudnn() == []
    assert loads == []


def test_stops_at_the_first_library_that_fails_to_load(wheel_dir, monkeypatch):
    calls = []

    def failing_cdll(path, mode=0):
        calls.append(path)
        if path.endswith("libcudnn_graph.so.9"):
            raise OSError("cannot open shared object file")
        return object()

    monkeypatch.setattr(preload.ctypes, "CDLL", failing_cdll)

    loaded = preload.preload_cudnn()

    assert [p.split("/")[-1] for p in loaded] == ["libcudnn.so.9"]
    assert len(calls) == 2
