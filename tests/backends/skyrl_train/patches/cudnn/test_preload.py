"""CPU tests for preloading cuDNN's sublibraries from the ``nvidia-cudnn`` wheel.

Empty files stand in for the wheel's and the system's ``libcudnn*.so.9``, and ``ctypes.CDLL`` is replaced by a
recorder, so the tests check which libraries would be loaded, in which order, and when nothing is loaded.
"""

import ctypes
import os

import pytest

from skyrl.backends.skyrl_train.patches.cudnn import preload

_WHEEL_LIBS = (
    "libcudnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_ext.so.9",
)


def _make_libs(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


@pytest.fixture
def wheel_dir(tmp_path, monkeypatch):
    lib = _make_libs(tmp_path / "venv" / "nvidia" / "cudnn" / "lib", _WHEEL_LIBS)
    monkeypatch.setattr(preload, "_is_linux", lambda: True)
    monkeypatch.setattr(preload, "_wheel_cudnn_lib_dir", lambda: str(lib))
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: set())
    return lib


@pytest.fixture
def system_cudnn(tmp_path, monkeypatch):
    """An older cuDNN on LD_LIBRARY_PATH, as the Megatron image's /opt/cudnn/lib."""
    system = _make_libs(tmp_path / "opt" / "cudnn" / "lib", ("libcudnn.so.9", "libcudnn_engines_precompiled.so.9"))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/usr/local/nvidia/lib64:{system}")
    return system


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def fake_cdll(path, mode=0):
        calls.append((path, mode))
        return object()

    monkeypatch.setattr(preload.ctypes, "CDLL", fake_cdll)
    return calls


def test_loads_every_wheel_library_globally_in_dependency_order(wheel_dir, system_cudnn, loads):
    loaded = preload.preload_cudnn()

    names = [os.path.basename(p) for p, _ in loads]
    assert names[:3] == ["libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_engines_precompiled.so.9"]
    assert sorted(names) == sorted(_WHEEL_LIBS)
    assert all(os.path.dirname(p) == str(wheel_dir) for p, _ in loads)
    assert all(mode == ctypes.RTLD_GLOBAL for _, mode in loads)
    assert loaded == [p for p, _ in loads]


def test_no_op_without_another_cudnn_on_ld_library_path(wheel_dir, loads, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64:/usr/local/cuda/lib64")

    assert preload.preload_cudnn() == []
    assert loads == []


def test_wheel_dir_on_ld_library_path_is_not_foreign(wheel_dir, loads, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", str(wheel_dir))

    assert preload.preload_cudnn() == []


def test_symlinked_wheel_files_count_as_the_wheel(tmp_path, monkeypatch, loads):
    """Package caches can install the wheel's files as symlinks into a shared cache directory."""
    cache = _make_libs(tmp_path / "cache" / "nvidia_cudnn" / "lib", _WHEEL_LIBS)
    venv_lib = tmp_path / "venv" / "nvidia" / "cudnn" / "lib"
    venv_lib.mkdir(parents=True)
    for name in _WHEEL_LIBS:
        (venv_lib / name).symlink_to(cache / name)
    monkeypatch.setattr(preload, "_is_linux", lambda: True)
    monkeypatch.setattr(preload, "_wheel_cudnn_lib_dir", lambda: str(venv_lib))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(cache))
    # torch already loaded libcudnn.so.9 from the wheel; /proc/self/maps shows the resolved cache path.
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: {str(cache.resolve())})

    assert preload.preload_cudnn() == []  # the "foreign" LD_LIBRARY_PATH entry is the wheel itself


def test_skips_when_cudnn_is_already_loaded_from_another_directory(wheel_dir, system_cudnn, loads, monkeypatch):
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: {str(system_cudnn)})

    assert preload.preload_cudnn() == []
    assert loads == []


def test_proceeds_when_cudnn_is_already_loaded_from_the_wheel(wheel_dir, system_cudnn, loads, monkeypatch):
    monkeypatch.setattr(preload, "_loaded_cudnn_dirs", lambda: {os.path.realpath(wheel_dir)})

    assert len(preload.preload_cudnn()) == len(_WHEEL_LIBS)


def test_no_op_without_the_wheel(system_cudnn, loads, monkeypatch):
    monkeypatch.setattr(preload, "_is_linux", lambda: True)
    monkeypatch.setattr(preload, "_wheel_cudnn_lib_dir", lambda: None)

    assert preload.preload_cudnn() == []
    assert loads == []


def test_no_op_off_linux(wheel_dir, system_cudnn, loads, monkeypatch):
    monkeypatch.setattr(preload, "_is_linux", lambda: False)

    assert preload.preload_cudnn() == []
    assert loads == []


def test_a_library_that_fails_to_load_is_skipped(wheel_dir, system_cudnn, monkeypatch):
    def failing_cdll(path, mode=0):
        if path.endswith("libcudnn_graph.so.9"):
            raise OSError("cannot open shared object file")
        return object()

    monkeypatch.setattr(preload.ctypes, "CDLL", failing_cdll)

    loaded = [os.path.basename(p) for p in preload.preload_cudnn()]

    assert "libcudnn_graph.so.9" not in loaded
    assert len(loaded) == len(_WHEEL_LIBS) - 1


def test_loaded_cudnn_dirs_parses_proc_maps(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f10 r-xp 00000000 08:01 1 /opt/cudnn/lib/libcudnn_engines_precompiled.so.9\n"
        "7f10-7f20 r-xp 00000000 08:01 2 /venv/nvidia/cudnn/lib/libcudnn.so.9\n"
        "7f20-7f30 r-xp 00000000 08:01 3 /venv/torch/lib/libtorch_cuda.so\n"
        "7f30-7f40 rw-p 00000000 00:00 0 [heap]\n"
    )

    assert preload._loaded_cudnn_dirs(str(maps)) == {"/opt/cudnn/lib", "/venv/nvidia/cudnn/lib"}
    assert preload._loaded_cudnn_dirs(str(tmp_path / "missing")) == set()
