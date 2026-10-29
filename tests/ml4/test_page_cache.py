import os
import sys
import types

import pytest

from skyrl.backends.skyrl_train.utils import page_cache
from skyrl.backends.skyrl_train.utils.page_cache import (
    checkpoint_files,
    drop_page_cache,
)


def test_checkpoint_files_selects_weights_recursively(tmp_path):
    for rel in ("a.safetensors", "sub/b.safetensors", "c.bin", "config.json", "tekken.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 10)
    got = [os.path.relpath(f, os.path.realpath(tmp_path)).replace(os.sep, "/") for f in checkpoint_files(str(tmp_path))]
    assert got == ["a.safetensors", "c.bin", "sub/b.safetensors"]
    assert checkpoint_files(str(tmp_path / "missing")) == []


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="posix_fadvise is POSIX-only")
def test_drop_page_cache_counts_files(tmp_path):
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"x" * 4096)
    assert drop_page_cache([str(f), str(tmp_path / "missing.safetensors")]) == (1, 4096)


@pytest.mark.parametrize(
    "strategy,fs,per_file",
    [(None, "ext4", True), ("lazy", "lustre", True), (None, "lustre", False), ("prefetch", "xfs", False)],
    ids=["local-default", "lazy-on-lustre", "auto-prefetch-on-lustre", "explicit-prefetch"],
)
def test_per_file_drop_drops_each_file_after_its_tensors(monkeypatch, strategy, fs, per_file):
    wu = types.SimpleNamespace(_natural_sort_key=lambda f: int(f.split("-")[1]), _get_fs_type=lambda files: fs)
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.weight_utils", wu)
    sys.modules["vllm.model_executor.model_loader"].weight_utils = wu
    events = []
    monkeypatch.setattr(page_cache, "drop_page_cache", lambda paths: events.append(("drop", *paths)) or (len(paths), 0))

    def iterator(files, use_tqdm, strategy=None, *args, **kwargs):
        for f in sorted(files, key=wu._natural_sort_key):
            for k in range(2):
                events.append(("yield", f))
                yield f"{f}.t{k}", k

    files = ["ckpt-10-x", "ckpt-2-x", "ckpt-1-x"]
    got = list(page_cache.per_file_drop(iterator)(files, True, strategy))
    order = ["ckpt-1-x", "ckpt-2-x", "ckpt-10-x"]
    assert got == [(f"{f}.t{k}", k) for f in order for k in range(2)]
    if per_file:
        # a file is dropped only after the first tensor of the next file has been handed out (the
        # consumer holds the previous tensor until then), again for the next files, and at the end
        y = lambda f: ("yield", f)  # noqa: E731
        assert events == [
            y(order[0]),
            y(order[0]),
            y(order[1]),
            ("drop", order[0]),
            y(order[1]),
            y(order[2]),
            ("drop", *order[:2]),
            y(order[2]),
            ("drop", *order),
        ]
    else:
        assert all(e[0] == "yield" for e in events)
