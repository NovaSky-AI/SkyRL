"""
uv run --isolated --extra dev pytest tests/train/utils/test_fetch_export_once.py
"""

from pathlib import Path

from skyrl.train.utils import utils as train_utils
from skyrl.train.utils.utils import fetch_export_once


def _fake_download(calls):
    def download_directory(uri, local_path):
        calls.append(uri)
        Path(local_path, "model.safetensors").write_bytes(b"weights")

    return download_directory


def test_fetch_export_once_downloads_a_step_once_per_node(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_utils.io, "download_directory", _fake_download(calls))
    cache = tmp_path / "global_step_5"

    first = fetch_export_once("s3://bucket/exports/global_step_5/policy", cache)
    second = fetch_export_once("s3://bucket/exports/global_step_5/policy", cache)  # another worker on the node

    assert first == second == cache
    assert calls == ["s3://bucket/exports/global_step_5/policy"]
    assert (cache / "model.safetensors").read_bytes() == b"weights"
    assert (cache / ".complete").exists()


def test_fetch_export_once_evicts_the_other_steps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_utils.io, "download_directory", _fake_download(calls))
    fetch_export_once("s3://bucket/exports/global_step_5/policy", tmp_path / "global_step_5")

    fetch_export_once("s3://bucket/exports/global_step_10/policy", tmp_path / "global_step_10")

    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["global_step_10"]
    assert len(calls) == 2


def test_fetch_export_once_redownloads_without_the_completion_marker(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_utils.io, "download_directory", _fake_download(calls))
    cache = tmp_path / "global_step_5"
    cache.mkdir()
    (cache / "model.safetensors").write_bytes(b"half-written")  # an earlier attempt that died mid-download

    fetch_export_once("s3://bucket/exports/global_step_5/policy", cache)

    assert calls == ["s3://bucket/exports/global_step_5/policy"]
    assert (cache / "model.safetensors").read_bytes() == b"weights"
