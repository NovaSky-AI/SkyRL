"""Tests for storage utilities."""

from pathlib import Path

from tx.utils.storage import pack_and_upload


def test_pack_and_upload_skips_write_for_non_rank_0_on_shared_fs(tmp_path: Path):
    """Non-rank-0 skips writing when probe file exists (shared filesystem)."""
    output_path = tmp_path / "checkpoint.tar.gz"
    (tmp_path / "checkpoint.tar.gz.probe").write_text("write_probe")

    with pack_and_upload(output_path, rank=1) as temp_dir:
        (temp_dir / "test.txt").write_text("test")

    assert not output_path.exists()


def test_pack_and_upload_writes_when_no_probe(tmp_path: Path):
    """All ranks write when no probe file exists (local disk)."""
    output_path = tmp_path / "checkpoint.tar.gz"

    with pack_and_upload(output_path, rank=1) as temp_dir:
        (temp_dir / "test.txt").write_text("test")

    assert output_path.exists()
