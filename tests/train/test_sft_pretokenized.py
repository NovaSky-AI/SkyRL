"""
CPU tests for pretokenized SFT dataset ingestion (local + object store paths).

uv run --extra dev pytest tests/train/test_sft_pretokenized.py -v
"""

import os
import shutil
from types import SimpleNamespace

import pytest
from datasets import Dataset

from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.config.sft_config import SFTConfig, validate_sft_cfg
from skyrl.train.dataset.pretokenized import load_pretokenized_dataset
from skyrl.train.sft_trainer import SFTTrainer, collate_sft_batch


def _native_rows():
    """Rows in SkyRL's native tokenized format (num_actions + window loss_mask)."""
    return [
        {"input_ids": [1, 2, 3, 4, 5], "num_actions": 2, "loss_mask": [1, 1]},
        {"input_ids": [6, 7, 8], "num_actions": 1, "loss_mask": [1]},
    ]


def _assert_normalized(rows):
    for row in rows:
        assert row["attention_mask"] == [1] * len(row["input_ids"])
        assert 0 < row["num_actions"] <= len(row["input_ids"])
        assert len(row["loss_mask"]) == row["num_actions"]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_load_parquet_file(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_native_rows()).to_parquet(path)

    rows = load_pretokenized_dataset(path)
    assert len(rows) == 2
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["num_actions"] == 2
    _assert_normalized(rows)


def test_load_jsonl_file(tmp_path):
    path = str(tmp_path / "data.jsonl")
    Dataset.from_list(_native_rows()).to_json(path)

    rows = load_pretokenized_dataset(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_save_to_disk_dir(tmp_path):
    path = str(tmp_path / "hf_dataset")
    Dataset.from_list(_native_rows()).save_to_disk(path)

    rows = load_pretokenized_dataset(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_arrow_file(tmp_path):
    saved = tmp_path / "hf_dataset"
    Dataset.from_list(_native_rows()).save_to_disk(str(saved))
    arrow_files = [f for f in os.listdir(saved) if f.endswith(".arrow")]
    assert arrow_files
    path = str(tmp_path / "data.arrow")
    shutil.copy(saved / arrow_files[0], path)

    rows = load_pretokenized_dataset(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_directory_of_parquet_shards(tmp_path):
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    Dataset.from_list(_native_rows()).to_parquet(str(data_dir / "shard-00000.parquet"))
    Dataset.from_list(_native_rows()).to_parquet(str(data_dir / "shard-00001.parquet"))

    rows = load_pretokenized_dataset(str(data_dir))
    assert len(rows) == 4


def test_mixed_formats_in_directory_raises(tmp_path):
    data_dir = tmp_path / "mixed"
    data_dir.mkdir()
    Dataset.from_list(_native_rows()).to_parquet(str(data_dir / "a.parquet"))
    Dataset.from_list(_native_rows()).to_json(str(data_dir / "b.jsonl"))

    with pytest.raises(ValueError, match="mix of data formats"):
        load_pretokenized_dataset(str(data_dir))


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No supported data files"):
        load_pretokenized_dataset(str(tmp_path))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pretokenized_dataset(str(tmp_path / "nope.parquet"))


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------


def test_num_actions_only_defaults_loss_mask(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "num_actions": 3}]).to_parquet(path)

    rows = load_pretokenized_dataset(path)
    assert rows[0]["loss_mask"] == [1, 1, 1]
    assert rows[0]["num_actions"] == 3


def test_full_sequence_loss_mask(tmp_path):
    path = str(tmp_path / "data.parquet")
    # Interior zero after the first 1 (multi-turn style) is preserved.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 1, 0, 1]}]).to_parquet(path)

    rows = load_pretokenized_dataset(path)
    assert rows[0]["num_actions"] == 3
    assert rows[0]["loss_mask"] == [1, 0, 1]


def test_labels_schema(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}]).to_parquet(path)

    rows = load_pretokenized_dataset(path)
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]


def test_num_actions_with_full_length_loss_mask(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "num_actions": 2, "loss_mask": [0, 0, 1, 1]}]).to_parquet(path)

    rows = load_pretokenized_dataset(path)
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]


def test_all_zero_loss_mask_row_dropped(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]},
            {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]},
        ]
    ).to_parquet(path)

    rows = load_pretokenized_dataset(path)
    assert len(rows) == 1


def test_all_rows_dropped_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]}]).to_parquet(path)

    with pytest.raises(ValueError, match="0 usable examples"):
        load_pretokenized_dataset(path)


def test_missing_loss_target_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3]}]).to_parquet(path)

    with pytest.raises(ValueError, match="no loss target"):
        load_pretokenized_dataset(path)


def test_missing_input_ids_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"num_actions": 1, "other": "x"}]).to_parquet(path)

    with pytest.raises(ValueError, match="input_ids"):
        load_pretokenized_dataset(path)


def test_padded_attention_mask_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [0, 0, 1, 2], "attention_mask": [0, 0, 1, 1], "num_actions": 1}]).to_parquet(path)

    with pytest.raises(ValueError, match="unpadded"):
        load_pretokenized_dataset(path)


def test_invalid_num_actions_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "num_actions": 5}]).to_parquet(path)

    with pytest.raises(ValueError, match="num_actions"):
        load_pretokenized_dataset(path)


# ---------------------------------------------------------------------------
# max_length truncation
# ---------------------------------------------------------------------------


def test_max_length_truncates_action_window(tmp_path):
    path = str(tmp_path / "data.parquet")
    # 3 prompt tokens + 4 response tokens; max_length=5 keeps 2 response tokens.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5, 6, 7], "num_actions": 4}]).to_parquet(path)

    rows = load_pretokenized_dataset(path, max_length=5)
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]


def test_max_length_drops_fully_truncated_response(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            # 4 prompt tokens; truncation to 3 removes the whole response.
            {"input_ids": [1, 2, 3, 4, 5], "num_actions": 1},
            {"input_ids": [1, 2], "num_actions": 1},
        ]
    ).to_parquet(path)

    rows = load_pretokenized_dataset(path, max_length=3)
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# Cloud (S3) paths
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_s3(tmp_path, monkeypatch):
    """Route the io module's cloud calls at a local directory acting as S3."""
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    Dataset.from_list(_native_rows()).to_parquet(str(remote_dir / "data.parquet"))
    calls = {"downloads": 0}

    def fake_isdir(path):
        assert path.startswith("s3://")
        return True

    def fake_download_directory(path, local_path):
        assert path.startswith("s3://")
        calls["downloads"] += 1
        shutil.copytree(remote_dir, local_path, dirs_exist_ok=True)

    monkeypatch.setattr(io, "isdir", fake_isdir)
    monkeypatch.setattr(io, "download_directory", fake_download_directory)
    return calls


def test_s3_path_download(tmp_path, fake_s3):
    rows = load_pretokenized_dataset("s3://bucket/prefix/data", cache_dir=None)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 1
    _assert_normalized(rows)


def test_s3_download_cached_across_calls(tmp_path, fake_s3):
    cache_dir = str(tmp_path / "cache")
    rows = load_pretokenized_dataset("s3://bucket/prefix/data", cache_dir=cache_dir)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 1

    # Second load reuses the cached download.
    rows = load_pretokenized_dataset("s3://bucket/prefix/data", cache_dir=cache_dir)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 1

    # force_redownload bypasses the cache.
    rows = load_pretokenized_dataset("s3://bucket/prefix/data", cache_dir=cache_dir, force_redownload=True)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 2


def test_s3_single_file_download(tmp_path, monkeypatch):
    remote_file = tmp_path / "data.parquet"
    Dataset.from_list(_native_rows()).to_parquet(str(remote_file))

    monkeypatch.setattr(io, "isdir", lambda path: False)
    monkeypatch.setattr(io, "download_file", lambda path, local: shutil.copy(remote_file, local))

    rows = load_pretokenized_dataset("s3://bucket/data.parquet", cache_dir=None)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Trainer / config integration
# ---------------------------------------------------------------------------


def test_trainer_load_dataset_routes_to_pretokenized(tmp_path):
    train_path = str(tmp_path / "train.parquet")
    eval_path = str(tmp_path / "eval.parquet")
    Dataset.from_list(_native_rows()).to_parquet(train_path)
    Dataset.from_list(_native_rows()[:1]).to_parquet(eval_path)

    cfg = SFTConfig(
        pretokenized_dataset_path=train_path,
        eval_pretokenized_dataset_path=eval_path,
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    trainer = SFTTrainer(cfg)

    # No tokenizer / workers needed: the pretokenized path never tokenizes.
    tokenized = trainer.load_dataset()
    assert len(tokenized) == 2
    eval_tokenized = trainer.load_eval_dataset()
    assert len(eval_tokenized) == 1


def test_pretokenized_rows_collate(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_native_rows()).to_parquet(path)
    rows = load_pretokenized_dataset(path)

    batch = collate_sft_batch(rows, SimpleNamespace(pad_token_id=0))
    assert batch["sequences"].shape == (2, 5)
    assert batch["loss_mask"].shape == (2, 2)
    assert batch.metadata["response_length"] == 2


def test_validate_cfg_accepts_pretokenized_eval_only():
    cfg = SFTConfig(
        pretokenized_dataset_path="/data/train",
        eval_pretokenized_dataset_path="/data/eval",
        eval_interval=10,
    )
    validate_sft_cfg(cfg)


def test_validate_cfg_eval_interval_requires_some_eval_dataset():
    cfg = SFTConfig(eval_interval=10)
    with pytest.raises(ValueError, match="eval_interval"):
        validate_sft_cfg(cfg)
