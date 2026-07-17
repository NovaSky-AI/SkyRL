"""
CPU tests for pretokenized SFT dataset ingestion (local + object store paths).

uv run --extra dev pytest tests/train/test_sft_pretokenized.py -v
"""

import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.config.sft_config import SFTConfig, validate_sft_cfg
from skyrl.train.dataset.pretokenized import load_from_pretokenized
from skyrl.train.sft_trainer import SFTTrainer, collate_sft_batch


def _rows():
    """Rows in the pretokenized input schema: input_ids + full-sequence loss_mask."""
    return [
        # Instruction-following style: loss on the trailing response tokens.
        {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 0, 1, 1]},
        # Single-token response.
        {"input_ids": [6, 7, 8], "loss_mask": [0, 0, 1]},
    ]


def _vlm_rows():
    """VLM rows: image tensors stored as nested lists (parquet-compatible)."""
    return [
        {
            "input_ids": [1, 2, 3, 4],
            "loss_mask": [0, 0, 1, 1],
            "pixel_values": [[0.1] * 8] * 4,
            "image_grid_thw": [[1, 2, 2]],
        },
        {
            "input_ids": [5, 6, 7],
            "loss_mask": [0, 1, 1],
            "pixel_values": [[0.2] * 8] * 4,
            "image_grid_thw": [[1, 2, 2]],
        },
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
    Dataset.from_list(_rows()).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]
    _assert_normalized(rows)


def test_load_jsonl_file(tmp_path):
    path = str(tmp_path / "data.jsonl")
    Dataset.from_list(_rows()).to_json(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_save_to_disk_dir(tmp_path):
    path = str(tmp_path / "hf_dataset")
    Dataset.from_list(_rows()).save_to_disk(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_arrow_file(tmp_path):
    saved = tmp_path / "hf_dataset"
    Dataset.from_list(_rows()).save_to_disk(str(saved))
    arrow_files = [f for f in os.listdir(saved) if f.endswith(".arrow")]
    assert arrow_files
    path = str(tmp_path / "data.arrow")
    shutil.copy(saved / arrow_files[0], path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_directory_of_parquet_shards(tmp_path):
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "shard-00000.parquet"))
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "shard-00001.parquet"))

    rows = load_from_pretokenized(str(data_dir))
    assert len(rows) == 4


def test_mixed_formats_in_directory_raises(tmp_path):
    data_dir = tmp_path / "mixed"
    data_dir.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "a.parquet"))
    Dataset.from_list(_rows()).to_json(str(data_dir / "b.jsonl"))

    with pytest.raises(ValueError, match="mix of data formats"):
        load_from_pretokenized(str(data_dir))


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No supported data files"):
        load_from_pretokenized(str(tmp_path))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_from_pretokenized(str(tmp_path / "nope.parquet"))


# ---------------------------------------------------------------------------
# Schema validation / normalization
# ---------------------------------------------------------------------------


def test_multi_turn_loss_mask_preserves_interior_zeros(tmp_path):
    path = str(tmp_path / "data.parquet")
    # Conversational data: assistant turns at positions 2 and 4, user turn between.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 1, 0, 1]}]).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert rows[0]["num_actions"] == 3
    assert rows[0]["loss_mask"] == [1, 0, 1]


def test_all_zero_loss_mask_row_dropped(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]},
            {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]},
        ]
    ).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 1


def test_all_rows_dropped_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]}]).to_parquet(path)

    with pytest.raises(ValueError, match="0 usable examples"):
        load_from_pretokenized(path)


def test_missing_loss_mask_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3]}]).to_parquet(path)

    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        load_from_pretokenized(path)


def test_labels_column_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    # HF-style labels are not a supported loss target; loss_mask is required.
    Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]).to_parquet(path)

    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        load_from_pretokenized(path)


def test_num_actions_column_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    # num_actions alone is not a loss target; the full-sequence loss_mask is required.
    Dataset.from_list([{"input_ids": [1, 2, 3], "num_actions": 2}]).to_parquet(path)

    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        load_from_pretokenized(path)


def test_window_form_loss_mask_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    # Window-form mask (len == 2 != len(input_ids) == 4) is no longer accepted.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "loss_mask": [1, 1]}]).to_parquet(path)

    with pytest.raises(ValueError, match="full-sequence"):
        load_from_pretokenized(path)


def test_non_binary_loss_mask_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 2, 1]}]).to_parquet(path)

    with pytest.raises(ValueError, match="only 0s and 1s"):
        load_from_pretokenized(path)


def test_redundant_num_actions_dropped_when_loss_mask_present(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "num_actions": 3}]).to_parquet(path)

    rows = load_from_pretokenized(path)
    # num_actions is always re-inferred from the mask, never read from the store.
    assert rows[0]["num_actions"] == 2


def test_missing_input_ids_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"loss_mask": [1], "other": "x"}]).to_parquet(path)

    with pytest.raises(ValueError, match="input_ids"):
        load_from_pretokenized(path)


def test_padded_attention_mask_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [{"input_ids": [0, 0, 1, 2], "attention_mask": [0, 0, 1, 1], "loss_mask": [0, 0, 0, 1]}]
    ).to_parquet(path)

    with pytest.raises(ValueError, match="unpadded"):
        load_from_pretokenized(path)


# ---------------------------------------------------------------------------
# max_length truncation
# ---------------------------------------------------------------------------


def test_max_length_truncates_action_window(tmp_path):
    path = str(tmp_path / "data.parquet")
    # 3 prompt tokens + 4 response tokens; max_length=5 keeps 2 response tokens.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5, 6, 7], "loss_mask": [0, 0, 0, 1, 1, 1, 1]}]).to_parquet(path)

    rows = load_from_pretokenized(path, max_length=5)
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]


def test_max_length_drops_fully_truncated_response(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            # 4 prompt tokens; truncation to 3 removes the whole response.
            {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 0, 0, 1]},
            {"input_ids": [1, 2], "loss_mask": [0, 1]},
        ]
    ).to_parquet(path)

    rows = load_from_pretokenized(path, max_length=3)
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# VLM rows
# ---------------------------------------------------------------------------


def test_vlm_rows_pass_through(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_vlm_rows()).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)
    assert rows[0]["pixel_values"] == [[0.1] * 8] * 4
    assert rows[0]["image_grid_thw"] == [[1, 2, 2]]


def test_vlm_rows_collate_to_tensor_lists(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_vlm_rows()).to_parquet(path)
    rows = load_from_pretokenized(path)

    batch = collate_sft_batch(rows, SimpleNamespace(pad_token_id=0))
    assert batch["sequences"].shape == (2, 4)
    assert len(batch["pixel_values"]) == 2
    assert torch.as_tensor(rows[0]["pixel_values"]).shape == batch["pixel_values"][0].shape
    assert batch["image_grid_thw"][0].tolist() == [[1, 2, 2]]


def test_vlm_row_over_max_length_dropped_not_truncated(tmp_path):
    path = str(tmp_path / "data.parquet")
    long_vlm = {
        "input_ids": [1, 2, 3, 4, 5, 6],
        "loss_mask": [0, 0, 0, 1, 1, 1],
        "pixel_values": [[0.1] * 8] * 4,
        "image_grid_thw": [[1, 2, 2]],
    }
    Dataset.from_list([long_vlm] + _vlm_rows()).to_parquet(path)

    rows = load_from_pretokenized(path, max_length=4)
    # The long VLM row is dropped (never truncated); the short ones survive intact.
    assert len(rows) == 2
    assert all(len(r["input_ids"]) <= 4 for r in rows)


def test_vlm_row_missing_grid_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "pixel_values": [[0.1] * 8] * 4}]).to_parquet(
        path
    )

    with pytest.raises(ValueError, match="both"):
        load_from_pretokenized(path)


def test_mixed_text_and_vlm_store_drops_null_image_columns(tmp_path):
    path = str(tmp_path / "data.parquet")
    # A store mixing text and VLM rows materializes the image columns as None
    # on text rows; those keys must not leak into the normalized text rows.
    Dataset.from_list(_vlm_rows() + [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}]).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 3
    text_rows = [r for r in rows if "pixel_values" not in r]
    assert len(text_rows) == 1


# ---------------------------------------------------------------------------
# Cloud (S3) paths
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_s3(tmp_path, monkeypatch):
    """Route the io module's cloud calls at a local directory acting as S3."""
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(remote_dir / "data.parquet"))
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
    rows = load_from_pretokenized("s3://bucket/prefix/data", cache_dir=None)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 1
    _assert_normalized(rows)


def test_s3_download_cached_across_calls(tmp_path, fake_s3):
    cache_dir = str(tmp_path / "cache")
    rows = load_from_pretokenized("s3://bucket/prefix/data", cache_dir=cache_dir)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 1

    # Second load reuses the cached download.
    rows = load_from_pretokenized("s3://bucket/prefix/data", cache_dir=cache_dir)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 1

    # force_redownload bypasses the cache.
    rows = load_from_pretokenized("s3://bucket/prefix/data", cache_dir=cache_dir, force_redownload=True)
    assert len(rows) == 2
    assert fake_s3["downloads"] == 2


def test_s3_single_file_download(tmp_path, monkeypatch):
    remote_file = tmp_path / "data.parquet"
    Dataset.from_list(_rows()).to_parquet(str(remote_file))

    monkeypatch.setattr(io, "isdir", lambda path: False)
    monkeypatch.setattr(io, "download_file", lambda path, local: shutil.copy(remote_file, local))

    rows = load_from_pretokenized("s3://bucket/data.parquet", cache_dir=None)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Trainer / config integration
# ---------------------------------------------------------------------------


def test_trainer_load_dataset_routes_to_pretokenized(tmp_path):
    train_path = str(tmp_path / "train.parquet")
    eval_path = str(tmp_path / "eval.parquet")
    Dataset.from_list(_rows()).to_parquet(train_path)
    Dataset.from_list(_rows()[:1]).to_parquet(eval_path)

    cfg = SFTConfig(
        pretokenized_dataset_paths=[train_path],
        eval_pretokenized_dataset_paths=[eval_path],
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    trainer = SFTTrainer(cfg)

    # No tokenizer / workers needed: the pretokenized path never tokenizes.
    tokenized, dataset_lengths = trainer.load_dataset()
    assert len(tokenized) == 2
    assert dataset_lengths == [2]
    eval_sets = trainer.load_eval_datasets()
    assert len(eval_sets) == 1
    eval_name, eval_tokenized = eval_sets[0]
    assert eval_name == "eval.parquet"
    assert len(eval_tokenized) == 1


def test_trainer_concatenates_multiple_pretokenized_stores(tmp_path):
    path_a = str(tmp_path / "store_a.parquet")
    path_b = str(tmp_path / "store_b.parquet")
    Dataset.from_list(_rows()).to_parquet(path_a)
    Dataset.from_list(_rows()[:1]).to_parquet(path_b)

    cfg = SFTConfig(
        pretokenized_dataset_paths=[path_a, path_b],
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    # Equal mixing weights are defaulted for the random sampler.
    assert cfg.train_dataset_weights == [0.5, 0.5]

    trainer = SFTTrainer(cfg)
    tokenized, dataset_lengths = trainer.load_dataset()
    assert len(tokenized) == 3
    assert dataset_lengths == [2, 1]

    # Multiple sources -> DataMixingSampler configured from the store lengths.
    sampler = trainer.build_train_sampler(tokenized, dataset_lengths)
    assert sampler is not None
    assert len(list(iter(sampler))) > 0


def test_multiple_pretokenized_eval_stores_named_by_basename(tmp_path):
    path_a = str(tmp_path / "eval_a.parquet")
    path_b = str(tmp_path / "eval_b.parquet")
    Dataset.from_list(_rows()).to_parquet(path_a)
    Dataset.from_list(_rows()[:1]).to_parquet(path_b)

    cfg = SFTConfig(
        pretokenized_dataset_paths=[path_a],
        eval_pretokenized_dataset_paths=[path_a, path_b],
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    assert cfg.eval_dataset_names == ["eval_a.parquet", "eval_b.parquet"]

    trainer = SFTTrainer(cfg)
    eval_sets = trainer.load_eval_datasets()
    assert [(name, len(rows)) for name, rows in eval_sets] == [("eval_a.parquet", 2), ("eval_b.parquet", 1)]


def test_validate_cfg_rejects_pretokenized_with_train_datasets():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        train_datasets=["yahma/alpaca-cleaned"],
        train_dataset_splits=["train[:100]"],
    )
    with pytest.raises(ValueError, match="only one of pretokenized_dataset_paths"):
        validate_sft_cfg(cfg)


def test_validate_cfg_rejects_colliding_default_eval_names():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        eval_pretokenized_dataset_paths=["/data/a/eval", "/data/b/eval"],
    )
    with pytest.raises(ValueError, match="collide"):
        validate_sft_cfg(cfg)

    # Explicit names disambiguate.
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        eval_pretokenized_dataset_paths=["/data/a/eval", "/data/b/eval"],
        eval_dataset_names=["eval_a", "eval_b"],
    )
    validate_sft_cfg(cfg)


def test_validate_cfg_weights_validated_against_pretokenized_paths():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/a", "/data/b"],
        train_dataset_weights=[0.9],
    )
    with pytest.raises(ValueError, match="one weight per entry of pretokenized_dataset_paths"):
        validate_sft_cfg(cfg)

    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/a", "/data/b"],
        train_dataset_weights=[0.9, 0.1],
    )
    validate_sft_cfg(cfg)
    assert cfg.train_dataset_weights == [0.9, 0.1]


def test_pretokenized_rows_collate(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_rows()).to_parquet(path)
    rows = load_from_pretokenized(path)

    batch = collate_sft_batch(rows, SimpleNamespace(pad_token_id=0))
    assert batch["sequences"].shape == (2, 5)
    assert batch["loss_mask"].shape == (2, 2)
    assert batch.metadata["response_length"] == 2


def test_validate_cfg_accepts_pretokenized_eval_only():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        eval_pretokenized_dataset_paths=["/data/eval"],
        eval_interval=10,
    )
    validate_sft_cfg(cfg)


def test_validate_cfg_eval_interval_requires_some_eval_dataset():
    cfg = SFTConfig(eval_interval=10)
    with pytest.raises(ValueError, match="eval_interval"):
        validate_sft_cfg(cfg)
