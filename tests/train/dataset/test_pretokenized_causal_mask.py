"""Pretokenized loss windows must contain only next-token targets."""

import pytest
from datasets import Dataset

from skyrl.train.dataset.pretokenized import load_from_pretokenized


@pytest.mark.parametrize("file_format", ["parquet", "jsonl", "arrow"])
@pytest.mark.parametrize(
    "mask, expected_actions, expected_mask",
    [
        ([1, 1, 1], 2, [1, 1]),
        ([1, 0, 1], 1, [1]),
        ([0, 1, 1], 2, [1, 1]),
        ([0, 0, 1], 1, [1]),
    ],
)
def test_loss_window_excludes_first_token(tmp_path, file_format, mask, expected_actions, expected_mask):
    path = tmp_path / f"data.{file_format}"
    data = Dataset.from_list([{"input_ids": [10, 20, 30], "loss_mask": mask}])
    if file_format == "parquet":
        data.to_parquet(str(path))
    elif file_format == "jsonl":
        data.to_json(str(path))
    else:
        data.save_to_disk(str(path))

    dataset = load_from_pretokenized(str(path))
    assert len(dataset) == 1
    row = dataset[0]
    assert row["input_ids"] == [10, 20, 30]
    assert row["attention_mask"] == [1, 1, 1]
    assert row["num_actions"] == expected_actions
    assert row["loss_mask"] == expected_mask
    assert dataset.__getitems__([0]) == [row]


@pytest.mark.parametrize("max_length", [None, 2, 3])
def test_first_token_only_rows_are_dropped(tmp_path, max_length):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            {"input_ids": [10], "loss_mask": [1]},
            {"input_ids": [10, 20, 30], "loss_mask": [1, 0, 0]},
            {"input_ids": [], "loss_mask": []},
            {"input_ids": [40, 50], "loss_mask": [0, 1]},
        ]
    ).to_parquet(path)

    dataset = load_from_pretokenized(path, max_length=max_length)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [40, 50]
    assert dataset[0]["num_actions"] == 1
    assert dataset[0]["loss_mask"] == [1]


@pytest.mark.parametrize("max_length", [1, 2, 3])
def test_truncated_loss_window_excludes_first_token(tmp_path, max_length):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [10, 20, 30], "loss_mask": [1, 1, 1]}]).to_parquet(path)

    if max_length == 1:
        with pytest.raises(ValueError, match="0 usable examples"):
            load_from_pretokenized(path, max_length=max_length)
    else:
        row = load_from_pretokenized(path, max_length=max_length)[0]
        assert row["input_ids"] == [10, 20, 30][:max_length]
        assert row["num_actions"] == max_length - 1
        assert row["loss_mask"] == [1] * (max_length - 1)
