import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset
from skyrl_train.dataset import PromptDataset


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda x, add_generation_prompt: x
    return tokenizer


@pytest.fixture
def sample_dataset():
    # 3 samples: one too long, two valid
    data = {
        "prompt": [
            "short prompt",  # length 13
            "a" * 120,  # length 120
            "b" * 200,  # length 200 (to be filtered out if max len < 200)
        ],
        "answer": ["a1", "a2", "a3"],
    }
    return Dataset.from_dict(data)


@patch("datasets.load_dataset")
def test_prompt_dataset_filtering(mock_load_dataset, mock_tokenizer, sample_dataset):
    mock_load_dataset.return_value = {"train": sample_dataset}

    dataset = PromptDataset(
        datasets=["dummy1.parquet"],
        tokenizer=mock_tokenizer,
        max_prompt_length=150,  # should exclude third item
        num_workers=1,
        prompt_key="prompt",
        env_class_key="env_class",
    )

    # Only first two prompts should remain
    assert len(dataset) == 2
    messages, env, extra, uid = dataset[0]
    assert env is None
    assert messages == "short prompt"
    assert extra == {"answer": "a1"}


def test_collate_fn():
    dataset = PromptDataset.__new__(PromptDataset)  # Bypass __init__
    sample_data = [("prompt 1", "env", {"answer": "a1"}, "1"), ("prompt 2", "env", {"answer": "a2"}, "2")]
    expected = [
        {"prompt": "prompt 1", "env_class": "env", "env_extras": {"answer": "a1"}, "uid": "1"},
        {"prompt": "prompt 2", "env_class": "env", "env_extras": {"answer": "a2"}, "uid": "2"},
    ]

    output = dataset.collate_fn(sample_data)
    assert output == expected


@patch("datasets.load_dataset")
def test_prompt_dataset_hf_name_defaults_to_train(mock_load_dataset, mock_tokenizer, sample_dataset):
    # When only a dataset name is provided, we default to the 'train' split
    mock_load_dataset.return_value = {"train": sample_dataset}

    dataset = PromptDataset(
        datasets=["my_hf_dataset"],
        tokenizer=mock_tokenizer,
        max_prompt_length=150,
        num_workers=1,
        prompt_key="prompt",
        env_class_key="env_class",
    )

    assert len(dataset) == 2
    messages, env, extra, uid = dataset[1]
    assert messages == "a" * 120
    assert extra == {"answer": "a2"}


@patch("datasets.load_dataset")
def test_prompt_dataset_hf_name_with_split(mock_load_dataset, mock_tokenizer, sample_dataset):
    # When a dataset name with split is provided, respect the split
    mock_load_dataset.return_value = {"validation": sample_dataset}

    dataset = PromptDataset(
        datasets=["my_hf_dataset:validation"],
        tokenizer=mock_tokenizer,
        max_prompt_length=150,
        num_workers=1,
        prompt_key="prompt",
        env_class_key="env_class",
    )

    assert len(dataset) == 2
    messages, env, extra, uid = dataset[0]
    assert messages == "short prompt"
    assert extra == {"answer": "a1"}


@patch("datasets.load_dataset")
def test_prompt_dataset_hf_missing_train_raises(mock_load_dataset, mock_tokenizer, sample_dataset):
    # No split provided and 'train' not available
    mock_load_dataset.return_value = {"validation": sample_dataset}

    with pytest.raises(ValueError, match=r"Split `train` not found"):
        PromptDataset(
            datasets=["my_hf_dataset"],
            tokenizer=mock_tokenizer,
            max_prompt_length=150,
            num_workers=1,
        )


@patch("datasets.load_dataset")
def test_prompt_dataset_hf_invalid_split_raises(mock_load_dataset, mock_tokenizer, sample_dataset):
    # Split provided but does not exist
    mock_load_dataset.return_value = {"train": sample_dataset}

    with pytest.raises(ValueError, match=r"Split `bogus` not found"):
        PromptDataset(
            datasets=["my_hf_dataset:bogus"],
            tokenizer=mock_tokenizer,
            max_prompt_length=150,
            num_workers=1,
        )


def test_prompt_dataset_hf_real_dataset(mock_tokenizer):
    # Real integration test with a small public dataset and known split
    ds = PromptDataset(
        datasets=["HuggingFaceH4/aime_2024:train"],
        tokenizer=mock_tokenizer,
        max_prompt_length=1024,
        num_workers=1,
        prompt_key="problem",
        env_class_key="env_class",
    )
    assert len(ds) > 0


@patch("datasets.load_dataset")
def test_prompt_dataset_uids(mock_load_dataset, sample_dataset, mock_tokenizer):
    # When only a dataset name is provided, we default to the 'train' split
    mock_load_dataset.return_value = {"train": sample_dataset}

    ds = PromptDataset(
        datasets=["my_hf_dataset"],
        tokenizer=mock_tokenizer,
        max_prompt_length=1024,
        num_workers=1,
        prompt_key="prompt",
        env_class_key="env_class",
    )
    rows = [ds[i] for i in range(len(ds))]
    uids = [row[3] for row in rows]

    # UID is simply the row index
    assert uids[0] == "0"
    # UIDs must be unique
    assert len(set(uids)) == len(uids)

    rows_again = [ds[i] for i in range(len(ds))]
    uids_again = [row[3] for row in rows_again]
    # When sampled the second time, UIDs should not change
    assert uids_again == uids
