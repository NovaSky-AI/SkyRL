"""
CPU tests for ``lazy_vlm_dataset``: VLM chat datasets tokenized per fetch
(:class:`~skyrl.train.dataset.sft_dataset.LazyVLMDataset`) instead of on load.

uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_lazy_vlm.py -v

The trainer methods run on a bare ``SFTTrainer`` (no Ray, no GPU) carrying just
the attributes they read, as ``tests/train/test_sft_tokenization.py`` does. The
image fixture is a few tiny PNGs written under ``tmp_path`` and referenced by
path from the chat ``messages`` (content-part lists, the format
``prepare_cauldron_vlm.py`` produces), so spawned dataloader workers can load
them independently.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from skyrl.train.config.sft_config import SFTConfig
from skyrl.train.dataset.sft_dataset import (
    ConcatSFTDataset,
    LazyVLMDataset,
    TextDataset,
)
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.utils.tok import get_processor, get_tokenizer

# NOTE: These tests require torchvision (the Qwen3-VL processor) and thus the
# fsdp extra -- the vllm marker runs them with the appropriate extras, as for
# the VLM tests in test_sft_tokenization.py.
pytestmark = pytest.mark.vllm

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ROW_KEYS = ("input_ids", "attention_mask", "num_actions", "loss_mask", "pixel_values", "image_grid_thw")


@pytest.fixture(scope="module")
def vlm_processor():
    return get_processor(MODEL, use_fast=True, padding_side="left")


@pytest.fixture(scope="module")
def vlm_tokenizer():
    return get_tokenizer(MODEL, use_fast=True, padding_side="left")


# ---------------------------------------------------------------------------
# Fixtures: an image chat dataset on disk
# ---------------------------------------------------------------------------


def _image_part(path):
    return {"type": "image", "path": path}


def _text_part(text):
    return {"type": "text", "text": text}


def _write_image_dataset(dataset_dir, image_dir):
    """Write a 6-row image chat dataset as ``<dataset_dir>/train.parquet``.

    Rows 0, 1, 4 and 5 tokenize; row 2 ends with a user turn (dropped) and row 3
    is over-length (dropped, VLM rows are never truncated) at ``max_length=200``.
    Returns the source indices that survive.
    """
    from datasets import Dataset
    from PIL import Image

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    paths = []
    for i, (w, h) in enumerate([(64, 64), (96, 64), (64, 96), (128, 96), (64, 64)]):
        path = os.path.join(image_dir, f"img{i}.png")
        Image.new("RGB", (w, h), (40 * i, 80, 200 - 30 * i)).save(path)
        paths.append(path)

    def row(user_parts, answer):
        return {
            "messages": [
                {"role": "user", "content": user_parts},
                {"role": "assistant", "content": [_text_part(answer)]},
            ]
        }

    rows = [
        row([_image_part(paths[0]), _text_part("What is in the image?")], "A square."),
        row([_image_part(paths[1]), _image_part(paths[2]), _text_part("Compare the two images.")], "Two rectangles."),
        {
            "messages": [
                {"role": "user", "content": [_image_part(paths[0]), _text_part("Describe.")]},
                {"role": "assistant", "content": [_text_part("A square.")]},
                {"role": "user", "content": [_text_part("Thanks")]},
            ]
        },
        row([_image_part(paths[3]), _text_part("Describe at length.")], "Blue. " * 150),
        row([_image_part(paths[3]), _text_part("Colour?")], "Blue."),
        row([_image_part(paths[4]), _text_part("Shape?")], "Square."),
    ]
    Dataset.from_list(rows).to_parquet(os.path.join(dataset_dir, "train.parquet"))
    return [0, 1, 4, 5]


@pytest.fixture(scope="module")
def image_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("vlm_data")
    dataset_dir = str(root / "ds")
    kept = _write_image_dataset(dataset_dir, str(root / "images"))
    return dataset_dir, kept


def _build_trainer(vlm_tokenizer, vlm_processor, **overrides) -> SFTTrainer:
    """Bare SFTTrainer with the attributes the data-loading methods read."""
    cfg = SFTConfig()
    cfg.model.path = MODEL
    cfg.max_length = 200
    cfg.num_workers = 0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    trainer = object.__new__(SFTTrainer)
    trainer.sft_cfg = cfg
    trainer.tokenizer = vlm_tokenizer
    trainer.processor = vlm_processor
    trainer.is_vlm = True
    return trainer


def _holds_tensor(obj, depth=0) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if depth > 2:
        return False
    if isinstance(obj, (list, tuple)):
        return any(_holds_tensor(x, depth + 1) for x in obj[:64])
    if isinstance(obj, dict):
        return any(_holds_tensor(v, depth + 1) for v in obj.values())
    return False


def _assert_rows_equal(lazy_row, eager_row):
    for key in ROW_KEYS:
        assert (key in lazy_row) == (key in eager_row), key
        if key in lazy_row:
            lazy_v, eager_v = lazy_row[key], eager_row[key]
            if isinstance(lazy_v, torch.Tensor) or isinstance(eager_v, torch.Tensor):
                assert torch.equal(torch.as_tensor(lazy_v), torch.as_tensor(eager_v)), key
            else:
                assert lazy_v == eager_v, key


@pytest.fixture(scope="module")
def eager_rows(image_dataset, vlm_tokenizer, vlm_processor):
    """The eager (fully materialized) path's rows: the reference."""
    dataset_dir, _ = image_dataset
    trainer = _build_trainer(vlm_tokenizer, vlm_processor, lazy_vlm_dataset=False, disable_cache=True)
    rows = trainer._load_and_tokenize(dataset_dir, "train")
    assert isinstance(rows, list)
    return rows


# ---------------------------------------------------------------------------
# Parity with the eager path
# ---------------------------------------------------------------------------


def test_lazy_rows_match_eager_and_hold_no_tensors(image_dataset, eager_rows, vlm_tokenizer, vlm_processor, tmp_path):
    dataset_dir, kept = image_dataset
    trainer = _build_trainer(vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, cache_dir=str(tmp_path / "cache"))
    lazy = trainer._load_and_tokenize(dataset_dir, "train")

    assert isinstance(lazy, LazyVLMDataset)
    assert len(lazy) == len(eager_rows) == len(kept)
    assert lazy.source_indices == kept
    assert lazy.sequence_lengths == TextDataset(eager_rows).sequence_lengths
    # Only indices, lengths, the arrow-backed source and the tokenize fn are held.
    assert not any(_holds_tensor(v) for v in vars(lazy).values())

    for i in range(len(lazy)):
        _assert_rows_equal(lazy[i], eager_rows[i])
    # Multi-image row round-trips every image.
    assert torch.as_tensor(lazy[1]["image_grid_thw"]).shape[0] == 2
    # Negative indexing, like a list.
    _assert_rows_equal(lazy[-1], eager_rows[-1])
    with pytest.raises(IndexError):
        lazy[len(lazy)]


def test_lazy_batched_fetch_and_dataset_stats(image_dataset, eager_rows, vlm_tokenizer, vlm_processor, tmp_path):
    """``__getitems__`` (torch's preferred fetch) and ``_log_dataset_stats`` work on the lazy dataset."""
    dataset_dir, _ = image_dataset
    trainer = _build_trainer(vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, cache_dir=str(tmp_path / "cache"))
    lazy = trainer._load_and_tokenize(dataset_dir, "train")
    rows = lazy.__getitems__([2, 0])
    _assert_rows_equal(rows[0], eager_rows[2])
    _assert_rows_equal(rows[1], eager_rows[0])
    trainer._log_dataset_stats(lazy)


def test_lazy_rows_match_eager_for_data_uri_images(vlm_tokenizer, vlm_processor, tmp_path):
    """Images embedded as base64 data URIs (``{"type": "image", "url": "data:image/jpeg;base64,..."}``)
    are decoded per fetch and match the eager rows.

    The URI sits under ``url`` rather than ``image``: parquet unifies every
    content part to one struct, so a text part would carry ``image: None``, and
    the Qwen chat template renders an image placeholder for any part with an
    ``image`` key (``'image' in content``), which fails tokenization on both
    paths with ``StopIteration`` in the processor."""
    import base64
    import io

    from datasets import Dataset
    from PIL import Image

    def data_uri(w, h, colour):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), colour).save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "url": data_uri(64, 64, (200, 0, 0))}, _text_part("Colour?")],
                },
                {"role": "assistant", "content": [_text_part("Red.")]},
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": data_uri(96, 64, (0, 200, 0))},
                        {"type": "image", "url": data_uri(64, 96, (0, 0, 200))},
                        _text_part("Colours?"),
                    ],
                },
                {"role": "assistant", "content": [_text_part("Green and blue.")]},
            ]
        },
    ]
    dataset_dir = str(tmp_path / "uri_ds")
    os.makedirs(dataset_dir)
    Dataset.from_list(rows).to_parquet(os.path.join(dataset_dir, "train.parquet"))

    eager = _build_trainer(vlm_tokenizer, vlm_processor, lazy_vlm_dataset=False, disable_cache=True)
    eager_rows = eager._load_and_tokenize(dataset_dir, "train")
    lazy = _build_trainer(vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, cache_dir=str(tmp_path / "cache"))
    lazy_dataset = lazy._load_and_tokenize(dataset_dir, "train")

    assert isinstance(lazy_dataset, LazyVLMDataset) and len(lazy_dataset) == len(eager_rows) == 2
    for i in range(2):
        _assert_rows_equal(lazy_dataset[i], eager_rows[i])
    assert torch.as_tensor(lazy_dataset[1]["image_grid_thw"]).shape[0] == 2


# ---------------------------------------------------------------------------
# Index cache
# ---------------------------------------------------------------------------


def _count_tokenize_calls(monkeypatch):
    import skyrl.train.sft_trainer as sft_trainer

    calls = {"n": 0}
    real = sft_trainer.tokenize_chat_example

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sft_trainer, "tokenize_chat_example", counting)
    return calls


def test_lazy_index_cache_roundtrip(image_dataset, vlm_tokenizer, vlm_processor, tmp_path, monkeypatch):
    dataset_dir, kept = image_dataset
    cache_dir = str(tmp_path / "cache")
    calls = _count_tokenize_calls(monkeypatch)

    trainer = _build_trainer(vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, cache_dir=cache_dir)
    first = trainer._load_and_tokenize(dataset_dir, "train")
    assert calls["n"] == 6  # one pass over every source row
    index_files = [f for f in os.listdir(cache_dir) if f.endswith(".lazy_vlm.json")]
    assert len(index_files) == 1
    with open(os.path.join(cache_dir, index_files[0])) as f:
        meta = json.load(f)
    assert meta == {"num_rows": 6, "indices": kept, "lengths": first.sequence_lengths}
    # No eager cache directory was written next to it.
    assert not any(os.path.isdir(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir))

    # Cache hit: no tokenization on load, identical index.
    calls["n"] = 0
    second = trainer._load_and_tokenize(dataset_dir, "train")
    assert calls["n"] == 0
    assert isinstance(second, LazyVLMDataset)
    assert second.source_indices == first.source_indices
    assert second.sequence_lengths == first.sequence_lengths
    # ...and fetching still tokenizes (the cache holds no rows).
    row = second[0]
    assert calls["n"] == 1 and "pixel_values" in row

    # force_recache re-runs the pass.
    calls["n"] = 0
    trainer.sft_cfg.force_recache = True
    third = trainer._load_and_tokenize(dataset_dir, "train")
    assert calls["n"] == 6 and third.source_indices == kept
    trainer.sft_cfg.force_recache = False

    # A stale index (built over a different row count) is ignored and rebuilt.
    with open(os.path.join(cache_dir, index_files[0]), "w") as f:
        json.dump({"num_rows": 7, "indices": [0], "lengths": [10]}, f)
    calls["n"] = 0
    fourth = trainer._load_and_tokenize(dataset_dir, "train")
    assert calls["n"] == 6 and fourth.source_indices == kept
    with open(os.path.join(cache_dir, index_files[0])) as f:
        assert json.load(f)["num_rows"] == 6

    # Malformed indexes are ignored and rebuilt rather than failing startup:
    # corrupt JSON, a negative index, a non-integer entry, a zero length, a
    # length/index count mismatch.
    for bad in (
        "{not json",
        json.dumps({"num_rows": 6, "indices": [-1, 0, 1, 4], "lengths": [10, 10, 10, 10]}),
        json.dumps({"num_rows": 6, "indices": [0, "x", 1, 4], "lengths": [10, 10, 10, 10]}),
        json.dumps({"num_rows": 6, "indices": [0, 1, 4, 5], "lengths": [10, 0, 10, 10]}),
        json.dumps({"num_rows": 6, "indices": [0, 1, 4, 5], "lengths": [10, 10]}),
    ):
        with open(os.path.join(cache_dir, index_files[0]), "w") as f:
            f.write(bad)
        calls["n"] = 0
        rebuilt = trainer._load_and_tokenize(dataset_dir, "train")
        assert calls["n"] == 6 and rebuilt.source_indices == kept, bad
    # No temp files are left behind by the writes.
    assert not [f for f in os.listdir(cache_dir) if ".tmp" in f]


def test_lazy_disable_cache_writes_nothing(image_dataset, vlm_tokenizer, vlm_processor, tmp_path):
    dataset_dir, kept = image_dataset
    cache_dir = str(tmp_path / "cache")
    trainer = _build_trainer(
        vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, disable_cache=True, cache_dir=cache_dir
    )
    lazy = trainer._load_and_tokenize(dataset_dir, "train")
    assert lazy.source_indices == kept
    assert not os.path.exists(cache_dir)


# ---------------------------------------------------------------------------
# When the flag does not apply
# ---------------------------------------------------------------------------


def test_lazy_flag_ignored_for_text_models(image_dataset, vlm_tokenizer, vlm_processor, tmp_path):
    """A text-only model (``is_vlm=False``) tokenizes eagerly whatever the flag says."""
    from datasets import Dataset

    text_dir = str(tmp_path / "text_ds")
    os.makedirs(text_dir)
    Dataset.from_list(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}] * 3
    ).to_parquet(os.path.join(text_dir, "train.parquet"))
    trainer = _build_trainer(
        vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, disable_cache=True, cache_dir=str(tmp_path / "cache")
    )
    trainer.is_vlm = False
    trainer.processor = None
    rows = trainer._load_and_tokenize(text_dir, "train")
    assert isinstance(rows, list) and len(rows) == 3


def test_lazy_falls_back_to_eager_for_alpaca_format(vlm_tokenizer, vlm_processor, tmp_path):
    """Alpaca rows carry no images, so a VLM with the flag set still tokenizes them eagerly."""
    from datasets import Dataset

    alpaca_dir = str(tmp_path / "alpaca_ds")
    os.makedirs(alpaca_dir)
    Dataset.from_list(
        [{"instruction": f"Add {i} and {i}.", "input": "", "output": str(2 * i)} for i in range(3)]
    ).to_parquet(os.path.join(alpaca_dir, "train.parquet"))
    trainer = _build_trainer(
        vlm_tokenizer, vlm_processor, lazy_vlm_dataset=True, disable_cache=True, cache_dir=str(tmp_path / "cache")
    )
    rows = trainer._load_and_tokenize(alpaca_dir, "train")
    assert isinstance(rows, list) and len(rows) == 3
    assert all("pixel_values" not in r for r in rows)


# ---------------------------------------------------------------------------
# LazyVLMDataset construction and fetch errors
# ---------------------------------------------------------------------------


def test_lazy_dataset_validates_inputs():
    source = list(range(5))
    with pytest.raises(ValueError, match="differ in length"):
        LazyVLMDataset(source, [0, 1], [3], lambda ex: {"input_ids": [1]})
    with pytest.raises(ValueError, match="must lie in"):
        LazyVLMDataset(source, [0, 5], [3, 3], lambda ex: {"input_ids": [1]})
    with pytest.raises(ValueError, match="must lie in"):
        LazyVLMDataset(source, [-1], [3], lambda ex: {"input_ids": [1]})
    empty = LazyVLMDataset(source, [], [], lambda ex: None)
    assert len(empty) == 0 and empty.sequence_lengths == []


def test_lazy_dataset_nondeterministic_tokenize_fn_raises():
    dataset = LazyVLMDataset(["a", "b"], [0, 1], [1, 1], lambda ex: None)
    with pytest.raises(RuntimeError, match="deterministic"):
        dataset[0]


def test_lazy_dataset_batched_fetch_on_plain_sequence():
    """Sources without batched indexing fall back to row-wise fetches; bad indices raise."""
    dataset = LazyVLMDataset(["a", "b", "c"], [2, 0], [1, 1], lambda ex: {"input_ids": [ord(ex)]})
    assert dataset.__getitems__([1, 0, -1]) == [{"input_ids": [97]}, {"input_ids": [99]}, {"input_ids": [97]}]
    with pytest.raises(IndexError):
        dataset.__getitems__([2])


# ---------------------------------------------------------------------------
# Through the trainer: load_dataset, samplers, dataloaders (incl. spawned workers), eval
# ---------------------------------------------------------------------------


def _identity_collator(examples, batch_size):
    """Module-level (picklable) collator: hands the fetched rows back."""
    return list(examples)


def test_load_dataset_concatenates_lazy_sources(image_dataset, eager_rows, vlm_tokenizer, vlm_processor, tmp_path):
    dataset_dir, kept = image_dataset
    trainer = _build_trainer(
        vlm_tokenizer,
        vlm_processor,
        lazy_vlm_dataset=True,
        cache_dir=str(tmp_path / "cache"),
        train_datasets=[dataset_dir, dataset_dir],
        train_dataset_splits=["train", "train"],
        train_dataset_weights=[0.5, 0.5],
    )
    single = _build_trainer(
        vlm_tokenizer,
        vlm_processor,
        lazy_vlm_dataset=True,
        cache_dir=str(tmp_path / "cache"),
        train_datasets=[dataset_dir],
        train_dataset_splits=["train"],
    ).load_dataset()
    assert isinstance(single, LazyVLMDataset)

    concat = trainer.load_dataset()
    assert isinstance(concat, ConcatSFTDataset)
    assert concat.dataset_lengths == [len(kept), len(kept)]
    assert all(isinstance(d, LazyVLMDataset) for d in concat.datasets)
    assert concat.sequence_lengths == TextDataset(eager_rows).sequence_lengths * 2
    # Batched fetch across sources returns rows in the requested order.
    rows = concat.__getitems__([len(kept) + 1, 0])
    _assert_rows_equal(rows[0], eager_rows[1])
    _assert_rows_equal(rows[1], eager_rows[0])
    # The mixing sampler is built from the concat's lengths.
    sampler = trainer.build_train_sampler(concat)
    assert sampler is not None


@pytest.mark.parametrize("num_workers", [0, 2])
def test_train_dataloader_fetches_lazy_rows(
    image_dataset, eager_rows, vlm_tokenizer, vlm_processor, tmp_path, num_workers
):
    """The StatefulDataLoader (in-process and with spawned workers, which must
    unpickle the arrow-backed source, tokenizer and processor) yields the eager rows."""
    dataset_dir, kept = image_dataset
    trainer = _build_trainer(
        vlm_tokenizer,
        vlm_processor,
        lazy_vlm_dataset=True,
        cache_dir=str(tmp_path / "cache"),
        sampler="sequential",
        batch_size=2,
        dataloader_num_workers=num_workers,
        use_sequence_packing=True,  # no row padding (the identity collator returns a list)
    )
    trainer.collator = _identity_collator
    lazy = trainer._load_and_tokenize(dataset_dir, "train")
    dataloader = trainer.build_train_dataloader(lazy)
    fetched = [row for batch in dataloader for row in batch]
    assert len(fetched) == len(kept)
    for lazy_row, eager_row in zip(fetched, eager_rows):
        _assert_rows_equal(lazy_row, eager_row)


def test_eval_dataloader_collates_lazy_rows(image_dataset, eager_rows, vlm_tokenizer, vlm_processor, tmp_path):
    """``load_eval_datasets`` returns the lazy dataset and the eval dataloader
    collates its rows (through the real ``DefaultCollator``) into VLM batches."""
    from skyrl.backends.skyrl_train.training_batch import TensorList
    from skyrl.train.dataset.collators import DefaultCollator

    dataset_dir, kept = image_dataset
    trainer = _build_trainer(
        vlm_tokenizer,
        vlm_processor,
        lazy_vlm_dataset=True,
        cache_dir=str(tmp_path / "cache"),
        eval_datasets=[dataset_dir],
        eval_dataset_splits=["train"],
        eval_dataset_names=["imgs"],
        micro_train_batch_size_per_gpu=2,
    )
    trainer.collator = DefaultCollator(tokenizer=vlm_tokenizer, micro_train_batch_size_per_gpu=2)
    trainer.dispatch = SimpleNamespace(dp_size=lambda _name: 1)

    eval_sets = trainer.load_eval_datasets()
    assert len(eval_sets) == 1
    name, eval_dataset = eval_sets[0]
    assert name == "imgs" and isinstance(eval_dataset, LazyVLMDataset) and len(eval_dataset) == len(kept)

    batches = list(trainer.build_eval_dataloader(eval_dataset))
    assert len(batches) == 2
    first = batches[0]
    assert first["sequences"].shape[0] == 2
    assert isinstance(first["pixel_values"], TensorList) and len(first["pixel_values"]) == 2
    assert torch.equal(first["pixel_values"][1], torch.as_tensor(eager_rows[1]["pixel_values"]))
    assert first.metadata["response_length"] == max(eager_rows[0]["num_actions"], eager_rows[1]["num_actions"])
