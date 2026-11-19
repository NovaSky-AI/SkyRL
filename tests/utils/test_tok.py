"""``get_tokenizer`` names a missing eos token from the model's ``config.json``."""

import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl.utils.tok import get_tokenizer


def _save_tokenizer(path, config: dict, eos_token=None):
    vocab = {"<unk>": 0, "hello": 1, "<|end|>": 2}
    fast = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")), unk_token="<unk>")
    if eos_token is not None:
        fast.eos_token = eos_token
    fast.save_pretrained(path)
    (path / "config.json").write_text(json.dumps(config))


def test_eos_from_config(tmp_path):
    _save_tokenizer(tmp_path, {"eos_token_id": 2})
    tokenizer = get_tokenizer(str(tmp_path))
    assert tokenizer.eos_token == "<|end|>" and tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 2  # pad falls back to the recovered eos


def test_eos_from_text_config_list(tmp_path):
    _save_tokenizer(tmp_path, {"text_config": {"eos_token_id": [2, 1]}})
    assert get_tokenizer(str(tmp_path)).eos_token_id == 2


def test_tokenizer_eos_wins(tmp_path):
    _save_tokenizer(tmp_path, {"eos_token_id": 1}, eos_token="<|end|>")
    assert get_tokenizer(str(tmp_path)).eos_token_id == 2


def test_no_eos_anywhere(tmp_path):
    _save_tokenizer(tmp_path, {})
    assert get_tokenizer(str(tmp_path)).eos_token_id is None
