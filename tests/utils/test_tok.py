import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, PreTrainedTokenizerFast

from skyrl.utils.tok import get_tokenizer


def save_tokenizer(path, eos_token=None, pad_token=None):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
        eos_token=eos_token,
        pad_token=pad_token,
    )
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("eos_token_id", [1, [1, 2]])
def test_missing_tokenizer_eos_and_pad_use_model_config(tmp_path, eos_token_id):
    save_tokenizer(tmp_path)
    LlamaConfig(eos_token_id=eos_token_id).save_pretrained(tmp_path)
    tokenizer = get_tokenizer(tmp_path, local_files_only=True)
    assert tokenizer.eos_token_id == tokenizer.pad_token_id == 1
    assert tokenizer.eos_token == tokenizer.pad_token == "[EOS]"
    assert len(tokenizer) == 3


@pytest.mark.parametrize(
    "eos,pad,expected_eos,expected_pad",
    [
        ("[EOS]", "[PAD]", 1, 2),
        ("[EOS]", None, 1, 1),
        (None, "[PAD]", None, 2),
    ],
)
def test_existing_tokenizer_special_tokens_are_preserved(tmp_path, eos, pad, expected_eos, expected_pad):
    save_tokenizer(tmp_path, eos, pad)
    # There is deliberately no model config: existing tokens need no extra lookup.
    tokenizer = get_tokenizer(tmp_path, local_files_only=True)
    assert tokenizer.eos_token_id == expected_eos
    assert tokenizer.pad_token_id == expected_pad


def test_missing_eos_in_both_sources_fails_explicitly(tmp_path):
    save_tokenizer(tmp_path)
    LlamaConfig(eos_token_id=None).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Neither tokenizer nor model config defines EOS"):
        get_tokenizer(tmp_path, local_files_only=True)
