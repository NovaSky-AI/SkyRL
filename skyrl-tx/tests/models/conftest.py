import os
import tempfile

import pytest
from transformers import AutoModelForCausalLM

_WEIGHTS_CACHE = os.path.join(tempfile.gettempdir(), "skyrl-test-weights")


def _get_or_save_hf_weights(model_name: str) -> str:
    """Return path to saved HF weights, downloading and saving only on first call.

    Uses a filesystem-based cache so that multiple tests (including across
    pytest --forked subprocesses) share the same saved weights instead of each
    calling save_pretrained() separately.
    """
    safe_name = model_name.replace("/", "--")
    weights_dir = os.path.join(_WEIGHTS_CACHE, safe_name)
    marker = os.path.join(weights_dir, ".done")
    if not os.path.exists(marker):
        os.makedirs(weights_dir, exist_ok=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager", use_safetensors=True
        )
        hf_model.save_pretrained(weights_dir, safe_serialization=True)
        del hf_model
        open(marker, "w").close()
    return weights_dir


@pytest.fixture
def hf_weights_dir():
    """Fixture providing a function to get cached HF model weights directories."""
    return _get_or_save_hf_weights
