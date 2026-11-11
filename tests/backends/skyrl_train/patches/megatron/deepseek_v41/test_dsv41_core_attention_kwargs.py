"""DeepSeek-V4.1's core attention accepts every keyword megatron-core's DSv4 hybrid attention passes it.

megatron-core ed9c2c8c (#2558) started passing ``boundary_hidden`` / ``boundary_kv`` (context-parallel boundary
rows) to the core attention module, and every DSv4.1 training forward failed with ``TypeError: ...
unexpected keyword argument 'boundary_hidden'``. The first test reads the call from the installed megatron-core,
so a future signature change fails here instead of in a GPU run.
"""

import inspect
import re

import pytest
import torch

pytestmark = pytest.mark.megatron
hybrid = pytest.importorskip("megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention")

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.attention import (  # noqa: E402
    CompressedSparseAttention2,
)


def _core_attention_call_kwargs() -> set:
    source = inspect.getsource(hybrid)
    match = re.search(r"apply_module\(self\.core_attention\)\((.*?)\n\s*\)\n", source, re.S)
    assert match, "megatron-core's DSv4 hybrid attention no longer calls apply_module(self.core_attention)(...)"
    return set(re.findall(r"^\s*(\w+)=", match.group(1), re.M))


def test_accepts_every_keyword_megatron_core_passes():
    accepted = set(inspect.signature(CompressedSparseAttention2.forward).parameters)
    missing = _core_attention_call_kwargs() - accepted
    assert not missing, f"CompressedSparseAttention2.forward does not accept {sorted(missing)}"


@pytest.mark.parametrize("name", ["boundary_hidden", "boundary_kv"])
def test_rejects_context_parallel_boundary_rows(name):
    attention = object.__new__(CompressedSparseAttention2)  # forward checks its arguments before touching state
    with pytest.raises(AssertionError, match="context-parallel boundary rows"):
        CompressedSparseAttention2.forward(attention, None, None, None, None, **{name: torch.zeros(1)})
