"""Targeted CPU regression test for streamed ``ChunkedDistributedLogprob.backward``.

Run with:
  uv run --isolated --extra skyrl-train --extra dev -- pytest -s \
    tests/backends/skyrl_train/distributed/test_chunked_logprob_backward_streaming.py
"""

import os
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.utils import get_free_port

# Stub megatron so CPU CI can import model_utils without megatron-core.
# The fixture restores prior modules, leaving GPU lanes with real megatron intact.

_MEGATRON_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
]

_mock_modules: dict[str, ModuleType] = {}
for _name in _MEGATRON_MODULES:
    _mock_modules[_name] = ModuleType(_name)

_mock_modules["megatron.core"].parallel_state = _mock_modules["megatron.core.parallel_state"]


@pytest.fixture(scope="module", autouse=True)
def _stub_megatron_modules():
    """Install the mock ``megatron`` modules for this module only."""
    saved = {_name: sys.modules.get(_name) for _name in _MEGATRON_MODULES}
    for _name in _MEGATRON_MODULES:
        sys.modules[_name] = _mock_modules[_name]
    try:
        yield
    finally:
        for _name in _MEGATRON_MODULES:
            if saved[_name] is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = saved[_name]


@pytest.fixture(scope="module")
def tp_group():
    """Single-rank gloo TP group; only destroy it if this fixture created it."""
    initialized_here = False
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialized_here = True
    yield dist.group.WORLD
    if initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _backward_grad(func_cls, logits, target, vocab_start, vocab_end, tp_group, *, chunk_size=None):
    """Return the input grad using a non-uniform upstream gradient."""
    leaf = logits.detach().clone().requires_grad_(True)
    if chunk_size is None:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, tp_group, False)
    else:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, chunk_size, tp_group, False)
    grad_seed = torch.linspace(0.5, 1.5, steps=out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
    out.backward(grad_seed)
    return leaf.grad.detach()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param((3, 17, 64, 5, "mixed_oov"), id="ragged_mixed_oov"),
        pytest.param((2, 8, 32, 64, "all_oov"), id="single_chunk_all_oov"),
    ],
)
def test_streamed_backward_matches_non_chunked_for_targeted_cases(tp_group, case):
    """Chunked backward stays bit-identical while writing into the streamed buffer."""
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    batch_size, seq_len, vocab_size, chunk_size, target_mode = case
    device = torch.device("cpu")
    torch.manual_seed(1)

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device=device) * 2.0
    if target_mode == "all_oov":
        target = torch.full((batch_size, seq_len), vocab_size + 5, device=device, dtype=torch.long)
    else:
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
        target[:, ::3] = vocab_size + 5

    grad_ref = _backward_grad(DistributedLogprob, logits, target, 0, vocab_size, tp_group)
    grad_chunk = _backward_grad(
        ChunkedDistributedLogprob,
        logits,
        target,
        0,
        vocab_size,
        tp_group,
        chunk_size=chunk_size,
    )

    assert grad_chunk.shape == grad_ref.shape == logits.shape
    assert grad_chunk.dtype == torch.float32
    assert torch.equal(grad_chunk, grad_ref), "streamed chunked grad must be bit-identical to non-chunked grad"
