"""CPU test that the torch fused LM-head path applies temperature like the unfused path.

The unfused path projects in the weight dtype, casts the logits to FP32 and then
divides by the temperature. These tests check that the fused log-probs, their
gradients and the fused entropy metric match that reference for BF16 weights.

Run with:
  uv run --isolated --extra skyrl-train --extra dev -- pytest -s \
    tests/backends/skyrl_train/distributed/test_fused_lm_head_temperature.py
"""

import os
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.utils import get_free_port

# Stub megatron so CPU CI can import model_utils without megatron-core.
_MEGATRON_MODULES = ["megatron", "megatron.core", "megatron.core.parallel_state"]
_mock_modules = {name: ModuleType(name) for name in _MEGATRON_MODULES}
_mock_modules["megatron.core"].parallel_state = _mock_modules["megatron.core.parallel_state"]

TEMPERATURE = 0.7
BATCH, SEQ, HIDDEN, VOCAB, CHUNK = 2, 24, 64, 96, 8


@pytest.fixture(scope="module", autouse=True)
def _stub_megatron_modules():
    """Install the mock ``megatron`` modules for this module only."""
    saved = {name: sys.modules.get(name) for name in _MEGATRON_MODULES}
    sys.modules.update(_mock_modules)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


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


def _inputs():
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(BATCH, SEQ, HIDDEN, generator=generator).to(torch.bfloat16)
    weight = (torch.randn(VOCAB, HIDDEN, generator=generator) * 0.2).to(torch.bfloat16)
    target = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)
    grad_seed = torch.linspace(0.5, 1.5, steps=BATCH * (SEQ - 1)).reshape(BATCH, SEQ - 1)
    return hidden, weight, target, grad_seed


def _unfused_logits(hidden, weight):
    return torch.matmul(hidden.to(weight.dtype), weight.t()).to(torch.float32) / TEMPERATURE


@pytest.mark.parametrize("chunk_size", [None, CHUNK], ids=["one_chunk", "chunked"])
def test_fused_logprobs_and_grads_match_unfused_temperature_scaling(tp_group, chunk_size):
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        from_parallel_hidden_to_logprobs,
        from_parallel_logits_to_logprobs,
    )

    hidden, weight, target, grad_seed = _inputs()

    ref_hidden = hidden.clone().requires_grad_(True)
    ref_weight = weight.clone().requires_grad_(True)
    expected = from_parallel_logits_to_logprobs(
        _unfused_logits(ref_hidden, ref_weight), target, 0, VOCAB, tp_group, chunk_size=chunk_size
    )
    expected.backward(grad_seed)

    fused_hidden = hidden.clone().requires_grad_(True)
    fused_weight = weight.clone().requires_grad_(True)
    actual = from_parallel_hidden_to_logprobs(
        fused_hidden,
        fused_weight,
        target,
        0,
        VOCAB,
        tp_group,
        chunk_size=chunk_size,
        temperature=TEMPERATURE,
        fused_backend="torch",
    )
    actual.backward(grad_seed)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(fused_hidden.grad, ref_hidden.grad, rtol=2**-8, atol=1e-3)
    # The fused weight gradient rounds each chunk's BF16 product before summing
    # the chunks in FP32, so allow one BF16 ulp at the gradient's largest entry.
    weight_atol = 2**-7 * ref_weight.grad.abs().max().item()
    torch.testing.assert_close(fused_weight.grad, ref_weight.grad, rtol=0, atol=weight_atol)


def test_fused_entropy_matches_unfused_temperature_scaling(tp_group):
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _fused_vocab_parallel_entropy_from_hidden,
    )

    hidden, weight, _, _ = _inputs()
    log_probs = torch.log_softmax(_unfused_logits(hidden, weight), dim=-1)
    expected = -(log_probs.exp() * log_probs).sum(dim=-1)

    actual = _fused_vocab_parallel_entropy_from_hidden(
        hidden, weight, tp_group, chunk_size=CHUNK, temperature=TEMPERATURE
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
