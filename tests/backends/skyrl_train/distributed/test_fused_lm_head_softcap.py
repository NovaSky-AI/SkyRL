"""Final logit softcap in the fused LM-head log-prob and entropy ops (CPU, one gloo rank).

The fused path bypasses the output layer, so for a softcapped model (Beam) it has to
apply ``softcap * tanh(logits / softcap)`` and then the temperature itself. Checked
against plain autograd on materialized logits.

Run with:
  uv run --isolated --extra skyrl-train --extra dev -- pytest -s \
    tests/backends/skyrl_train/distributed/test_fused_lm_head_softcap.py
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


B, S, H, V = 2, 7, 16, 11


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    # Large logits so the cap actually bends them (|logits| well past softcap).
    hidden = torch.randn(B, S, H, generator=g, dtype=torch.float32) * 4
    weight = torch.randn(V, H, generator=g, dtype=torch.float32)
    target = torch.randint(0, V, (B, S), generator=g)
    return hidden, weight, target


def _reference(hidden, weight, target, softcap, temperature):
    logits = hidden @ weight.t()
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits / temperature
    return torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1), logits


@pytest.mark.parametrize("softcap", [None, 3.0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("chunk_size", [3, S])
def test_fused_logprob_matches_autograd(tp_group, softcap, temperature, chunk_size):
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        FusedLinearChunkedDistributedLogprob,
    )

    hidden, weight, target = _inputs()
    seed = torch.linspace(0.5, 1.5, steps=B * S, dtype=torch.float32).reshape(B, S)

    h_ref, w_ref = hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    ref, _ = _reference(h_ref, w_ref, target, softcap, temperature)
    ref.backward(seed)

    h, w = hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    # Without a softcap, callers fold the temperature into the weight (from_parallel_hidden_to_logprobs).
    w_in = w if softcap is not None else w / temperature
    out = FusedLinearChunkedDistributedLogprob.apply(
        h, w_in, target, 0, V, chunk_size, tp_group, False, softcap, temperature
    )
    out.backward(seed)

    torch.testing.assert_close(out, ref.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(h.grad, h_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(w.grad, w_ref.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("softcap", [None, 3.0])
def test_fused_entropy_matches_reference(tp_group, softcap):
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _fused_vocab_parallel_entropy_from_hidden,
    )

    hidden, weight, target = _inputs(seed=1)
    _, logits = _reference(hidden, weight, target, softcap, temperature=0.7)
    probs = torch.softmax(logits, dim=-1)
    expected = -(probs * torch.log_softmax(logits, dim=-1)).sum(-1)

    with torch.no_grad():
        got = _fused_vocab_parallel_entropy_from_hidden(
            hidden, weight, tp_group, chunk_size=3, temperature=0.7, softcap=softcap
        )
    torch.testing.assert_close(got, expected.to(got.dtype))
