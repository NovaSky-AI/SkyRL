"""DeepSeek-V4.1 stateful layer recompute (``deepseek_v41/recompute.py``) against no recompute.

A toy stack mirrors the V4.1 state flow: source layers publish a differentiable ``global_kv`` (consumers
backpropagate into the source) and integer ``global_indices``; later layers read them; every layer updates the
mHC ``pending_pre``. Under the stateful checkpoint the output and every parameter / input gradient must equal
the plain forward / backward.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41 import (
    recompute,  # noqa: E402
)
from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.attention import (  # noqa: E402
    CSA2State,
)

HIDDEN, SEQ = 16, 12


class ToyLayer(torch.nn.Module):
    def __init__(self, source: bool):
        super().__init__()
        self.source = source
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN)
        self.kv = torch.nn.Linear(HIDDEN, HIDDEN)


def toy_layer_forward(layer, hidden, scale, *, state_getter):
    state = state_getter()
    if layer.source:
        state.csa2.global_kv = layer.kv(hidden)  # differentiable, consumed by later layers
        state.csa2.global_indices = (hidden.sum(-1) > 0).to(torch.int32)  # integer state
        state.csa2.compress_ratio = 2
    pre = state.pending_pre if state.pending_pre is not None else torch.ones(SEQ, 1)
    out = torch.tanh(layer.proj(hidden)) * pre * scale
    if state.csa2.global_kv is not None:
        out = out + 0.5 * state.csa2.global_kv * state.csa2.global_indices.unsqueeze(-1)
    state.pending_pre = torch.sigmoid(out.mean(-1, keepdim=True))  # next layer's pre-mix
    return out, None


def run_stack(layers, x, stateful: bool):
    holder = SimpleNamespace(csa2=CSA2State(), pending_pre=None)

    def state_getter():
        return holder

    def layer_forward(layer, hidden, scale):
        return toy_layer_forward(layer, hidden, scale, state_getter=state_getter)

    hidden = x
    for layer in layers:
        if stateful:
            hidden, _ = recompute.stateful_layer_forward(layer_forward, layer, hidden, (0.9,), {}, state_getter)
        else:
            hidden, _ = layer_forward(layer, hidden, 0.9)
    return (hidden * holder.pending_pre).sum()


@pytest.mark.parametrize("pattern", [(True, False, False, True, False), (False, True, True, False)])
def test_stateful_recompute_matches_no_recompute(pattern):
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(ToyLayer(source) for source in pattern).double()
    x = torch.randn(SEQ, HIDDEN, dtype=torch.float64, requires_grad=True)

    # As with megatron-core's CheckpointFunction, parameter gradients accumulate into ``.grad`` during the
    # replay in backward (``loss.backward()``, as training does), not through ``torch.autograd.grad``.
    def grads_after_backward(stateful):
        for p in [x, *layers.parameters()]:
            p.grad = None
        loss = run_stack(layers, x, stateful)
        loss.backward()
        return loss.detach(), [None if p.grad is None else p.grad.clone() for p in [x, *layers.parameters()]]

    ref_loss, ref_grads = grads_after_backward(stateful=False)
    loss, grads = grads_after_backward(stateful=True)

    torch.testing.assert_close(loss, ref_loss)
    for got, want in zip(grads, ref_grads):
        if want is None:
            assert got is None or torch.count_nonzero(got) == 0
        else:
            torch.testing.assert_close(got, want)
