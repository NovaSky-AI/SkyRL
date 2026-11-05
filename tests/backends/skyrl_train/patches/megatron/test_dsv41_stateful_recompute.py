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
    pre = state.pending_pre if state.pending_pre is not None else hidden.new_ones(hidden.shape[0], 1)
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


def test_stateful_recompute_trains_params_when_the_input_is_frozen():
    """LoRA: the embedding is frozen, so the stack's input does not require grad; the layers' parameters must still
    get the same gradients as without recompute (they are not inputs of the checkpoint Function)."""
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(ToyLayer(source) for source in (True, False, True)).double()
    x = torch.randn(SEQ, HIDDEN, dtype=torch.float64)  # requires_grad=False

    def grads(stateful):
        for p in layers.parameters():
            p.grad = None
        run_stack(layers, x, stateful).backward()
        return [None if p.grad is None else p.grad.clone() for p in layers.parameters()]

    ref = grads(stateful=False)
    got = grads(stateful=True)
    assert any(g is not None and torch.count_nonzero(g) for g in got)
    for g, r in zip(got, ref):
        if r is None:
            assert g is None or torch.count_nonzero(g) == 0
        else:
            torch.testing.assert_close(g, r)


def test_stateful_recompute_with_offload_matches_on_cuda(monkeypatch):
    """Offloaded layer inputs (pinned pool, side-stream copies, backward prefetch) give the same gradients;
    two iterations so pooled host buffers are reused."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(recompute, "OFFLOAD", True)
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(ToyLayer(source) for source in (True, False, True, False, False)).double().cuda()
    for _ in range(2):
        x = torch.randn(SEQ, HIDDEN, dtype=torch.float64, device="cuda", requires_grad=True)
        results = []
        for stateful in (False, True):
            for p in [x, *layers.parameters()]:
                p.grad = None
            loss = run_stack(layers, x, stateful)
            loss.backward()
            results.append((loss.detach(), [p.grad.clone() for p in [x, *layers.parameters()] if p.grad is not None]))
        (ref_loss, ref_grads), (loss, grads) = results
        torch.testing.assert_close(loss, ref_loss)
        assert len(grads) == len(ref_grads)
        for got, want in zip(grads, ref_grads):
            torch.testing.assert_close(got, want)
    assert not recompute._PENDING
    assert recompute._POOL.free or recompute._POOL.returning  # chunks returned to the arena


def test_offload_arena_splits_inputs_across_chunks_and_reuses_them(monkeypatch):
    """Tiny chunks: inputs span several chunks with ragged tails and vary in size between iterations, as RL
    batch lengths do; values round-trip exactly and the arena stops pinning once its peak demand is reached."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(recompute, "OFFLOAD", True)
    monkeypatch.setattr(recompute, "_CHUNK", 1000)  # bytes: a [SEQ, HIDDEN] fp64 input spans 2 chunks
    monkeypatch.setattr(recompute, "_POOL", recompute._PinnedPool())
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(ToyLayer(source) for source in (True, False, True)).double().cuda()
    pinned = []
    for seq in (SEQ, SEQ - 5, SEQ, SEQ - 3):
        x = torch.randn(seq, HIDDEN, dtype=torch.float64, device="cuda", requires_grad=True)
        grads = []
        for stateful in (False, True):
            x.grad = None
            for p in layers.parameters():
                p.grad = None
            run_stack(layers, x, stateful).backward()
            grads.append([x.grad.clone()] + [p.grad.clone() for p in layers.parameters() if p.grad is not None])
        for got, want in zip(grads[1], grads[0]):
            torch.testing.assert_close(got, want)
        pinned.append(recompute._POOL.live_bytes)
    assert pinned[2] == pinned[1] == pinned[0] and pinned[3] == pinned[0]  # peak reached at the first step


def test_offload_respects_the_pinned_host_limit(monkeypatch):
    """With no pinned budget every layer input stays on the GPU, and gradients are unchanged."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(recompute, "OFFLOAD", True)
    monkeypatch.setattr(recompute, "_POOL", recompute._PinnedPool())
    monkeypatch.setattr(recompute._POOL, "limit", 0)
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(ToyLayer(source) for source in (True, False, True)).double().cuda()
    x = torch.randn(SEQ, HIDDEN, dtype=torch.float64, device="cuda", requires_grad=True)
    ref = run_stack(layers, x, stateful=False)
    ref.backward()
    ref_grad = x.grad.clone()
    x.grad = None
    loss = run_stack(layers, x, stateful=True)
    loss.backward()
    torch.testing.assert_close(x.grad, ref_grad)
    assert recompute._POOL.live_bytes == 0 and not recompute._PENDING


def test_replay_reuses_the_forward_top_k(monkeypatch):
    """Index sources select once per step: the replay in backward gets the forward's integer selection
    (``attention.reuse_selection``) instead of running the indexer again; gradients are unchanged."""
    from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41 import attention

    selections = []

    def selecting_layer_forward(layer, hidden, scale, *, state_getter):
        state = state_getter()
        if layer.source:
            replayed = attention._replayed_selection()
            if replayed is not None and replayed[0] is not None:
                state.csa2.global_indices = replayed[0]
            else:
                selections.append(layer)
                state.csa2.global_indices = (hidden.sum(-1) > 0).to(torch.int32)
            state.csa2.global_kv = layer.kv(hidden)
            state.csa2.compress_ratio = 2
        pre = state.pending_pre if state.pending_pre is not None else hidden.new_ones(hidden.shape[0], 1)
        out = torch.tanh(layer.proj(hidden)) * pre * scale
        if state.csa2.global_kv is not None:
            out = out + 0.5 * state.csa2.global_kv * state.csa2.global_indices.unsqueeze(-1)
        state.pending_pre = torch.sigmoid(out.mean(-1, keepdim=True))
        return out, None

    monkeypatch.setitem(globals(), "toy_layer_forward", selecting_layer_forward)
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(ToyLayer(source) for source in (True, False, True, False)).double()
    x = torch.randn(SEQ, HIDDEN, dtype=torch.float64, requires_grad=True)
    grads = []
    for stateful in (False, True):
        selections.clear()
        for p in [x, *layers.parameters()]:
            p.grad = None
        run_stack(layers, x, stateful).backward()
        grads.append([None if p.grad is None else p.grad.clone() for p in [x, *layers.parameters()]])
        assert len(selections) == 2  # once per source, forward only
    for got, want in zip(grads[1], grads[0]):
        if want is None:
            assert got is None or torch.count_nonzero(got) == 0
        else:
            torch.testing.assert_close(got, want)
