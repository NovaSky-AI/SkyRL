"""CPU unit tests for the draft-projection adapter helpers.

Covers ``_CanonicalGradStrides``: the identity shim that re-canonicalizes gradient
strides between the (frozen-weight) output projection and the adapter's transpose.
Without it, a micro-batch-1 transpose-backward grad keeps a stale stride in its
size-1 batch dim; megatron's ``LinearWithFrozenWeight.backward`` then fails
``torch.matmul``'s fold-to-mm stride check (which does not skip size-1 dims) and
dispatches a batch=seq, M=1 broadcast bmm that is ~100x slower than the flat mm.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_adapter.py
"""

import torch

from skyrl.backends.skyrl_train.mtp.adapter import _CanonicalGradStrides


def _passes_matmul_fold_check(t: torch.Tensor) -> bool:
    """Mirror torch's ``should_fold`` stride loop: foldable iff the leading dims are
    dense in order, with no size-1 skipping (the rule the real dispatch applies)."""
    sizes, strides = t.shape, t.stride()
    return all(strides[i] == strides[i + 1] * sizes[i + 1] for i in range(t.dim() - 2))


def _projection_chain(x, use_shim: bool):
    """Mimic project_mtp_hidden_to_logits' post-projection ops on a [seq, 1, vocab]
    tensor: (shim) -> transpose(0,1) -> contiguous (a no-op view for batch 1)."""
    y = _CanonicalGradStrides.apply(x) if use_shim else x
    return y.transpose(0, 1).contiguous()


def test_canonical_grad_strides_restores_foldable_grad():
    # Capture the raw gradient flowing INTO the projection via a tensor hook — a
    # mid-graph consumer (like LinearWithFrozenWeight.backward) sees exactly this
    # tensor. Leaf `.grad` would not do: AccumulateGrad re-lays-out the grad to
    # match the leaf's strides, hiding the stale-stride problem.
    S, V = 6, 10
    base = torch.randn(S, 1, V, requires_grad=True)

    def run(use_shim):
        captured = []
        x = base.clone()  # mid-graph tensor standing in for the projection output
        x.register_hook(captured.append)
        out = _projection_chain(x, use_shim=use_shim)
        # Freshly-allocated canonical grad, like the real upstream (the depad's
        # zeros+scatter). ones_like(out) would inherit the view's strides and
        # accidentally survive the transpose with a foldable layout.
        out.backward(torch.ones(out.shape))
        return captured[0]

    grad_no_shim = run(use_shim=False)
    grad_shim = run(use_shim=True)

    # The shim must not change gradient values, only the stride layout.
    assert torch.equal(grad_no_shim, grad_shim)
    # Without the shim the transpose-backward grad carries a stale stride in the
    # size-1 dim and would fall off matmul's mm fast path; with it, it folds.
    assert not _passes_matmul_fold_check(grad_no_shim)
    assert _passes_matmul_fold_check(grad_shim)


def test_canonical_grad_strides_identity_forward():
    x = torch.randn(4, 1, 8, requires_grad=True)
    y = _CanonicalGradStrides.apply(x)
    assert y.data_ptr() == x.data_ptr()
    assert torch.equal(y, x)


def test_canonical_grad_strides_handles_noncontiguous_grad():
    # A genuinely non-contiguous upstream grad (not just stale size-1 strides) must
    # come out contiguous and value-identical.
    x = torch.randn(5, 3, 4, requires_grad=True)
    y = _CanonicalGradStrides.apply(x)
    g = torch.randn(4, 3, 5).permute(2, 1, 0)  # non-contiguous grad
    assert not g.is_contiguous()
    y.backward(g)
    assert x.grad.is_contiguous()
    assert torch.equal(x.grad, g)
