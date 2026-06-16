# Output-head projection for decoupled MTP/draft-head training.
#
# The loss (``soft_ce``) and the capture mechanism (``hidden_capture``) are
# backend-agnostic. This module holds the Megatron-specific piece: turning the
# captured MTP hidden states into vocab logits via the shared output layer. It
# works for any model that ships native MTP heads regardless of base class
# (``GPTModel`` for DeepSeek/GLM/Qwen3-Next, ``MambaModel`` for Qwen3.5/NemotronH).

from __future__ import annotations

from typing import List

import torch


class _CanonicalGradStrides(torch.autograd.Function):
    """Identity forward; backward hands upstream a stride-canonical gradient.

    The draft projection runs through megatron's ``LinearWithFrozenWeight`` (the
    detached-weight path), whose hand-written dgrad is ``grad_output.matmul(weight)``.
    ``torch.matmul`` only folds that 3D x 2D product into a flat ``mm`` when the grad
    passes ``should_fold``'s stride check, which does NOT skip size-1 dims. With
    micro-batch 1, the grad reaching it is a transpose-backward view whose size-1 batch
    dim carries a stale stride (the adapter's ``.contiguous()`` no-ops on such views),
    so matmul falls back to a broadcast ``bmm`` (batch=seq, M=1) that runs ~100x slower
    than the equivalent ``mm`` — measured 1.27s vs 10ms per microbatch at
    [8344, 1, 62080] x [62080, 4096] on H100. Re-viewing the (dense) grad rewrites the
    strides to canonical form at zero copy and restores the ``mm`` dispatch. The
    non-detached path is unaffected either way (``should_fold`` folds early when the
    weight requires grad), and the no-copy view keeps this shim free there too.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        if grad.is_contiguous():
            return grad.view(-1).view(grad.shape)
        return grad.contiguous()


def project_mtp_hidden_to_logits(hidden_states_per_layer, model, detach_output_weight: bool = False) -> List:
    """Run the model's shared output layer on each captured MTP hidden-state chunk.

    Mirrors the model's own ``_postprocess``: logits come out of the output layer
    in ``[seq, batch, vocab/tp]`` and are transposed to ``[batch, seq, vocab/tp]``
    to match the main-logits layout. The returned tensors are still in Megatron's
    *internal* (packed / left-removed) sequence layout; the caller applies the
    same de-padding transform used for the main logits so they align with the
    teacher.

    Args:
        hidden_states_per_layer: list of ``[seq, batch, hidden]`` tensors, one per
            MTP depth (from ``MTPHiddenCapture.compute_student_hidden_states``).
        model: the unwrapped Megatron model (``GPTModel``/``MambaModel``/...),
            which exposes ``output_layer`` and, when embeddings are tied,
            ``shared_embedding_or_output_weight``.

    Returns:
        list of ``[batch, seq, vocab/tp]`` student-logits tensors (internal layout).
    """
    output_weight = None
    if getattr(model, "share_embeddings_and_output_weights", False):
        output_weight = model.shared_embedding_or_output_weight()
    if detach_output_weight:
        # Isolate the output projection from the draft gradient: detach the shared/tied weight,
        # or — for untied models — pass the output layer's own weight explicitly as a detached
        # tensor (ColumnParallelLinear uses a provided ``weight`` verbatim). Mirrors slime's
        # MTP-RL megatron patch. Combined with the capture's detached re-embedding, the draft
        # loss then trains only the MTP-head parameters.
        if output_weight is None:
            output_weight = getattr(model.output_layer, "weight", None)
        if output_weight is not None:
            output_weight = output_weight.detach()

    logits_per_layer = []
    for hidden in hidden_states_per_layer:
        logits, _ = model.output_layer(hidden, weight=output_weight)
        logits = _CanonicalGradStrides.apply(logits)
        logits_per_layer.append(logits.transpose(0, 1).contiguous())
    return logits_per_layer
