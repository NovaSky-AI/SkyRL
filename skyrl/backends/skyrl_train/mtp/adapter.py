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

    The detached-weight projection (``LinearWithFrozenWeight``) backprops via ``grad.matmul(weight)``.
    At micro-batch 1 the grad is a transpose-backward view whose stale size-1 batch stride fails
    matmul's ``should_fold`` check, so it dispatches a broadcast ``bmm`` ~100x slower than the
    equivalent ``mm`` (measured 1.27s vs 10ms/microbatch on H100). Re-viewing the dense grad restores
    canonical strides at zero copy.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        if grad.is_contiguous():
            return grad.view(-1).view(grad.shape)
        return grad.contiguous()


def project_mtp_hidden_to_logits(hidden_states_per_layer, model) -> List:
    """Run the model's shared output layer on each captured MTP hidden-state chunk.

    Like the model's ``_postprocess``: output-layer logits come out ``[seq, batch, vocab/tp]`` and are
    transposed to ``[batch, seq, vocab/tp]``. The result is still in Megatron's internal (packed /
    left-removed) layout; the caller de-pads it with the same transform as the main logits.

    Takes one ``[seq, batch, hidden]`` tensor per MTP depth (from
    ``MTPHiddenCapture.compute_student_hidden_states``) and returns one ``[batch, seq, vocab/tp]``
    student-logits tensor each.

    The output weight is detached (tied weight, or the output layer's own on untied models -- either
    way a policy param), so the draft loss trains only the MTP-head params.
    """
    if getattr(model, "share_embeddings_and_output_weights", False):
        output_weight = model.shared_embedding_or_output_weight()
    else:
        output_weight = getattr(model.output_layer, "weight", None)
    if output_weight is not None:
        output_weight = output_weight.detach()

    logits_per_layer = []
    for hidden in hidden_states_per_layer:
        logits, _ = model.output_layer(hidden, weight=output_weight)
        logits = _CanonicalGradStrides.apply(logits)
        logits_per_layer.append(logits.transpose(0, 1).contiguous())
    return logits_per_layer
