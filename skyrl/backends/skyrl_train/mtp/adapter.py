# Backend seam for decoupled MTP/draft-head training.
#
# The loss (``soft_ce``), the combination (``draft_loss_wrapper``) and the
# capture mechanism (``hidden_capture``) are backend-agnostic. Only two things
# differ per backend: (a) where the trunk hidden states are captured and (b) how
# the draft head turns those hidden states into vocab logits. This module pins
# down that contract. Megatron is implemented today; FSDP is a planned follow-up
# that will implement the same protocol and reuse everything else unchanged.

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class DraftAdapter(Protocol):
    """Per-backend hooks for decoupled MTP training."""

    def capture_context(self, model, detach_trunk: bool): ...

    def project_to_logits(self, hidden_states_per_layer, model) -> List: ...


def project_mtp_hidden_to_logits(hidden_states_per_layer, gpt_model, detach_output_weight: bool = False) -> List:
    """Run the shared output layer on each captured MTP hidden-state chunk.

    Mirrors ``GPTModel._postprocess``: logits come out of the output layer in
    ``[seq, batch, vocab/tp]`` and are transposed to ``[batch, seq, vocab/tp]``
    to match the main-logits layout. The returned tensors are still in Megatron's
    *internal* (packed / left-removed) sequence layout; the caller applies the
    same de-padding transform used for the main logits so they align with the
    teacher.

    Args:
        hidden_states_per_layer: list of ``[seq, batch, hidden]`` tensors, one per
            MTP depth (from ``MTPHiddenCapture.compute_student_hidden_states``).
        gpt_model: the unwrapped ``GPTModel`` (exposes ``output_layer`` and, when
            embeddings are tied, ``shared_embedding_or_output_weight``).

    Returns:
        list of ``[batch, seq, vocab/tp]`` student-logits tensors (internal layout).
    """
    output_weight = None
    if getattr(gpt_model, "share_embeddings_and_output_weights", False):
        output_weight = gpt_model.shared_embedding_or_output_weight()
        if detach_output_weight:
            # Optionally isolate the shared embedding/output-layer from the draft
            # gradient too (default keeps NeMo's behaviour: only the trunk hidden
            # states are detached).
            output_weight = output_weight.detach()

    logits_per_layer = []
    for hidden in hidden_states_per_layer:
        logits, _ = gpt_model.output_layer(hidden, weight=output_weight)
        logits_per_layer.append(logits.transpose(0, 1).contiguous())
    return logits_per_layer
