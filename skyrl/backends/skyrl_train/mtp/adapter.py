# Backend seam for decoupled MTP/draft-head training.
#
# The loss (``soft_ce``), the combination (``draft_loss_wrapper``) and the
# capture mechanism (``hidden_capture``) are backend-agnostic. Only two things
# differ per backend / draft style: (a) where the trunk hidden states are
# captured and (b) how the draft head turns those hidden states into vocab
# logits. ``DraftAdapter`` pins down that contract.
#
# Implemented today: native-MTP draft training on Megatron (capture the model's
# built-in ``self.mtp`` block and project through the shared output layer), which
# works for any model that ships native MTP heads regardless of base class
# (``GPTModel`` for DeepSeek/GLM/Qwen3-Next, ``MambaModel`` for Qwen3.5/NemotronH).
#
# Planned follow-ups implement the SAME protocol and reuse the loss/combination
# unchanged: (1) a NeMo-RL-style Eagle adapter that captures auxiliary policy
# hidden states and projects them through a separate Eagle draft model (for models
# WITHOUT native MTP heads); (2) an FSDP backend adapter.

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class DraftAdapter(Protocol):
    """Per-backend / per-draft-style hooks for decoupled MTP/draft training.

    An Eagle adapter would implement the same two methods: ``capture_context``
    grabs the auxiliary policy hidden states, and ``project_to_logits`` runs the
    (separate) draft model's output head."""

    def capture_context(self, model, detach_trunk: bool): ...

    def project_to_logits(self, hidden_states_per_layer, model) -> List: ...


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
        logits_per_layer.append(logits.transpose(0, 1).contiguous())
    return logits_per_layer
