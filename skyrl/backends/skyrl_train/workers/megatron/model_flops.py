"""Model FLOPs of a training batch, from megatron-bridge's per-layer-type formula.

``num_floating_point_operations`` costs each layer by its type (dense/GQA, MLA, DSA, GDN, KDA,
Mamba, MoE, ...) from the finalized provider, so hybrid stacks are not costed as uniform
transformers. Two SkyRL-side adjustments on top:

- DSA k-pool indexer (GLM-5.3-Flash, ``dsa_indexer_kpool > 1``): the indexer scores pooled keys,
  ``1/kpool`` of the dense causal context bridge assumes; the excess is subtracted.
- HFU: under full activation recompute, every recomputed decoder layer runs its forward again.
"""

from types import SimpleNamespace
from typing import Any, Dict, Optional


def _indexer_layer_count(provider: Any, layer_count: int) -> int:
    """Counts the layers that compute DSA top-k indices rather than reuse them, as bridge does."""
    freq = getattr(provider, "dsa_indexer_topk_freq", 1) or 1
    offset = max(getattr(provider, "dsa_indexer_skip_topk_offset", 0) or 0, 1)
    return sum(max(n - offset, 0) % freq == 0 for n in range(1, layer_count + 1))


def kpool_indexer_overcount(provider: Any, seqlen_sum: int, seqlen_squared_sum: int) -> float:
    """Returns the FLOPs bridge counts for DSA index scoring beyond what a k-pool indexer computes.

    Bridge costs index scoring over the dense causal context of each query; with k-pool
    compression the indexer scores ``ceil(context / kpool)`` pooled keys instead.

    Args:
        provider: Finalized Megatron model provider.
        seqlen_sum: Sum of the batch's document lengths.
        seqlen_squared_sum: Sum of the squared document lengths.

    Returns:
        The FLOPs to subtract from bridge's count; 0 for models without a pooled DSA indexer.
    """
    kpool = getattr(provider, "dsa_indexer_kpool", 1) or 1
    if kpool <= 1 or getattr(provider, "experimental_attention_variant", None) != "dsa" or seqlen_sum <= 0:
        return 0.0
    num_layers = provider.num_layers
    num_indexer_layers = _indexer_layer_count(provider, num_layers)
    # Bridge's KDA branch keeps only the DSA layers' share of the attention term.
    kda_layers = getattr(provider, "kimi_kda_layers", None)
    share = (num_layers - len(set(kda_layers))) / num_layers if kda_layers else 1.0

    core_attn_seq_factor = seqlen_squared_sum / seqlen_sum
    dense_context = (core_attn_seq_factor + 1) / 2
    topk = provider.dsa_indexer_topk
    if core_attn_seq_factor <= topk:
        sparse_context = dense_context
    else:
        sparse_context = topk - topk * (topk - 1) / (2 * core_attn_seq_factor)
    unit = provider.dsa_indexer_n_heads * (provider.dsa_indexer_head_dim + 1)

    # Same terms as bridge's ``index_score_term``: forward scoring always, plus the score gradient
    # when the indexer is trained by its auxiliary loss.
    per_token = 2 * num_indexer_layers * dense_context * unit
    if (getattr(provider, "dsa_indexer_loss_coeff", 0.0) or 0.0) > 0:
        use_sparse = getattr(provider, "dsa_indexer_use_sparse_loss", False)
        per_token += 4 * num_indexer_layers * (sparse_context if use_sparse else dense_context) * unit
    return (1 - 1 / kpool) * share * per_token * seqlen_sum


def full_recompute_fraction(provider: Any) -> float:
    """Returns the fraction of decoder layers recomputed in backward; 0 unless recompute is full."""
    if getattr(provider, "recompute_granularity", None) != "full":
        return 0.0
    if getattr(provider, "recompute_method", None) == "block":
        pp = getattr(provider, "pipeline_model_parallel_size", 1) or 1
        layers_per_stage = provider.num_layers / pp
        return min(1.0, (getattr(provider, "recompute_num_layers", 0) or 0) / layers_per_stage)
    # "uniform" recomputes every layer, in chunks of recompute_num_layers.
    return 1.0


def logits_flops(provider: Any, seqlen_sum: int) -> float:
    """Returns the forward + backward FLOPs of the output layer over ``seqlen_sum`` tokens."""
    from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size

    padded_vocab_size = calculate_padded_vocab_size(
        provider.vocab_size,
        provider.make_vocab_size_divisible_by,
        provider.tensor_model_parallel_size,
        logging_enabled=False,
    )
    return 6.0 * provider.hidden_size * padded_vocab_size * seqlen_sum


def model_flops(provider: Any, num_seqs: int, seqlen_sum: int, seqlen_squared_sum: int) -> Optional[Dict[str, float]]:
    """Returns the forward + backward model FLOPs of a batch with these per-document statistics.

    The count assumes full fine-tuning and no active MTP heads; callers skip both.

    Args:
        provider: Finalized Megatron model provider.
        num_seqs: Number of documents in the batch.
        seqlen_sum: Sum of the document lengths.
        seqlen_squared_sum: Sum of the squared document lengths.

    Returns:
        ``{"flops": ...}``, plus ``"hfu_flops"`` (model FLOPs + recomputed forwards) when activation
        recompute is full; None for an empty batch.
    """
    if num_seqs <= 0 or seqlen_sum <= 0:
        return None
    from megatron.bridge.training.utils.flop_utils import (
        num_floating_point_operations,
    )

    # Only ``cfg.model`` is read outside bridge's LoRA branch, which needs ``cfg.peft``.
    flops = float(
        num_floating_point_operations(
            SimpleNamespace(model=provider),
            batch_size=num_seqs,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=seqlen_squared_sum,
        )
    )
    flops -= kpool_indexer_overcount(provider, seqlen_sum, seqlen_squared_sum)
    result = {"flops": flops}
    fraction = full_recompute_fraction(provider)
    if fraction > 0:
        # Recompute repeats the decoder layers' forward (a third of their fwd+bwd), not the logits.
        result["hfu_flops"] = flops + fraction * (flops - logits_flops(provider, seqlen_sum)) / 3
    return result
