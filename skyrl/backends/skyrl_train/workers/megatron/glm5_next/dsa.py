"""GLM-5.3-Flash sparse attention on top of megatron-core's ``DSAttention``.

GLM-5.3-Flash's DSA layers are NoPE MLA (``qk_pos_emb_head_dim == 0``) with a *k-pool*
compressed indexer: keys are pooled in groups of ``index_kpool`` consecutive tokens, the
indexer scores and selects ``index_topk / index_kpool`` pools, the selected pools are expanded
back to token indices and the query's own incomplete tail pool is always appended. For any
sequence of at most ``index_topk`` tokens every pool is selectable, so the selection covers all
causally visible tokens and sparse attention equals dense causal attention. megatron-core's
token-level indexer with ``dsa_indexer_topk = index_topk`` gives the same all-visible selection
in that regime, so it is reused as is; the pool-level scoring only changes which tokens are
dropped once a sequence exceeds ``index_topk``, which this module does not implement yet and
refuses instead of silently attending to a different subset.
"""

from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention


class Glm5NextDSAttention(DSAttention):
    """``DSAttention`` for GLM-5.3-Flash: exact for sequences up to ``dsa_indexer_topk`` tokens."""

    def _max_sequence_length(self, x: torch.Tensor, packed_seq_params: Optional[PackedSeqParams]) -> int:
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            max_seqlen = packed_seq_params.max_seqlen_q
            if isinstance(max_seqlen, int):
                return max_seqlen
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        tp_size = self.pg_collection.tp.size() if self.config.sequence_parallel else 1
        return x.size(0) * tp_size

    def forward(self, query, key, value, attention_mask, x, qr, *args, packed_seq_params=None, **kwargs):
        max_seqlen = self._max_sequence_length(x, packed_seq_params)
        if max_seqlen > self.index_topk:
            raise NotImplementedError(
                f"GLM-5.3-Flash sparse attention with sequences longer than dsa_indexer_topk="
                f"{self.index_topk} tokens (got {max_seqlen}) needs the k-pool indexer, which the "
                "Megatron backend does not implement yet."
            )
        return super().forward(
            query, key, value, attention_mask, x, qr, *args, packed_seq_params=packed_seq_params, **kwargs
        )
