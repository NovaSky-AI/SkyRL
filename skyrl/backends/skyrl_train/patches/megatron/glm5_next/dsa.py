"""GLM-5.3-Flash sparse attention on top of megatron-core's ``DSAttention``.

GLM-5.3-Flash's DSA layers are NoPE MLA (``qk_pos_emb_head_dim == 0``) with a *k-pool*
compressed indexer (NVIDIA/Megatron-LM#7522, in the pinned megatron-core): keys are pooled in
groups of ``dsa_indexer_kpool`` consecutive tokens, the indexer selects
``dsa_indexer_topk / dsa_indexer_kpool`` pools, and the query's own incomplete tail pool is
appended. ``Glm5NextDSAttention`` adds two things on top of megatron-core's ``DSAttention``:

- a guard: without k-pool the token-level indexer is exact only up to ``dsa_indexer_topk``;
- opt-in (``SKYRL_DSA_INDEXER_TP_SHARD=1``) tensor-parallel sharding of the k-pool scoring.
"""

import os
from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import dsa as mcore_dsa
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

# Opt-in: split the k-pool indexer's query rows across the tensor-parallel group (exact).
_DSA_INDEXER_TP_SHARD = os.environ.get("SKYRL_DSA_INDEXER_TP_SHARD", "0").lower() in ("1", "true")


def make_sharded_kpool_topk(kpool_topk_from_score_chunks, group):
    """Wrap megatron-core's ``_kpool_topk_from_score_chunks`` to score only this rank's query rows.

    Every rank of ``group`` must pass identical inputs (as the tensor-parallel group does: all
    ranks score the same gathered sequence with the same frozen indexer weights). Each rank
    selects pools for its contiguous slice of query rows and the selections are all-gathered.
    The scoring is O(sq * num_pools) and otherwise replicated on every rank; per query row
    nothing changes, so the result is bitwise identical to the unsharded call.
    """

    def sharded(*, q, mask, varlen_starts, varlen_ends, **kwargs):
        sq = q.size(0)
        world = group.size()
        rows_per_rank = -(-sq // world)
        q_lo = min(sq, group.rank() * rows_per_rank)
        q_hi = min(sq, q_lo + rows_per_rank)
        weights = kwargs.pop("weights")
        if mask is not None:
            mask = mask[q_lo:q_hi] if mask.ndim == 2 else mask[:, q_lo:q_hi]
        local = kpool_topk_from_score_chunks(
            q=q[q_lo:q_hi],
            weights=weights[q_lo:q_hi],
            mask=mask,
            varlen_starts=varlen_starts[q_lo:q_hi] if varlen_starts is not None else None,
            varlen_ends=varlen_ends[q_lo:q_hi] if varlen_ends is not None else None,
            **kwargs,
        )
        batch, _, budget = local.shape
        # Pool ids fit in int32; pad the (possibly short) last slice so shards are equal-sized.
        local = local.to(torch.int32)
        if local.shape[1] < rows_per_rank:
            local = torch.nn.functional.pad(local, (0, 0, 0, rows_per_rank - local.shape[1]), value=-1)
        # Concatenated (not stacked) output shape: the one every backend accepts.
        gathered = torch.empty((world * batch, rows_per_rank, budget), dtype=torch.int32, device=q.device)
        torch.distributed.all_gather_into_tensor(gathered, local.contiguous(), group=group)
        gathered = gathered.view(world, batch, rows_per_rank, budget)
        return gathered.permute(1, 0, 2, 3).reshape(batch, world * rows_per_rank, budget)[:, :sq].to(torch.int64)

    return sharded


class Glm5NextDSAttention(DSAttention):
    """``DSAttention`` for GLM-5.3-Flash."""

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
        if self.config.dsa_indexer_kpool <= 1:
            max_seqlen = self._max_sequence_length(x, packed_seq_params)
            if max_seqlen > self.index_topk:
                raise NotImplementedError(
                    f"GLM-5.3-Flash sparse attention with sequences longer than dsa_indexer_topk="
                    f"{self.index_topk} tokens (got {max_seqlen}) needs the k-pool indexer "
                    "(dsa_indexer_kpool > 1)."
                )
        shard_group = self._indexer_shard_group(query, x)
        if shard_group is None:
            return super().forward(
                query, key, value, attention_mask, x, qr, *args, packed_seq_params=packed_seq_params, **kwargs
            )
        saved = mcore_dsa._kpool_topk_from_score_chunks
        mcore_dsa._kpool_topk_from_score_chunks = make_sharded_kpool_topk(saved, shard_group)
        try:
            return super().forward(
                query, key, value, attention_mask, x, qr, *args, packed_seq_params=packed_seq_params, **kwargs
            )
        finally:
            mcore_dsa._kpool_topk_from_score_chunks = saved

    def _indexer_shard_group(self, query: torch.Tensor, x: torch.Tensor):
        """The TP group when ``SKYRL_DSA_INDEXER_TP_SHARD`` applies, else ``None``.

        Sharding is only exact when every TP rank scores the same full sequence. megatron-core
        already slices the indexer to the local rows when the attention queries are
        sequence-local, so in that layout there is nothing replicated to shard.
        """
        if not _DSA_INDEXER_TP_SHARD or self.config.dsa_indexer_kpool <= 1:
            return None
        tp_group = self.pg_collection.tp
        if tp_group.size() <= 1:
            return None
        if self.config.sequence_parallel and query.size(0) == x.size(0):
            return None
        return tp_group
