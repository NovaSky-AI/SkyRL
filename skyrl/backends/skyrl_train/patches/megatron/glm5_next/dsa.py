"""GLM-5.3-Flash sparse attention on top of megatron-core's ``DSAttention``.

GLM-5.3-Flash's DSA layers are NoPE MLA (``qk_pos_emb_head_dim == 0``) with a *k-pool*
compressed indexer: keys are pooled in groups of ``index_kpool`` consecutive tokens, the
indexer scores and selects ``index_topk / index_kpool`` pools, the selected pools are expanded
back to token indices and the query's own incomplete tail pool is always appended. The pinned
megatron-core only has the token-level indexer, and its ``DSAttention.forward`` selects tokens
itself, so ``Glm5NextDSAttention`` swaps that selection for the vendored k-pool kernels
(``mcore_ext/dsa_kpool.py``, NVIDIA/Megatron-LM#7522) whenever ``dsa_indexer_kpool > 1``.
"""

from typing import Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAttention,
    fused_qk_topk_naive,
    rotate_activation,
)

from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.dsa_kpool import (
    fused_qk_topk_kpool,
)


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
        # ``index_kpool`` is an attribute of DSAIndexer, not of DSAttention, so read the config.
        index_kpool = int(getattr(self.config, "dsa_indexer_kpool", 1) or 1)
        if index_kpool <= 1:
            if max_seqlen > self.index_topk:
                raise NotImplementedError(
                    f"GLM-5.3-Flash sparse attention with sequences longer than dsa_indexer_topk="
                    f"{self.index_topk} tokens (got {max_seqlen}) needs the k-pool indexer, which the "
                    "Megatron backend does not implement yet."
                )
            return super().forward(
                query, key, value, attention_mask, x, qr, *args, packed_seq_params=packed_seq_params, **kwargs
            )
        return self._forward_with_kpool_topk(
            query, key, value, attention_mask, x, qr, *args, packed_seq_params=packed_seq_params, **kwargs
        )

    def _forward_with_kpool_topk(self, *args, packed_seq_params=None, **kwargs):
        """Run the pinned ``DSAttention.forward`` with its top-k step swapped for k-pool selection.

        The pinned ``DSAttention.forward`` picks tokens itself -- through the fused cuDNN DSA path,
        the fused indexer top-k, or ``fused_qk_topk_naive`` -- and never calls
        ``DSAIndexer.forward_with_scores``, so the pooled selection has to be injected here.
        Mirrors NVIDIA/Megatron-LM#7522's ``DSAttention.forward``: both fused token-level indexer
        paths decline (a supported fallback) and the naive top-k call runs ``fused_qk_topk_kpool``
        on the same q/k/weights, masks and varlen bounds. The fused sparse-attention kernel that
        consumes the indices is left alone. DELETE together with ``Glm5NextDSAIndexer``.
        """
        from megatron.core.transformer.experimental_attention_variant import (
            dsa as mcore_dsa,
        )
        from megatron.core.transformer.experimental_attention_variant import dsa_kernels

        if self.index_share:
            raise NotImplementedError("GLM-5.3-Flash k-pool DSA does not support cross-layer index sharing.")
        if self.training and (self.config.dsa_indexer_loss_coeff or 0.0) > 0:
            raise NotImplementedError("GLM-5.3-Flash k-pool DSA does not support the indexer loss.")

        indexer = self.indexer
        cu_seqlens_kv = None
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            _, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        kpool_calls = 0

        def kpool_topk(
            q,
            k,
            weights,
            index_topk,
            mask=None,
            varlen_starts=None,
            varlen_ends=None,
            key_positions=None,
            use_relu=True,
        ):
            nonlocal kpool_calls
            kpool_calls += 1
            if indexer._kpool_gate_score is None:
                raise RuntimeError("k-pool gate score was not computed by Glm5NextDSAIndexer.forward_before_topk")
            return fused_qk_topk_kpool(
                q,
                k,
                weights,
                index_topk,
                indexer.index_kpool,
                indexer._kpool_gate_score,
                indexer.index_kpool_compress_ape,
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                cu_seqlens_kv=cu_seqlens_kv,
                use_relu=use_relu,
                always_select_tail=indexer.index_kpool_always_select_tail,
            )

        def decline(*_args, **_kwargs):
            return None

        saved = (mcore_dsa.fused_qk_topk_naive, dsa_kernels.run_fused_dsa_attention, dsa_kernels.run_fused_qk_topk)
        mcore_dsa.fused_qk_topk_naive = kpool_topk
        dsa_kernels.run_fused_dsa_attention = decline
        dsa_kernels.run_fused_qk_topk = decline
        try:
            output = super().forward(*args, packed_seq_params=packed_seq_params, **kwargs)
        finally:
            mcore_dsa.fused_qk_topk_naive, dsa_kernels.run_fused_dsa_attention, dsa_kernels.run_fused_qk_topk = saved
        # A pinned-megatron-core change that routes top-k elsewhere must fail here rather than
        # silently fall back to token-level selection.
        if not self.skip_topk and kpool_calls != 1:
            raise RuntimeError(
                f"GLM-5.3-Flash k-pool selection ran {kpool_calls} times in one DSAttention.forward "
                "(expected 1); the pinned megatron-core top-k path has changed."
            )
        return output


class Glm5NextDSAIndexer(DSAIndexer):
    """``DSAIndexer`` with the k-pool compressed indexer from NVIDIA/Megatron-LM#7522.

    The pinned megatron-core has the token-level indexer only, which is exact just up to
    ``dsa_indexer_topk``. GLM-5.3-Flash ships ``index_kpool=4`` and trains past that, so the
    pooled path is required. The kernels live in :mod:`mcore_ext.dsa_kpool`, copied verbatim
    from #7522; this class is the wiring, hand-merged onto the pinned ``DSAIndexer`` because
    #7522 interleaves k-pool with NoPE and FP8 changes that the pin does not have and this
    model does not use.

    Deviations from #7522, both because the branches are dead for this checkpoint:

    - ``rotate_activation`` handling is unchanged. The bridge sets
      ``dsa_indexer_rotate_activation = False`` for GLM-5.3-Flash, so both sides skip it.
    - the FP8 indexer path (``dsa_indexer_kpool_fp8``) is not carried over; this runs bf16.

    DELETE THIS CLASS once the megatron-core pin includes #7522.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.index_kpool = int(getattr(self.config, "dsa_indexer_kpool", 1) or 1)
        self.index_kpool_always_select_tail = bool(getattr(self.config, "dsa_indexer_kpool_always_select_tail", True))
        # Per-token gate score for the kpool path; set in forward_before_topk.
        self._kpool_gate_score: Optional[torch.Tensor] = None
        if self.index_kpool > 1:
            # fp32 [kpool, index_head_dim] additive positional bias per pool slot.
            self.index_kpool_compress_ape = torch.nn.Parameter(
                torch.zeros(self.index_kpool, self.index_head_dim, dtype=torch.float32)
            )
            # bf16 [index_head_dim, hidden_size]; gate_score = F.linear(x, gate) = x @ gate^T
            # -> [seqlen, index_head_dim]. Matches vLLM's checkpoint name (no .weight suffix).
            self.index_kpool_compress_gate = torch.nn.Parameter(
                torch.empty(self.index_head_dim, self.hidden_size, dtype=torch.bfloat16)
            )
            torch.nn.init.normal_(self.index_kpool_compress_gate, std=0.01)
        else:
            self.index_kpool_compress_ape = None
            self.index_kpool_compress_gate = None

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``DSAIndexer.forward_before_topk`` plus the k-pool gate score.

        Body copied from the pinned ``DSAIndexer``; the only changes are the two k-pool blocks
        at the end, marked below. ``x`` is re-gathered for sequence parallel inside this method
        (as upstream does), so the gate score sees the same tokens as ``weights``.
        """
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(None, None, x, self.config, packed_seq_params)
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        if packed_seq:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        seqlen, bsz, _ = x.size()

        q, _ = self.linear_wq_b(qr)
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_q)

        k, _ = self.linear_wk(x)
        if self.config.dsa_indexer_k_norm_fp32:
            k_dtype = k.dtype
            k = self.k_norm(k.float()).to(dtype=k_dtype)
        else:
            k = self.k_norm(k)
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_kv)
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        if self.config.dsa_indexer_rotate_activation:
            q = rotate_activation(q)
            k = rotate_activation(k)

        # --- k-pool change 1: the pooled indexer keeps the head-gate projection in FP32. ---
        if self.index_kpool > 1:
            weights = torch.nn.functional.linear(x.float(), self.linear_weights_proj.weight.float())
        else:
            weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        # --- k-pool change 2: token-aligned compression scores for the pool selection. ---
        if self.index_kpool > 1 and self.index_kpool_compress_gate is not None:
            self._kpool_gate_score = torch.nn.functional.linear(x, self.index_kpool_compress_gate)
        else:
            self._kpool_gate_score = None

        return q, k, weights

    def forward_with_scores(self, x, qr, mask=None, packed_seq_params=None):
        """``DSAIndexer.forward_with_scores`` with the pooled selection branch from #7522."""
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        if self.index_kpool > 1 and self._kpool_gate_score is not None:
            # Select pools, then expand them to token indices.
            _cu_kv = None
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
                _cu_kv, _ = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            index_scores, topk_indices = fused_qk_topk_kpool(
                q,
                k,
                weights,
                self.index_topk,
                self.index_kpool,
                self._kpool_gate_score,
                self.index_kpool_compress_ape,
                mask=mask,
                cu_seqlens_kv=_cu_kv,
                use_relu=self.config.dsa_indexer_scoring_relu,
                always_select_tail=self.index_kpool_always_select_tail,
            )
        else:
            index_scores, topk_indices = fused_qk_topk_naive(
                q, k, weights, self.index_topk, mask, use_relu=self.config.dsa_indexer_scoring_relu
            )

        return index_scores, topk_indices
