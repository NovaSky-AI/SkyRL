"""Megatron modules for the Qwen4Exp-specific pieces of Qwen3.8-Flash-Next.

Parameter names mirror the HF ``Qwen4ExpText*`` modules so the bridge maps them one to one.
All of them assume ``tensor_model_parallel_size == 1`` and no context parallelism (checked by
the provider): the hyper-connection and PLE parameters are replicated, and PLE needs the whole,
contiguous token stream of every packed sequence.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub import hf_hub_download
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.transformer_config import TransformerConfig
from safetensors import safe_open
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class ZeroCenteredRMSNorm(nn.Module):
    """HF ``Qwen4ExpTextRMSNorm``: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``, optionally per group."""

    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        if self.group_size is not None:
            out = out.unflatten(-1, (-1, self.group_size))
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            out = out.flatten(-2)
        return (out * (1.0 + self.weight.float())).type_as(x)


class Qwen4ExpGatedResidual(nn.Module):
    """HF ``Qwen4ExpTextGatedResidual``: the low-rank gated hyper-connection.

    Reads the ``hc_count``-stream residual ``x`` ([..., n*C]) and returns the single-stream
    sub-layer input, plus (``use_combine``) the per-stream injection weights for the update
    ``x + y ⊗ inject``.
    """

    def __init__(self, config: TransformerConfig, use_combine: bool = True):
        super().__init__()
        n, c = config.qwen4_hc_count, config.hidden_size
        self.hc_count, self.hidden_size = n, c
        self.hc_norm = ZeroCenteredRMSNorm(n * c, group_size=c, eps=config.layernorm_epsilon)
        self.input_mix_weight_down = nn.Linear(n * c, config.qwen4_hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.qwen4_hc_lowrank, n * c, bias=False)
        self.block_inject_weight = nn.Linear(n * c, n, bias=False) if use_combine else None

    def forward(self, x: Tensor):
        normed = self.hc_norm(x)
        mix = F.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up(mix)).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (mix * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        return mixed, 2 * torch.sigmoid(self.block_inject_weight(normed) / self.hc_count)

    @staticmethod
    def inject(residual: Tensor, update: Tensor, weights: Tensor) -> Tensor:
        """``residual + concat_streams(update * weights)``."""
        return residual + (update.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


class Qwen4ExpFinalMixer(Qwen4ExpGatedResidual):
    """The block's final "layer norm": HF ``hyper_connection_mixer`` (no inject, no final RMSNorm).

    Built from the block spec's ``layer_norm`` slot, so it accepts (and ignores) the
    ``hidden_size`` / ``eps`` arguments of that slot; it contracts ``[s, b, n*C]`` to ``[s, b, C]``.
    """

    def __init__(self, config: TransformerConfig, hidden_size: int = None, eps: float = None, **kwargs):
        super().__init__(config, use_combine=False)


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    """megatron-core GDN with a sigmoid output gate.

    megatron-core uses ``config.activation_func`` (SiLU here) both for the short conv and for the
    output gate. Qwen4Exp keeps SiLU on the conv but gates with ``output_gate_type`` (sigmoid), so
    only the unfused gated norm is overridden (the bridge disables the fused one).
    """

    def _apply_gated_norm_unfused(self, x: Tensor, gate: Tensor) -> Tensor:
        y = self.out_norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return (y * torch.sigmoid(gate.float())).to(x.dtype)


# ------------------------------------------------------------------------------------------------
# PLE (per-layer n-gram embedding)
# ------------------------------------------------------------------------------------------------


def _segment_shift(tokens: Tensor, seg_start: Tensor, eos: int, shift: int) -> Tensor:
    """HF ``_shift_right_ignore_eos`` on a packed stream.

    ``seg_start[t]`` is where t's EOS-delimited segment begins (packed-sequence starts count as a
    preceding EOS). Positions whose shifted source leaves the segment read EOS.
    """
    if shift == 0:
        return tokens
    positions = torch.arange(tokens.shape[0], device=tokens.device)
    src = (positions - shift).clamp_min(0)
    return torch.where(positions - seg_start >= shift, tokens[src], tokens.new_full((), eos))


class Qwen4ExpPLE(nn.Module):
    """HF ``Qwen4ExpTextPLELayer`` for packed (THD) sequences.

    The hashed n-gram table (``ple_embedding.ngram_embedding``, ~51B parameters) is frozen. It is
    sharded by n-gram head across ``shard_group`` (the expert-parallel group, whose ranks already
    run every forward in lockstep for the MoE all-to-all) and is loaded straight from the HF
    safetensors shards rather than through the bridge, which would materialize all of it. The
    shard lives in pinned host memory and is looked up on the CPU (a few MB per micro-batch).

    Forward expects ``set_tokens(input_ids, cu_seqlens)`` to have been called for the current
    micro-batch (a forward pre-hook on the GPTModel does it, see ``provider.py``).
    """

    def __init__(self, config: TransformerConfig, ple_layer_index: int):
        super().__init__()
        c, n = config.hidden_size, config.qwen4_hc_count
        e = config.qwen4_ple_embed_dim
        self.hidden_size, self.hc_count = c, n
        self.ngram_size = config.qwen4_ngram_size
        self.heads_per_ngram = config.qwen4_heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim = e // self.ngram_heads
        self.eos = config.qwen4_eos_token_id
        self.ple_layer_index = ple_layer_index
        self.dilation = self.ngram_size
        self.kernel = config.qwen4_ple_conv_kernel_size

        self.key_proj = nn.Linear(e, n * c, bias=False)
        self.value_proj = nn.Linear(e, c, bias=False)
        self.norm_key = ZeroCenteredRMSNorm(n * c, group_size=c, eps=config.layernorm_epsilon)
        self.norm_query = ZeroCenteredRMSNorm(n * c, group_size=c, eps=config.layernorm_epsilon)
        self.norm_conv = ZeroCenteredRMSNorm(n * c, group_size=c, eps=config.layernorm_epsilon)
        self.conv1d = nn.Conv1d(n * c, n * c, kernel_size=self.kernel, groups=n * c, dilation=self.dilation, bias=False)

        self._tokens: Optional[Tensor] = None
        self._cu_seqlens: Optional[Tensor] = None
        self._table: Optional[Tensor] = None
        self._group = None
        self._hf_path = config.qwen4_hf_path
        self._hf_prefix = config.qwen4_ple_hf_prefix

    # -- table -----------------------------------------------------------------------------------
    def load_table(self, group) -> None:
        """Load this rank's n-gram heads (and the hashing buffers) from the HF checkpoint."""
        self._group = group
        size = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0
        if self.ngram_heads % size:
            raise ValueError(f"PLE shard group size {size} must divide the {self.ngram_heads} n-gram heads.")
        self.heads_per_rank = self.ngram_heads // size
        heads = range(rank * self.heads_per_rank, (rank + 1) * self.heads_per_rank)

        def local(filename):
            # The bridge passes the model as given: a local directory or a hub repo id. The latter
            # only fetches the index and the PLE files, through the HF cache.
            if os.path.isdir(self._hf_path):
                return os.path.join(self._hf_path, filename)
            return hf_hub_download(self._hf_path, filename)

        with open(local("model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        prefix = f"{self._hf_prefix}ple_embedding."

        def read(name):
            with safe_open(local(weight_map[prefix + name]), framework="pt") as f:
                return f.get_tensor(prefix + name)

        dev = torch.cuda.current_device()
        multipliers, sizes, offsets = (
            read("layer_multipliers"),
            read("ngram_heads_vocab_sizes"),
            read("ngram_heads_offsets"),
        )
        # Rows owned by this rank: the contiguous range covering its heads.
        row_lo = int(offsets[heads[0]])
        row_hi = int(offsets[heads[-1]] + sizes[heads[-1]])
        shard_names = sorted(
            (k for k in weight_map if k.startswith(prefix + "ngram_embedding.shard_")),
            key=lambda k: int(k.split("shard_")[1].split(".")[0]),
        )
        # Pinned host memory, as vLLM does: the shard (~6.4 GB/rank at EP=16) is frozen and is not a
        # parameter, so model offload would never move it off the GPU during rollout / weight sync.
        table = torch.empty(row_hi - row_lo, self.head_dim, dtype=torch.bfloat16, pin_memory=True)
        start = 0
        for name in shard_names:
            with safe_open(local(weight_map[name]), framework="pt") as f:
                sl = f.get_slice(name)
                rows = sl.get_shape()[0]
                lo, hi = max(start, row_lo), min(start + rows, row_hi)
                if lo < hi:
                    table[lo - row_lo : hi - row_lo] = sl[lo - start : hi - start].to(torch.bfloat16)
                start += rows
        self.set_table(table, row_lo, multipliers.to(dev), sizes.to(dev), offsets.to(dev))

    def set_table(self, table: Tensor, row_lo: int, multipliers: Tensor, sizes: Tensor, offsets: Tensor) -> None:
        """Install this rank's rows ``[row_lo, row_lo + len(table))`` and the hashing constants."""
        if not hasattr(self, "heads_per_rank"):
            self.heads_per_rank = self.ngram_heads
        self.register_buffer("layer_multipliers", multipliers, persistent=False)
        self.register_buffer("head_vocab_sizes", sizes, persistent=False)
        self.register_buffer("head_offsets", offsets, persistent=False)
        self.row_lo = row_lo
        self._table = table  # deliberately not an nn.Parameter: frozen, never trained or synced

    # -- per-micro-batch context ------------------------------------------------------------------
    def set_tokens(self, tokens: Tensor, cu_seqlens: Optional[Tensor]) -> None:
        self._tokens = tokens.reshape(-1).long()
        self._cu_seqlens = cu_seqlens

    def _segment_starts(self, tokens: Tensor) -> Tensor:
        t = tokens.shape[0]
        positions = torch.arange(t, device=tokens.device)
        boundary = torch.where(tokens == self.eos, positions, -1)
        if self._cu_seqlens is not None:
            starts = self._cu_seqlens[1:-1].long()
            starts = starts[(starts > 0) & (starts < t)]
            boundary[starts - 1] = starts - 1  # a packed-sequence start acts like a preceding EOS
        prev = torch.cummax(boundary, dim=0).values
        prev = torch.cat([prev.new_full((1,), -1), prev[:-1]])
        return prev + 1

    def _ngram_ids(self, tokens: Tensor) -> Tensor:
        seg_start = self._segment_starts(tokens)
        shifted = [_segment_shift(tokens, seg_start, self.eos, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for p in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            sizes = self.head_vocab_sizes[lo : lo + self.heads_per_ngram]
            blocks.append(
                torch.remainder(mixed.unsqueeze(-1), sizes) + self.head_offsets[lo : lo + self.heads_per_ngram]
            )
        return torch.cat(blocks, dim=-1)  # [T, heads] global row ids

    def _embed(self, local_ids: Tensor) -> Tensor:
        """Rows of the local (possibly host-resident) table, on ``local_ids``' device."""
        return F.embedding(local_ids.to(self._table.device), self._table).to(local_ids.device)

    def _lookup(self, ids: Tensor) -> Tensor:
        """Embed ``ids`` ([T, heads]) with the head-sharded table; returns [T, heads * head_dim]."""
        size = dist.get_world_size(self._group) if self._group is not None else 1
        if size == 1:
            return self._embed(ids - self.row_lo).flatten(-2)
        rank = dist.get_rank(self._group)
        # Every rank needs its own tokens' rows from every head; exchange ids, look up the local
        # heads, then all-to-all the embeddings back. Lengths differ per rank: pad to the max.
        t = torch.tensor([ids.shape[0]], device=ids.device)
        lens = [torch.empty_like(t) for _ in range(size)]
        dist.all_gather(lens, t, group=self._group)
        lens = [int(x) for x in lens]
        tmax = max(lens)
        padded = F.pad(ids, (0, 0, 0, tmax - ids.shape[0]))
        gathered = torch.empty(size * tmax, ids.shape[1], dtype=ids.dtype, device=ids.device)
        dist.all_gather_into_tensor(gathered, padded, group=self._group)
        hp = self.heads_per_rank
        local = gathered[:, rank * hp : (rank + 1) * hp] - self.row_lo
        local = local.clamp(0, self._table.shape[0] - 1)
        emb = self._embed(local)  # [size * tmax, hp, d], chunk r belongs to rank r
        out = torch.empty_like(emb)
        dist.all_to_all_single(out, emb.contiguous(), group=self._group)
        # out chunk r = rank r's heads for our tokens
        out = out.view(size, tmax, hp, self.head_dim).permute(1, 0, 2, 3).reshape(tmax, -1)
        return out[: ids.shape[0]]

    def _short_conv(self, x: Tensor, seg_start_seq: Tensor) -> Tensor:
        """Causal dilated depthwise conv per packed sequence; x is [T, n*C]."""
        t = x.shape[0]
        positions = torch.arange(t, device=x.device)
        w = self.conv1d.weight.squeeze(1)  # [n*C, K]
        out = x * w[:, -1]
        for k in range(self.kernel - 1):
            shift = (self.kernel - 1 - k) * self.dilation
            src = (positions - shift).clamp_min(0)
            valid = (positions - seg_start_seq >= shift).unsqueeze(-1)
            out = out + torch.where(valid, x[src], 0) * w[:, k]
        return F.silu(out)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return the PLE update for ``hidden_states`` ([T, 1, n*C] packed)."""
        if self._tokens is None:
            raise RuntimeError("Qwen4ExpPLE.forward called without set_tokens().")
        s, b, nc = hidden_states.shape
        if b != 1 or self._tokens.shape[0] != s:
            raise NotImplementedError(
                "Qwen4Exp PLE needs packed sequences (remove_microbatch_padding=true), TP=1 and CP=1; "
                f"got hidden {tuple(hidden_states.shape)} for {self._tokens.shape[0]} tokens."
            )
        x = hidden_states.view(s, nc)
        tokens = self._tokens
        emb = self._lookup(self._ngram_ids(tokens)).to(x.dtype)

        key = self.norm_key(self.key_proj(emb)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(emb)
        query = self.norm_query(x).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        normed = self.norm_conv(gated)

        positions = torch.arange(s, device=x.device)
        if self._cu_seqlens is not None:
            starts = self._cu_seqlens[:-1].long()
            seq_start = starts[torch.searchsorted(starts, positions, right=True) - 1]
        else:
            seq_start = torch.zeros_like(positions)
        out = gated + self._short_conv(normed, seq_start)
        return out.view(s, b, nc)


# ------------------------------------------------------------------------------------------------
# QSA (Qwen sparse attention) indexer and the attention over its selection
# ------------------------------------------------------------------------------------------------

# TODO: replace the indexer with a fused kernel (cuDNN's ``qsa_select_forward``, or megatron-core's
# once NVIDIA/Megatron-LM#7393 / #7553 land) after checking its top-k ties against HF.

# Query rows per chunk of the indexer scores ([rows, heads, blocks] fp32) and of the masked
# attention ([heads, rows, keys]): bounds peak memory independently of the sequence length.
QSA_INDEXER_CHUNK = 1024
QSA_ATTENTION_CHUNK = 512
# CUDA fast path for sequences in the sparse regime: FlexAttention with the QSA selection as its
# mask (~3x faster than the chunked SDPA path at 8K tokens, which only gets SDPA's math kernel for
# head_dim 256 with a boolean mask). Sequences are padded to a multiple of QSA_FLEX_PAD so the
# compiled kernel sees few distinct shapes. SKYRL_QWEN4_QSA_FLEX=0 forces the SDPA path.
QSA_FLEX_PAD = 2048
_QSA_FLEX = os.environ.get("SKYRL_QWEN4_QSA_FLEX", "1") != "0"
_flex_attention = None
# Blackwell fast path: cuDNN frontend's QSA kernels (NVIDIA/cudnn-frontend#1765; SM100/SM103, BF16,
# head dim 256, 24Q/2KV or 12Q/1KV, 4-token blocks), 1.7-3.5x FlexAttention fwd+bwd at 8-10K
# tokens. Needs a cudnn-frontend with ``cudnn.ops.qsa_forward`` (>= 1.32; until it is on PyPI,
# the GitHub dev wheel via ``uv run --with``); without it, or elsewhere, FlexAttention runs.
# SKYRL_QWEN4_QSA_CUDNN=0 forces FlexAttention.
_QSA_CUDNN = os.environ.get("SKYRL_QWEN4_QSA_CUDNN", "1") != "0"
_CUDNN_QSA_HEADS = {(24, 2), (12, 1)}
_cudnn_qsa_ops = None


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """HF ``apply_rotary_pos_emb``: rotate-half RoPE on the leading ``cos.shape[-1]`` dims."""
    d = cos.shape[-1]
    x_rot, x_pass = x[..., :d], x[..., d:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    return torch.cat((x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin, x_pass), dim=-1)


@dataclass
class QSASequenceSelection:
    """The keys every query of one packed sequence attends to.

    ``block_mask[t, b]`` is True iff key block ``b`` (tokens ``[b*r, (b+1)*r)`` of the sequence)
    is selected for query ``t``; query ``t`` also always sees its incomplete tail
    ``[((t+1)//r)*r, t]``. ``None`` means every visible key is selected (causal dense).
    """

    start: int  # first token in the packed stream
    length: int  # real (unpadded) tokens
    block_mask: Optional[Tensor]  # [length, length // r] bool


@dataclass
class QSASelection:
    sequences: List[QSASequenceSelection]
    compress_ratio: int


class Qwen4ExpQSAIndexer(nn.Module):
    """HF ``Qwen4ExpTextQSAIndexer``: selects, per query, the keys of a full-attention layer.

    For query ``t`` of a sequence (0-based, causal), the visible tokens ``0..t`` split into
    ``(t+1)//r`` complete blocks of ``r = indexer_compress_ratio`` tokens plus an incomplete tail.
    Each block is pooled (fp32 mean of the raw indexer keys, ``k_layernorm``, RoPE at the block's
    first position) and scored as ``sum_h relu(q_h . k_block) / sqrt(d)``; the top
    ``indexer_budget // r`` blocks and the tail are selected. With at most ``indexer_budget // r``
    complete blocks every key is selected, so sequences of up to ``indexer_budget + r - 1`` tokens
    are exactly dense (``select`` returns None and the layer keeps the fused dense kernel). Note the
    query itself is only seen when it is in the tail or in a selected block, as in HF.

    The weights are frozen and the selection is non-differentiable (as in HF); gradients flow
    through the attention over the selected keys (``qsa_sparse_attention``).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        d = config.qwen4_indexer_head_dim
        if config.qwen4_indexer_kv_heads != 1:
            raise ValueError("The Qwen4Exp QSA indexer requires indexer_kv_heads=1 (as HF).")
        self.n_heads, self.head_dim = config.qwen4_indexer_n_heads, d
        self.compress_ratio = config.qwen4_indexer_compress_ratio
        self.block_topk = config.qwen4_indexer_budget // self.compress_ratio
        self.max_dense_length = config.qwen4_indexer_budget + self.compress_ratio - 1
        self.index_qk_proj = nn.Linear(config.hidden_size, (self.n_heads + 1) * d, bias=False)
        self.q_layernorm = ZeroCenteredRMSNorm(d, eps=config.layernorm_epsilon)
        self.k_layernorm = ZeroCenteredRMSNorm(d, eps=config.layernorm_epsilon)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def select_sequence(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Block mask ([L, L // r] bool) of one sequence; ``x`` is [L, C], cos/sin [L, rot] in x's dtype."""
        length, r, d = x.shape[0], self.compress_ratio, self.head_dim
        q, k = F.linear(x, self.index_qk_proj.weight).split([self.n_heads * d, d], dim=-1)
        q = _rope(self.q_layernorm(q.reshape(length, self.n_heads, d)), cos.unsqueeze(1), sin.unsqueeze(1))
        nb = length // r
        pooled = k[: nb * r].reshape(nb, r, d).float().mean(dim=1).to(k.dtype)
        keys = _rope(self.k_layernorm(pooled), cos[: nb * r : r], sin[: nb * r : r]).float()

        blocks = torch.arange(nb, device=x.device)
        block_mask = torch.empty(length, nb, dtype=torch.bool, device=x.device)
        for c0 in range(0, length, QSA_INDEXER_CHUNK):
            c1 = min(length, c0 + QSA_INDEXER_CHUNK)
            visible = (torch.arange(c0, c1, device=x.device) + 1) // r  # complete blocks per query
            complete = blocks < visible.unsqueeze(-1)
            if c1 // r > self.block_topk:  # some query in the chunk sees more blocks than the budget
                scores = torch.matmul(q[c0:c1].float(), keys.T).relu_().sum(dim=1) / math.sqrt(d)
                # Incomplete blocks score -inf; every relu score is >= 0, so they are never picked.
                top = scores.masked_fill_(~complete, float("-inf")).topk(self.block_topk, dim=-1).indices
                selected = torch.zeros_like(complete).scatter_(-1, top, True)
                complete = torch.where((visible > self.block_topk).unsqueeze(-1), selected, complete)
            block_mask[c0:c1] = complete
        return block_mask

    @torch.no_grad()
    def select(self, hidden_states: Tensor, rotary_pos_emb, packed_seq_params) -> Optional[QSASelection]:
        """Selection for the attention input ``hidden_states`` ([T, 1, C] packed), or None if dense.

        ``rotary_pos_emb`` is megatron-core's RoPE angle table ([>=L, 1, 1, rot], or a (q, k) pair
        of them); positions restart at 0 in every packed sequence, as in the main attention.
        """
        s, b, _ = hidden_states.shape
        if packed_seq_params is not None:
            cu = packed_seq_params.cu_seqlens_q
            cu_padded = packed_seq_params.cu_seqlens_q_padded
            cu_padded = cu if cu_padded is None else cu_padded
            starts, lengths = cu_padded[:-1].tolist(), (cu[1:] - cu[:-1]).tolist()
        else:
            starts, lengths = [0], [s]
        if max(lengths) <= self.max_dense_length:
            return None
        if b != 1:
            raise NotImplementedError(
                f"Qwen4Exp QSA sparse attention (sequences over {self.max_dense_length} tokens) needs packed "
                "sequences (remove_microbatch_padding=true)."
            )
        if rotary_pos_emb is None:
            raise NotImplementedError("Qwen4Exp QSA sparse attention needs the RoPE angle table (rotary_pos_emb).")
        freqs = rotary_pos_emb[0] if isinstance(rotary_pos_emb, tuple) else rotary_pos_emb
        freqs = freqs.reshape(freqs.shape[0], -1)
        if freqs.shape[0] < max(lengths):
            raise ValueError(f"RoPE table of {freqs.shape[0]} positions for a {max(lengths)}-token sequence.")
        x = hidden_states.reshape(s, -1)
        sequences = []
        for start, length in zip(starts, lengths):
            block_mask = None
            if length > self.max_dense_length:
                f = freqs[:length].float()
                # HF casts cos/sin to the activation dtype before applying them.
                block_mask = self.select_sequence(x[start : start + length], f.cos().to(x.dtype), f.sin().to(x.dtype))
            sequences.append(QSASequenceSelection(start, length, block_mask))
        return QSASelection(sequences, self.compress_ratio)


def _qsa_token_mask(block_mask: Optional[Tensor], c0: int, c1: int, r: int, device) -> Tensor:
    """[c1 - c0, c1] bool: keys ``0..c1-1`` of the sequence each query ``c0..c1-1`` attends to."""
    rows = torch.arange(c0, c1, device=device).unsqueeze(-1)
    cols = torch.arange(c1, device=device)
    causal = cols <= rows
    if block_mask is None:
        return causal
    tail = cols >= (rows + 1) // r * r
    nb = block_mask.shape[-1]
    in_block = block_mask[c0:c1][:, (cols // r).clamp_max(nb - 1)] & (cols < nb * r)
    return causal & (in_block | tail)


def _masked_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float) -> Tensor:
    """Softmax attention of q ([Q, np, d]) over k / v ([np, K, d]) under ``mask`` ([Q, K])."""
    out = F.scaled_dot_product_attention(q.transpose(0, 1), k, v, attn_mask=mask, scale=scale)
    return out.transpose(0, 1)


def _compiled_flex_attention():
    global _flex_attention
    if _flex_attention is None:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention

        # One compiled graph per padded length (QSA_FLEX_PAD steps).
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        _flex_attention = torch.compile(flex_attention, dynamic=False)
    return _flex_attention


def _flex_sequence_attention(q: Tensor, k: Tensor, v: Tensor, block_mask: Tensor, r: int, scale: float) -> Tensor:
    """``qsa_sparse_attention`` of one sequence (q [L, np, d], k / v [L, ng, d]) with FlexAttention."""
    from torch.nn.attention.flex_attention import create_block_mask

    length, nb = q.shape[0], block_mask.shape[1]
    padded = -(-length // QSA_FLEX_PAD) * QSA_FLEX_PAD
    pad = padded - length
    # Padded queries keep key 0 (so no softmax row is empty); their outputs are dropped.
    bm = F.pad(block_mask, (0, padded // r - nb, 0, pad))
    # Per-sequence sizes enter the mask as tensors, not Python ints: torch.compile would guard on
    # (and recompile for) every distinct int captured by mask_mod.
    real_len = torch.tensor(length, device=q.device)
    block_end = torch.tensor(nb * r, device=q.device)

    def mask_mod(b, h, qi, kv):
        tail = (kv >= (qi + 1) // r * r) & (kv < real_len)
        in_block = bm[qi, kv // r] & (kv < block_end)
        return ((kv <= qi) & (in_block | tail)) | ((qi >= real_len) & (kv == 0))

    mask = create_block_mask(mask_mod, None, None, padded, padded, device=q.device)
    q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out = _compiled_flex_attention()(q, k, v, block_mask=mask, scale=scale, enable_gqa=True)
    return out[0, :, :length].transpose(0, 1)


def _cudnn_qsa_usable(query: Tensor, key: Tensor, value: Tensor, compress_ratio: int) -> bool:
    """Whether the cuDNN QSA kernels support these packed q / k / v ([T, heads, d]) on this device."""
    global _cudnn_qsa_ops
    if not (_QSA_CUDNN and query.is_cuda and query.dtype == torch.bfloat16 and compress_ratio == 4):
        return False
    if query.shape[-1] != 256 or value.shape[-1] != 256 or (query.shape[1], key.shape[1]) not in _CUDNN_QSA_HEADS:
        return False
    if torch.cuda.get_device_capability(query.device)[0] != 10:
        return False
    if _cudnn_qsa_ops is None:
        try:
            from cudnn.ops import qsa_backward, qsa_forward

            _cudnn_qsa_ops = (qsa_forward, qsa_backward)
            # WARNING, not INFO: Megatron workers drop INFO, and the recipe log should show which path ran.
            logger.warning("Qwen4Exp QSA sparse attention: using cuDNN frontend qsa_forward / qsa_backward.")
        except (ImportError, AttributeError):
            _cudnn_qsa_ops = False
            logger.warning(
                "Qwen4Exp QSA sparse attention: cudnn-frontend has no cudnn.ops.qsa_forward (needs >= 1.32); "
                "falling back to the FlexAttention path."
            )
    return bool(_cudnn_qsa_ops)


def cudnn_qsa_metadata(selection: QSASelection, num_tokens: int, device) -> tuple:
    """The cuDNN QSA selection tensors for ``selection`` over a packed stream of ``num_tokens``.

    Returns ``(sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last)`` (cudnn-frontend
    ``docs/fe-oss-apis/qsa.md``): query ``t`` sees key ``j`` iff ``lo[t] <= j <= hi[t]`` and block
    ``blk_base[t] + (j - tok_base[t]) // r`` is set in ``sel_blocks[t]``. Every packed span (a
    sequence and the padding after it) numbers its own blocks; real queries see ``[start, t]``,
    padding queries none (``lo > hi``, zero output). The incomplete causal tail block is set
    explicitly: the kernel does not add it.
    """
    r = selection.compress_ratio
    seqs = selection.sequences
    # (span start, span end, real tokens, block mask or None for dense); a leading gap is all padding.
    spans = [(0, seqs[0].start, 0, None)] if seqs[0].start > 0 else []
    ends = [seq.start for seq in seqs[1:]] + [num_tokens]
    spans += [(seq.start, end, seq.length, seq.block_mask) for seq, end in zip(seqs, ends)]
    blocks = [-(-(end - start) // r) for start, end, _, _ in spans]
    nb_total = sum(blocks)
    i32 = dict(dtype=torch.int32, device=device)
    sel = torch.zeros(num_tokens, nb_total, dtype=torch.uint8, device=device)
    lo, hi = torch.empty(num_tokens, **i32), torch.empty(num_tokens, **i32)
    blk_base, tok_base = torch.empty(num_tokens, **i32), torch.empty(num_tokens, **i32)
    block_first, block_last = torch.empty(nb_total, **i32), torch.empty(nb_total, **i32)
    b0 = 0
    for (start, end, n, block_mask), nb in zip(spans, blocks):
        u = torch.arange(end - start, device=device)
        lo[start:end] = torch.where(u < n, start, start + u)
        hi[start:end] = torch.where(u < n, start + u, start + u - 1)
        blk_base[start:end] = b0
        tok_base[start:end] = start
        first = start + r * torch.arange(nb, device=device)
        block_first[b0 : b0 + nb] = first
        block_last[b0 : b0 + nb] = (first + r - 1).clamp_max(end - 1)
        rows = u[:n]
        if block_mask is None:  # dense: every block up to the query's own
            sel[start : start + n, b0 : b0 + nb] = (torch.arange(nb, device=device) <= (rows // r)[:, None]).to(
                torch.uint8
            )
        elif n:
            sel[start : start + n, b0 : b0 + block_mask.shape[1]] = block_mask.to(torch.uint8)
            tail = rows[(rows + 1) % r != 0]  # queries whose own block is still incomplete
            sel[start + tail, b0 + tail // r] = 1
        b0 += nb
    return sel, lo, hi, blk_base, tok_base, block_first, block_last


class _CudnnQSA(torch.autograd.Function):
    """``cudnn.ops.qsa_forward`` / ``qsa_backward`` (which install no autograd) as one op."""

    @staticmethod
    def forward(ctx, query, key, value, scale, *metadata):
        qsa_forward, _ = _cudnn_qsa_ops
        out, out_fp32, lse = qsa_forward(query, key, value, *metadata, scale)
        ctx.save_for_backward(query, key, value, out_fp32, lse, *metadata)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        _, qsa_backward = _cudnn_qsa_ops
        query, key, value, out_fp32, lse, *metadata = ctx.saved_tensors
        dq, dk, dv = qsa_backward(grad_out.contiguous(), query, key, value, out_fp32, lse, *metadata, ctx.scale)
        return (dq, dk, dv, None) + (None,) * len(metadata)


def qsa_sparse_attention(query: Tensor, key: Tensor, value: Tensor, selection: QSASelection, scale: float) -> Tensor:
    """Causal attention restricted to the QSA selection, for packed (THD) q / k / v.

    ``query`` is [T, np, d] and ``key`` / ``value`` [T, ng, d]; returns [T, np * d] (zeros on the
    padding between sequences; [T, 1, ...] in and out for one unpacked sequence). On Blackwell with
    cuDNN's QSA kernels available, the whole pack runs through them (``_cudnn_qsa_usable``).
    Otherwise sparse-regime sequences use FlexAttention on CUDA, and the rest (or every sequence
    with SKYRL_QWEN4_QSA_FLEX=0) chunked SDPA: queries in chunks of ``QSA_ATTENTION_CHUNK`` over
    the keys before the chunk end, each chunk recomputed in backward (non-reentrant checkpoint),
    so the [heads, chunk, keys] scores of only one chunk are alive at a time. The selection never
    leaves a query without keys (at least one block or the tail), so no softmax row is empty.
    """
    batched = query.dim() == 4  # [T, 1, np, d]: unpacked single sequence
    if batched:
        query, key, value = (t.squeeze(1) for t in (query, key, value))
    num_tokens, num_heads, _ = query.shape
    if _cudnn_qsa_usable(query, key, value, selection.compress_ratio):
        metadata = cudnn_qsa_metadata(selection, num_tokens, query.device)
        q, k, v = (t.contiguous() for t in (query, key, value))
        out = _CudnnQSA.apply(q, k, v, scale, *metadata).reshape(num_tokens, -1)
        return out.unsqueeze(1) if batched else out
    out_dim = value.shape[-1]
    checkpoint = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    pieces, pos = [], 0
    for seq in selection.sequences:
        if seq.start > pos:
            pieces.append(query.new_zeros(seq.start - pos, num_heads, out_dim))
        q, k, v = (t[seq.start : seq.start + seq.length] for t in (query, key, value))
        if _QSA_FLEX and query.is_cuda and seq.block_mask is not None:
            pieces.append(_flex_sequence_attention(q, k, v, seq.block_mask, selection.compress_ratio, scale))
            pos = seq.start + seq.length
            continue
        # GQA: query head h reads key group h // (np / ng) (megatron-core and HF ``repeat_kv``).
        k, v = (t.repeat_interleave(num_heads // t.shape[1], dim=1).transpose(0, 1).contiguous() for t in (k, v))
        for c0 in range(0, seq.length, QSA_ATTENTION_CHUNK):
            c1 = min(seq.length, c0 + QSA_ATTENTION_CHUNK)
            mask = _qsa_token_mask(seq.block_mask, c0, c1, selection.compress_ratio, query.device)
            args = (q[c0:c1], k[:, :c1], v[:, :c1], mask, scale)
            if checkpoint:
                pieces.append(torch.utils.checkpoint.checkpoint(_masked_attention, *args, use_reentrant=False))
            else:
                pieces.append(_masked_attention(*args))
        pos = seq.start + seq.length
    if pos < num_tokens:
        pieces.append(query.new_zeros(num_tokens - pos, num_heads, out_dim))
    out = torch.cat(pieces).reshape(num_tokens, num_heads * out_dim)
    return out.unsqueeze(1) if batched else out
