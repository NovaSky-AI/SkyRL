"""Megatron modules for the Qwen4Exp-specific pieces of Qwen3.8-Flash-Next.

Parameter names mirror the HF ``Qwen4ExpText*`` modules so the bridge maps them one to one.
All of them assume ``tensor_model_parallel_size == 1`` and no context parallelism (checked by
the provider): the hyper-connection and PLE parameters are replicated, and PLE needs the whole,
contiguous token stream of every packed sequence.
"""

import json
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.transformer_config import TransformerConfig
from safetensors import safe_open
from torch import Tensor, nn


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

    megatron-core gates the GDN output with ``config.activation_func``, which is also the MLP
    activation (SiLU here); Qwen4Exp's ``output_gate_type`` is ``sigmoid``.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.act_fn = torch.sigmoid
        self.activation = "sigmoid"


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
    safetensors shards rather than through the bridge, which would materialize all of it.

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

        with open(os.path.join(self._hf_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        prefix = f"{self._hf_prefix}ple_embedding."

        def read(name):
            with safe_open(os.path.join(self._hf_path, weight_map[prefix + name]), framework="pt") as f:
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
        table = torch.empty(row_hi - row_lo, self.head_dim, dtype=torch.bfloat16, device=dev)
        start = 0
        for name in shard_names:
            with safe_open(os.path.join(self._hf_path, weight_map[name]), framework="pt") as f:
                sl = f.get_slice(name)
                rows = sl.get_shape()[0]
                lo, hi = max(start, row_lo), min(start + rows, row_hi)
                if lo < hi:
                    table[lo - row_lo : hi - row_lo] = sl[lo - start : hi - start].to(dev, torch.bfloat16)
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
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + self.head_offsets[lo : lo + self.heads_per_ngram])
        return torch.cat(blocks, dim=-1)  # [T, heads] global row ids

    def _lookup(self, ids: Tensor) -> Tensor:
        """Embed ``ids`` ([T, heads]) with the head-sharded table; returns [T, heads * head_dim]."""
        size = dist.get_world_size(self._group) if self._group is not None else 1
        if size == 1:
            return F.embedding(ids - self.row_lo, self._table).flatten(-2)
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
        emb = F.embedding(local, self._table)  # [size * tmax, hp, d], chunk r belongs to rank r
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


class Qwen4ExpIndexerParams(nn.Module):
    """Frozen holder for the QSA indexer weights of a full-attention layer.

    The indexer only selects which keys a query may attend to (block top-k over pooled keys,
    non-differentiable). With at most ``indexer_budget`` (+ compress_ratio - 1) visible tokens per
    query it selects every key, so attention is exactly dense; the provider enforces that bound.
    The weights are kept so checkpoints round-trip.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        d = config.qwen4_indexer_head_dim
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (config.qwen4_indexer_n_heads + config.qwen4_indexer_kv_heads) * d, bias=False
        )
        self.q_layernorm = ZeroCenteredRMSNorm(d, eps=config.layernorm_epsilon)
        self.k_layernorm = ZeroCenteredRMSNorm(d, eps=config.layernorm_epsilon)
        for p in self.parameters():
            p.requires_grad_(False)
