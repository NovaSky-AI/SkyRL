"""DeepSeek-V4.1 attention on top of megatron-core's DeepSeek-V4 (``dsv4_hybrid``) attention.

Parity reference: ``inference/model.py`` (``Attention``, ``Compressor``, ``Indexer``,
``select_candidate_blocks``) in the DeepSeek-V4.1-Flash-1017 checkpoint. Deltas from V4:

- No per-head RMS norm on the query after ``wq_b``.
- Compress ratios are 0 (window only), 1 and 2 instead of 0 / 4 / 128. Every compressed layer, ratio 1
  included, rotates with ``compress_rope_theta`` + YaRN (V4 keys that off ``ratio > 1``).
- The compressor has no APE and no overlapping windows; ratio 1 is a plain ``norm(wkv(x))``.
- Cross-layer sharing: only ``kv_source_layers`` own a compressor; every later compressed layer reads the
  latest source's compressed KV. Only ``index_source_layers`` run an indexer; the layers between reuse the
  latest source's top-k. Index keys come from the source compressor's pre-RoPE latent via ``wk`` +
  ``k_norm`` (V4's indexer had its own compressor and a Hadamard rotation).
- Two-level top-k: the ``candidate_source_layer`` keeps its ``candidate_topk_blocks`` best blocks of
  ``candidate_block_size`` compressed positions; later index sources pick their top-k inside those blocks.

The indexer only selects positions, so it runs without autograd (no indexer loss in RL). The compressed
KV a source publishes stays in the autograd graph, so consumer layers backpropagate into the source's
compressor. The reference's FP8/FP4 activation fake-quant (KV cache, indexer q/k) is not reproduced,
like megatron-core's V4.

Interface follows upstream megatron-core's V4.1 work (NVIDIA/Megatron-LM#8224 "CSA2", stacked under #8226; open,
not in the pinned megatron-core), so moving to it later is mostly deleting this file:

- ``config.dsv4_version == "v4.1"`` gates the V4.1 deltas; layer roles are the ``csa2_kv_source_layers`` /
  ``csa2_index_source_layers`` / ``csa2_candidate_source_layer`` / ``csa2_candidate_topk_blocks`` /
  ``csa2_candidate_block_size`` config fields (set by the bridge; logical DeepSeek layer ids). Upstream keeps the
  V4.1 ratios in ``csa_compress_ratios``; the pinned ``TransformerConfig`` only accepts 0 / 4 / 128 there (the bridge
  stores V4 placeholders to build the layer pattern), so the real ratios are ``config.csa2_compress_ratios``.
- ``CompressedSparseAttention2`` / ``CSA2Compressor`` / ``CSA2Indexer`` and the shared ``CSA2State`` use upstream's
  names. Upstream passes the state as a ``csa2_state`` forward argument from the stack; the pinned
  ``DSv4HybridSelfAttention`` does not forward one, so it defaults to the model forward's state (``forward_state``).

Megatron's hybrid stack splits each DeepSeek block into an attention layer and a MoE layer, so logical layer
``(layer_number - 1) // 2`` owns attention hybrid layer ``layer_number``.

Beyond upstream: selective recompute (``mla_up_proj`` recompute keeps V4.1's q path) and the optional KV
fake-quant below.
"""

import copy
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as dsv4_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    _apply_rope,
    _pool_compressor_values,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    build_flat_topk_idxs,
    csa_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import build_module
from torch import nn

# Queries scored per indexer chunk: [chunk, b, n_heads, n_compressed] fp32 stays ~0.3 GB at 10K tokens.
_INDEXER_CHUNK = 256

# Optional KV fake-quant, straight-through for gradients (SKYRL_DSV41_FAKE_QUANT):
# - "vllm": vLLM's fp8_ds_mla KV cache -- the 448 NoPE dims in FP8 e4m3 per 64 with power-of-2 scales, the
#   64 RoPE dims kept in BF16 -- for window and compressed KV alike. Matches the rollout engine.
# - "reference" (or "1"): DeepSeek's inference/model.py -- window KV FP8 per 32 over all 512 dims, compressed
#   KV FP4 e2m1 per 16 with e4m3 scales.
FAKE_QUANT_KV = {"1": "reference"}.get(
    os.environ.get("SKYRL_DSV41_FAKE_QUANT", ""), os.environ.get("SKYRL_DSV41_FAKE_QUANT", "")
)
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _straight_through(x, y):
    return x + (y - x).detach()


def fake_quant_fp8_block(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Reference ``act_quant(x, block, scale_fmt='ue8m0', inplace=True)``."""
    with torch.no_grad():
        v = x.float().unflatten(-1, (-1, block))
        amax = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        y = ((v / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale).flatten(-2).to(x.dtype)
    return _straight_through(x, y)


def fake_quant_fp8_ds_mla(x: torch.Tensor, nope_dim: int = 448, block: int = 64) -> torch.Tensor:
    """vLLM's ``fp8_ds_mla`` KV cache row: FP8 NoPE dims with power-of-2 scales, BF16 RoPE dims."""
    nope, rope = x[..., :nope_dim], x[..., nope_dim:]
    return torch.cat([fake_quant_fp8_block(nope, block), rope], dim=-1)


def fake_quant_fp4_e4m3(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Reference ``fp4_act_quant(x, block, inplace=True, scale_dtype=float8_e4m3fn)``."""
    with torch.no_grad():
        v = x.float().unflatten(-1, (-1, block))
        amax = v.abs().amax(dim=-1, keepdim=True).clamp_min(6 * 2**-9)
        scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
        q = (v / scale).clamp(-6.0, 6.0)
        grid = torch.tensor(_FP4_GRID, device=x.device)
        idx = torch.bucketize(q.abs(), (grid[1:] + grid[:-1]) / 2)
        y = (torch.copysign(grid[idx], q) * scale).flatten(-2).to(x.dtype)
    return _straight_through(x, y)


@dataclass
class CSA2State:
    """Attention tensors source layers hand to later layers within one model forward (upstream ``CSA2State``,
    NVIDIA/Megatron-LM#8224; the reference's ``SharedAttentionRuntime``). Same field names and meaning as upstream
    for the subset used here. Layers run in order and every source writes before its consumers read."""

    global_kv: Optional[torch.Tensor] = None  # [n_c, b, head_dim], RoPE'd, in the autograd graph
    compress_ratio: Optional[int] = None
    indexer_k: Optional[torch.Tensor] = None  # [n_c, b, index_head_dim], RoPE'd, detached
    global_indices: Optional[torch.Tensor] = None  # [b, sq, topk] int32 compressed positions, -1 = none
    candidates: Optional[torch.Tensor] = None  # [sq, b, n_c] bool


class ForwardState:
    """Per-model-forward state, reset when ``HybridModel.forward`` starts (``model.py``): the attention's
    ``CSA2State`` plus the model-level mHC pre-mix and Engram hashes (upstream keeps those with the stack, #8226)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.csa2 = CSA2State()
        self.pending_pre = None  # [s, b, hc] mHC pre-mix for the next sublayer
        self.engram_hashes = None  # [s, b, n_engram_layers, cols]


_state = threading.local()


def forward_state() -> ForwardState:
    if not hasattr(_state, "value"):
        _state.value = ForwardState()
    return _state.value


def is_v41(config) -> bool:
    return getattr(config, "dsv4_version", None) == "v4.1"


def logical_layer_id(layer_number: int) -> int:
    return (layer_number - 1) // 2


@torch.no_grad()
def select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int):
    """Level one of the two-level top-k (reference ``select_candidate_blocks``).

    ``logits`` is ``[..., n_positions]`` with unreachable positions at -inf; ``compress_lens`` broadcasts
    against its leading dims. Returns a bool mask shaped like ``logits``.
    """
    width = logits.size(-1)
    scores = nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    # The block holding the query's newest position is only partly filled: pin it in.
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _rms_norm_module(config, hidden_size):
    norm_config = copy.copy(config)
    norm_config.normalization = "RMSNorm"
    from megatron.core.extensions.transformer_engine import TENorm

    return build_module(TENorm, config=norm_config, hidden_size=hidden_size, eps=config.layernorm_epsilon)


def _duplicated_linear(config, in_features, out_features, name):
    from megatron.core.extensions.transformer_engine import TELinear

    return build_module(
        TELinear,
        in_features,
        out_features,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        parallel_mode="duplicated",
        name=name,
    )


class CSA2Compressor(MegatronModule):
    """Pools ``compress_ratio`` consecutive tokens into one KV latent (reference ``Compressor``).

    Returns the normalized latent before RoPE, ``[sq // ratio, b, head_dim]``, or None when the sequence
    is shorter than one group.
    """

    def __init__(self, config, compress_ratio: int, head_dim: int, name=None):
        super().__init__(config=config)
        self.compress_ratio = compress_ratio
        self.linear_wkv = _duplicated_linear(config, config.hidden_size, head_dim, f"{name}.linear_wkv")
        self.linear_wgate = (
            _duplicated_linear(config, config.hidden_size, head_dim, f"{name}.linear_wgate")
            if compress_ratio > 1
            else None
        )
        self.norm = _rms_norm_module(config, head_dim)

    def forward(self, x: torch.Tensor):
        sq, b, _ = x.size()
        ratio = self.compress_ratio
        if sq < ratio:
            return None
        kv, _ = self.linear_wkv(x)
        if ratio == 1:
            return self.norm(kv)
        score, _ = self.linear_wgate(x)
        n_c = sq // ratio
        kv = kv[: n_c * ratio].view(n_c, ratio, b, -1)
        score = score[: n_c * ratio].view(n_c, ratio, b, -1)
        # Softmax pooling in fp32, as the reference does.
        return self.norm(_pool_compressor_values(kv, score, x.dtype))


class CSA2Indexer(MegatronModule):
    """Scores compressed positions for each query and keeps the top-k (reference ``Indexer``)."""

    def __init__(self, config, owns_k: bool, name=None):
        super().__init__(config=config)
        self.n_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.softmax_scale = self.head_dim**-0.5
        self.linear_wq_b = _duplicated_linear(
            config, config.q_lora_rank, self.n_heads * self.head_dim, f"{name}.linear_wq_b"
        )
        self.linear_weights_proj = _duplicated_linear(
            config, config.hidden_size, self.n_heads, f"{name}.linear_weights_proj"
        )
        self.owns_k = owns_k
        if owns_k:
            self.linear_wk = _duplicated_linear(config, config.v_head_dim, self.head_dim, f"{name}.linear_wk")
            self.k_norm = _rms_norm_module(config, self.head_dim)
        # As megatron-core main's DSA indexer: top-k selection runs under no_grad, so without an indexer loss these
        # parameters never get a gradient; keep them out of DDP / optimizer bookkeeping (overlap_grad_reduce would
        # otherwise wait on hooks that never fire).
        if (getattr(config, "dsa_indexer_loss_coeff", None) or 0.0) <= 0:
            for param in self.parameters():
                param.requires_grad_(False)


class CompressedSparseAttention2(MegatronModule):
    """Sliding-window attention plus, on compressed layers, ``index_topk`` positions of the shared
    compressed KV. Constructor signature matches megatron-core's ``CompressedSparseAttention`` so the V4
    attention module can build it; ``compress_ratio`` from the hybrid symbol is ignored in favour of
    the ``csa2_*`` config fields."""

    def __init__(
        self,
        config,
        submodules=None,
        layer_number: int = 1,
        attn_mask_type=None,
        attention_type: str = "self",
        attention_dropout=None,
        softmax_scale=None,
        k_channels=None,
        v_channels=None,
        cp_comm_type="p2p",
        pg_collection=None,
        rotary_pos_emb=None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
        name=None,
    ):
        super().__init__(config=config)
        assert not is_mtp_layer, "DeepSeek-V4.1 DSpark MTP layers are not supported"
        assert pg_collection is not None and pg_collection.tp.size() == 1, "DeepSeek-V4.1 attention is TP=1 only"
        assert pg_collection.cp.size() == 1, "DeepSeek-V4.1 attention does not support context parallelism"
        self.pg_collection = pg_collection
        assert is_v41(config), "CompressedSparseAttention2 requires config.dsv4_version == 'v4.1'"
        # Cross-layer sharing goes through one per-forward state (forward_state), which full / block recompute
        # (layers re-run in backward) and pipeline microbatch interleaving would read stale. megatron-core rejects
        # both for mHC models today; upstream's explicit CSA2 state (NVIDIA/Megatron-LM#7678) is the general fix.
        assert config.recompute_granularity != "full", "DeepSeek-V4.1 attention does not support full recompute"
        assert (config.pipeline_model_parallel_size or 1) == 1, "DeepSeek-V4.1 attention requires trainer PP=1"
        self.layer_id = logical_layer_id(layer_number)
        self.compress_ratio = config.csa2_compress_ratios[self.layer_id]
        self.is_kv_source = self.compress_ratio > 0 and self.layer_id in config.csa2_kv_source_layers
        self.is_index_source = self.compress_ratio > 0 and self.layer_id in config.csa2_index_source_layers
        candidate_source = config.csa2_candidate_source_layer  # None disables hierarchical candidates
        self.is_candidate_source = candidate_source is not None and self.layer_id == candidate_source
        self.uses_candidates = candidate_source is not None and candidate_source < self.layer_id
        self.candidate_topk_blocks = config.csa2_candidate_topk_blocks
        self.candidate_block_size = config.csa2_candidate_block_size
        self.window_size = config.csa_window_size
        self.num_attention_heads = config.num_attention_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else config.v_head_dim**-0.5
        self.use_fused_kernels = use_fused_dsa_kernels(config)
        # Set by DSv41HybridSelfAttention (it owns the compressed-RoPE module).
        self.rotary_pos_emb = rotary_pos_emb
        self.attn_sink = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.num_attention_heads, dtype=torch.float32)))
        self.compressor = (
            CSA2Compressor(config, self.compress_ratio, config.v_head_dim, name=f"{name}.compressor")
            if self.is_kv_source
            else None
        )
        self.indexer = (
            CSA2Indexer(config, owns_k=self.is_kv_source, name=f"{name}.indexer") if self.is_index_source else None
        )

    def _rope(self, x, n, ratio):
        pos_dim = self.config.qk_pos_emb_head_dim
        return _apply_rope(
            x,
            x.size(-1) - pos_dim,
            pos_dim,
            self.rotary_pos_emb,
            self.config,
            n,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

    @torch.no_grad()
    def _select_topk(self, x, qr, state, sq, n_c):
        """Indexer top-k over the shared compressed positions: ``[b, sq, topk]`` int32, -1 = none."""
        indexer = self.indexer
        ratio = self.compress_ratio
        q, _ = indexer.linear_wq_b(qr.detach())
        q = q.view(sq, q.size(1), indexer.n_heads, indexer.head_dim)
        q = self._rope(q, sq, 1).float()
        weights, _ = indexer.linear_weights_proj(x.detach())
        weights = weights.float() * (indexer.softmax_scale * indexer.n_heads**-0.5)
        k = state.indexer_k.float()  # [n_c, b, d]
        topk = min(indexer.index_topk, n_c)
        positions = torch.arange(n_c, device=x.device)
        candidates = [] if self.is_candidate_source else None
        out = []
        for s0 in range(0, sq, _INDEXER_CHUNK):
            s1 = min(s0 + _INDEXER_CHUNK, sq)
            scores = torch.einsum("sbhd,tbd->sbht", q[s0:s1], k).relu_()
            scores = (scores * weights[s0:s1].unsqueeze(-1)).sum(dim=2)  # [c, b, n_c]
            compress_lens = (torch.arange(s0 + 1, s1 + 1, device=x.device) // ratio).view(-1, 1, 1)
            scores.masked_fill_(positions >= compress_lens, -torch.inf)
            if self.is_candidate_source:
                candidates.append(
                    select_candidate_blocks(
                        scores, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
                    )
                )
            elif self.uses_candidates:
                scores.masked_fill_(~state.candidates[s0:s1], -torch.inf)
            vals, idxs = scores.topk(topk, dim=-1)
            out.append(torch.where(torch.isfinite(vals) & (idxs < compress_lens), idxs, -1))
        if self.is_candidate_source:
            state.candidates = torch.cat(candidates, dim=0)
        return torch.cat(out, dim=0).permute(1, 0, 2).to(torch.int32).contiguous()

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        x=None,
        qr=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        csa2_state: Optional[CSA2State] = None,
    ):
        assert attention_mask is None, "DeepSeek-V4.1 attention uses an implicit causal mask"
        assert packed_seq_params is None, "packed sequences are not supported for DeepSeek-V4.1 attention"
        sq, b, _, _ = query.size()
        kv = key.squeeze(-2)
        if FAKE_QUANT_KV == "vllm":
            kv = fake_quant_fp8_ds_mla(kv, kv.size(-1) - self.config.qk_pos_emb_head_dim)
        elif FAKE_QUANT_KV == "reference":
            kv = fake_quant_fp8_block(kv)
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)
        compressed_idxs = None
        kv_full = kv
        if self.compress_ratio:
            # Upstream's stack passes csa2_state; the pinned DSv4HybridSelfAttention does not, so use this forward's.
            state = csa2_state if csa2_state is not None else forward_state().csa2
            ratio = self.compress_ratio
            if self.is_kv_source:
                latent = self.compressor(x)
                state.compress_ratio = ratio
                if latent is None:
                    state.global_kv = state.indexer_k = None
                else:
                    n_c = latent.size(0)
                    if self.indexer is not None:
                        with torch.no_grad():
                            k, _ = self.indexer.linear_wk(latent.detach())
                            state.indexer_k = self._rope(self.indexer.k_norm(k), n_c, ratio)
                    state.global_kv = self._rope(latent, n_c, ratio)
                    if FAKE_QUANT_KV == "vllm":
                        state.global_kv = fake_quant_fp8_ds_mla(
                            state.global_kv, state.global_kv.size(-1) - self.config.qk_pos_emb_head_dim
                        )
                    elif FAKE_QUANT_KV == "reference":
                        state.global_kv = fake_quant_fp4_e4m3(state.global_kv)
            assert (
                state.compress_ratio == ratio
            ), f"layer {self.layer_id} (ratio {ratio}) reads compressed KV of ratio {state.compress_ratio}"
            if state.global_kv is not None:
                n_c = state.global_kv.size(0)
                if self.is_index_source:
                    state.global_indices = self._select_topk(x, qr, state, sq, n_c)
                compressed_idxs = torch.where(state.global_indices >= 0, state.global_indices + sq, -1)
                kv_full = torch.cat([kv, state.global_kv], dim=0)

        groups = (window_idxs,) if compressed_idxs is None else (window_idxs, compressed_idxs)
        if self.use_fused_kernels:
            flat_idxs, topk_length = build_flat_topk_idxs(*groups, batch_size=b, compact=True)
            return csa_sparse_attn(
                query, kv_full, self.attn_sink.float(), flat_idxs, self.softmax_scale, topk_length=topk_length
            )
        topk_idxs = torch.cat(groups, dim=-1).int()
        return unfused_compressed_sparse_attn(query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale)


@contextmanager
def _without_q_head_norm():
    original = dsv4_attn._q_rms_norm
    dsv4_attn._q_rms_norm = lambda q, eps: q
    try:
        yield
    finally:
        dsv4_attn._q_rms_norm = original


class DSv41HybridSelfAttention(dsv4_attn.DSv4HybridSelfAttention):
    """V4 attention minus the per-head query RMS norm, with compressed RoPE on every compressed layer."""

    def __init__(self, config, submodules, layer_number, *args, compress_ratio=None, **kwargs):
        ratio = config.csa2_compress_ratios[logical_layer_id(layer_number)]
        super().__init__(config, submodules, layer_number, *args, compress_ratio=ratio, **kwargs)
        if is_v41(config) and ratio == 1:
            # megatron-core's V4 attention keys YaRN + compress_rope_theta off ratio > 1; upstream V4.1 uses
            # ``use_compressed_yarn = compress_ratio > (0 if config.dsv4_version == "v4.1" else 1)``.
            self._dsv4_rope_base = self.config.csa_compress_rotary_base
            self._dsv4_uses_yarn_rope = True
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self._dsv4_rope_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        self.core_attention.rotary_pos_emb = self.rotary_pos_emb

    def get_query_key_value_tensors(self, *args, **kwargs):
        # Upstream: ``if config.dsv4_version != "v4.1": q = _q_rms_norm(q, ...)`` inside the up projection.
        if not is_v41(self.config):
            return super().get_query_key_value_tensors(*args, **kwargs)
        with _without_q_head_norm():
            qkv = super().get_query_key_value_tensors(*args, **kwargs)
        checkpoint = getattr(self, "qkv_up_checkpoint", None)
        if self.recompute_up_proj and checkpoint is not None:
            # mla_up_proj recompute re-runs the up projection in backward, outside the context above.
            up_proj = checkpoint.run_function

            def up_proj_without_q_head_norm(*inputs):
                with _without_q_head_norm():
                    return up_proj(*inputs)

            checkpoint.run_function = up_proj_without_q_head_norm
        return qkv
