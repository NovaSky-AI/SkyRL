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
import functools
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
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import build_module
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.candidate_kernels import (
    candidate_kernel_supported,
    candidate_scores,
    candidate_topk_ids,
)

# Queries scored per indexer chunk: [chunk, b, n_heads, n_compressed] fp32 stays ~0.3 GB at 10K tokens.
_INDEXER_CHUNK = 256

# Indexer scoring: "reference" (fp32 einsum over all heads, chunks of ``_INDEXER_CHUNK`` queries) or
# "fused" (cuDNN DSA dense indexer kernel, as upstream's ``--csa2-indexer-impl fused``; falls back to
# the reference when cudnn-frontend's DSA namespace is unavailable).
_INDEXER_IMPL = os.environ.get("SKYRL_DSV41_INDEXER_IMPL", "reference").lower()
# Per-chunk budget for the fused path's fp32 score matrix (scores, masked copy and top-k workspace,
# about three [rows, n_c] buffers), as upstream's ``INDEXER_SCORE_BUDGET_BYTES``.
INDEXER_SCORE_BUDGET_BYTES = 1 << 30
_FUSED_INDEXER_SCORE = None


def _rows_within_budget(n_keys: int, budget_bytes: int = INDEXER_SCORE_BUDGET_BYTES) -> int:
    """Query rows whose dense fp32 scores (three buffers) fit the budget; at least one."""
    return max(1, budget_bytes // (3 * 4 * max(1, n_keys)))


def _fused_indexer_score_fn():
    """cudnn-frontend's DSA dense indexer score kernel, or ``None`` when unavailable."""
    global _FUSED_INDEXER_SCORE
    if _FUSED_INDEXER_SCORE is None:
        try:
            from cudnn import DSA

            _FUSED_INDEXER_SCORE = DSA.dense_indexer_score_recompute_wrapper
        except Exception:  # noqa: BLE001 - missing cudnn-frontend DSA namespace
            _FUSED_INDEXER_SCORE = False
    return _FUSED_INDEXER_SCORE or None


def fused_indexer_scores_rows(q_rows, keys, head_weights, head_weight_scale, ratio, first_position):
    """``sum_h relu(q_h . k) * w_h`` for consecutive query positions through the cuDNN DSA dense indexer
    kernel (one THD segment, all keys). Backport of ``fused_indexer_scores_rows`` from
    NVIDIA/Megatron-LM#7285 (open; DeepSeek-V4.1-Flash training), which reaches the same kernel through
    a newer ``_compute_dense_indexer_score``; the pinned megatron-core's wrapper is BSHD-only, so this
    calls cudnn-frontend directly.

    Args:
        q_rows: ``[rows, h_i, d_i]`` rotated indexer queries (bf16).
        keys: ``[n, d_i]`` rotated index keys (bf16).
        head_weights: ``[rows, h_i]`` raw head weights (the scale goes to the kernel, which applies it
            in fp32 before the ReLU -- equivalent for a positive scale).
        head_weight_scale: ``(d_i * h_i) ** -0.5``.
        ratio: compress ratio (causal limit ``(position + 1) // ratio`` inside the kernel).
        first_position: sequence position of ``rows[0]``; rows are consecutive positions.

    Returns:
        ``[rows, n]`` fp32 scores (entries beyond the causal limit are not meaningful; the caller
        masks them), or ``None`` when the kernel is unavailable.
    """
    fn = _fused_indexer_score_fn()
    if fn is None or not q_rows.is_cuda:
        return None
    rows, n_keys = q_rows.size(0), keys.size(0)
    device = q_rows.device
    scores, _ = fn(
        q_rows.contiguous(),
        keys.unsqueeze(1).contiguous(),
        head_weights.to(q_rows.dtype).contiguous(),
        qhead_per_kv_head=q_rows.size(1),
        sm_scale=float(head_weight_scale),
        ratio=int(ratio),
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, n_keys], dtype=torch.int32, device=device),
        max_seqlen_q=rows,
        max_seqlen_k=n_keys,
        q_causal_offsets=torch.tensor([first_position], dtype=torch.int32, device=device),
    )
    return scores[:, :n_keys].float()


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
    # Candidate blocks of the candidate-source layer: [sq, b, <= candidate_topk_blocks] int32 block ids, -1 = none
    # (block ids, as NVIDIA/Megatron-LM#7285; a dense [sq, b, n_c] position mask is quadratic in the sequence).
    candidate_blocks: Optional[torch.Tensor] = None


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


_replay = threading.local()


@contextmanager
def reuse_selection(global_indices: Optional[torch.Tensor], candidate_blocks: Optional[torch.Tensor]):
    """While a layer is replayed for backward (``recompute.py``), hand its index-source attention the top-k (and
    candidate blocks) it computed in the forward, instead of scoring and selecting again: they are integers (no
    gradient), the forward kept them alive for the later layers anyway, and reusing them guarantees the replay
    attends to exactly the positions the forward did (a top-k with ties need not)."""
    _replay.selection = (global_indices, candidate_blocks)
    try:
        yield
    finally:
        _replay.selection = None


def _replayed_selection():
    return getattr(_replay, "selection", None)


def is_v41(config) -> bool:
    return getattr(config, "dsv4_version", None) == "v4.1"


def logical_layer_id(layer_number: int) -> int:
    return (layer_number - 1) // 2


@torch.no_grad()
def select_candidate_block_ids(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Level one of the two-level top-k as block ids: ``[..., min(topk_blocks, n_blocks)]`` int32, -1 = none.

    Same selection as :func:`select_candidate_blocks` (best blocks by max position score, the query's
    newest, partly filled block pinned in), returned compactly (``block_size`` x fewer entries than the
    position mask, which is ``[sq, n_c]`` -- quadratic in the sequence)."""
    width = logits.size(-1)
    scores = nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    return torch.where(top.values > -torch.inf, top.indices, -1).to(torch.int32)


def candidate_blocks_to_mask(block_ids: torch.Tensor, block_size: int, width: int) -> torch.Tensor:
    """Position mask ``[..., width]`` bool of the candidate blocks in ``block_ids`` (-1 ignored)."""
    num_blocks = -(-width // block_size)
    keep = torch.zeros(*block_ids.shape[:-1], num_blocks + 1, dtype=torch.bool, device=block_ids.device)
    keep.scatter_(-1, torch.where(block_ids >= 0, block_ids, num_blocks).long(), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


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


_NANCHECK = os.environ.get("SKYRL_DSV41_CP_NANCHECK", "")  # a directory: per-rank diagnostics file


def _nanlog(msg: str) -> None:
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    with open(os.path.join(_NANCHECK, f"nancheck_rank{rank}.txt"), "a") as f:
        f.write(msg + "\n")


def _nancheck(layer_id, cp_rank, kv_full_rows, **tensors):
    """Debug: per-tensor NaN counts and index ranges in a compressed layer under CP (printed once per layer/rank)."""
    key = (layer_id, cp_rank)
    if key in _NANCHECK_SEEN:
        return
    _NANCHECK_SEEN.add(key)
    parts = []
    for name, t in tensors.items():
        if t is None:
            continue
        if t.is_floating_point():
            parts.append(f"{name}{list(t.shape)} nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())}")
        else:
            valid = t[t >= 0]
            parts.append(f"{name}{list(t.shape)} valid={valid.numel()} min={int(valid.min()) if valid.numel() else -1} "
                         f"max={int(valid.max()) if valid.numel() else -1}")
    _nanlog(f"[cp_nancheck] layer {layer_id} cp_rank {cp_rank} kv_full_rows={kv_full_rows} " + " | ".join(parts))


_NANCHECK_SEEN: set = set()


# Debug (SKYRL_DSV41_SELDUMP=<dir>): logical layer 2's top-k selections and attention output for global positions
# [0, 256) and [L/2, L/2 + 256) (L = the whole sequence), first microbatch of data-parallel rank 0.
_SELDUMP = os.environ.get("SKYRL_DSV41_SELDUMP", "")
_SELDUMP_DONE: set = set()


def _dump_selection(attn, cp, global_start, sq, global_indices, out):
    import megatron.core.parallel_state as mpu

    if mpu.get_data_parallel_rank() != 0:
        return
    total = sq * cp.size()
    for g0 in (0, total // 2):
        if global_start <= g0 < global_start + sq and g0 not in _SELDUMP_DONE:
            _SELDUMP_DONE.add(g0)
            l0 = g0 - global_start
            torch.save(
                {"idx": global_indices[0, l0 : l0 + 256].cpu(), "out": out[l0 : l0 + 256].float().cpu()},
                os.path.join(_SELDUMP, f"sel_g{g0}.pt"),
            )


class _NoCPGroup:
    """Stand-in CP group of size 1 for RoPE on tensors that already hold every position (no CP slicing)."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


_NO_CP = _NoCPGroup()


@functools.lru_cache(maxsize=16)
def _window_idxs_cached(window_size: int, seqlen: int, global_start: int, boundary_rows: int, device_str: str):
    """Sliding-window indices ``[seqlen, window_size]`` for queries at global positions ``global_start + i``, into a
    KV buffer whose first ``boundary_rows`` rows are the positions just before ``global_start`` (-1 = none). With
    ``global_start == boundary_rows == 0`` this is megatron-core's ``get_window_topk_idxs`` (same order)."""
    pos = torch.arange(global_start, global_start + seqlen, device=device_str).unsqueeze(1)
    first = (pos - window_size + 1).clamp(min=0)
    idx = first + torch.arange(window_size, device=device_str)
    return torch.where(idx > pos, -1, idx - global_start + boundary_rows)


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

    def _rope_at(self, x, start, n, ratio):
        """RoPE for (compressed) positions ``start .. start + n - 1`` in units of ``ratio`` tokens: CP's local queries
        at their global positions, and the gathered compressed KV at every position.

        The rotary module is CP-aware: asked normally it returns cos/sin for this rank's zigzag share of positions.
        Ask it for the whole table (``packed_seq=True``, as megatron-core's packed path does) and pick the rows."""
        from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
        from megatron.core.models.common.embeddings.rope_utils import (
            apply_rotary_pos_emb,
        )

        pos_dim = self.config.qk_pos_emb_head_dim
        nope_dim = x.size(-1) - pos_dim
        total = (start + n) * ratio
        rows = torch.arange(start, start + n, device=x.device) * ratio
        squeeze_head = x.dim() == 3
        if squeeze_head:
            x = x.unsqueeze(-2)
        if self.config.apply_rope_fusion:
            cos, sin = self.rotary_pos_emb.get_cached_cos_sin(total, dtype=x.dtype, packed_seq=True, mscale=1.0)
            out = fused_mla_rope_inplace(
                x, cos.index_select(0, rows), sin.index_select(0, rows), nope_dim, pos_dim, None, 0, 1,
                remove_interleaving=True,
            )
        else:
            emb = self.rotary_pos_emb(total, packed_seq=True)
            emb = emb[0] if isinstance(emb, tuple) else emb
            x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
            x_pe = apply_rotary_pos_emb(
                x_pe, emb.index_select(0, rows), config=self.config, cu_seqlens=None, mscale=1.0, cp_group=_NO_CP,
                mla_rotary_interleaved=True, mla_output_remove_interleaving=True,
            )
            out = torch.cat([x_nope, x_pe], dim=-1)
        return out.squeeze(-2) if squeeze_head else out

    @torch.no_grad()
    def _select_topk(self, x, qr, state, sq, n_c, global_start=0):
        """Indexer top-k over the shared compressed positions: ``[b, sq, topk]`` int32, -1 = none.

        The candidate source publishes its candidate blocks as ids (``[sq, b, <= candidate_topk_blocks]``
        int32, -1 = none) rather than a dense ``[sq, b, n_c]`` position mask, which is quadratic in the
        sequence (64 GiB at 256k). Layers that use them score only the candidate positions with
        NVIDIA/Megatron-LM#7285's Triton candidate kernel when it supports the geometry, else densely with a
        per-chunk mask rebuilt from the ids.
        """
        indexer = self.indexer
        ratio = self.compress_ratio
        q, _ = indexer.linear_wq_b(qr.detach())
        q = q.view(sq, q.size(1), indexer.n_heads, indexer.head_dim)
        q = self._rope(q, sq, 1) if self.pg_collection.cp.size() == 1 else self._rope_at(q, global_start, sq, 1)
        raw_weights, _ = indexer.linear_weights_proj(x.detach())
        head_weight_scale = indexer.softmax_scale * indexer.n_heads**-0.5
        topk = min(indexer.index_topk, n_c)
        batch, block_size = q.size(1), self.candidate_block_size
        if (
            self.uses_candidates
            and q.is_cuda
            and candidate_kernel_supported(
                indexer.n_heads, indexer.head_dim, block_size, state.candidate_blocks.size(-1)
            )
        ):
            return self._select_topk_candidates(q, raw_weights, head_weight_scale, state, sq, topk, global_start)
        positions = torch.arange(n_c, device=x.device)
        candidate_blocks = [] if self.is_candidate_source else None
        out = []
        fused = _INDEXER_IMPL == "fused" and _fused_indexer_score_fn() is not None and q.is_cuda
        if fused:
            q_lp, k_lp = q.to(torch.bfloat16), state.indexer_k.to(torch.bfloat16)  # [sq, b, h, d], [n_c, b, d]
            chunk = _rows_within_budget(n_c * batch)
        else:
            q = q.float()
            weights = raw_weights.float() * head_weight_scale
            k = state.indexer_k.float()  # [n_c, b, d]
            chunk = _INDEXER_CHUNK
        for s0 in range(0, sq, chunk):
            s1 = min(s0 + chunk, sq)
            if fused:
                scores = torch.stack(
                    [
                        fused_indexer_scores_rows(
                            q_lp[s0:s1, i], k_lp[:, i], raw_weights[s0:s1, i], head_weight_scale, ratio,
                            global_start + s0,
                        )
                        for i in range(batch)
                    ],
                    dim=1,
                )  # [c, b, n_c]
            else:
                scores = torch.einsum("sbhd,tbd->sbht", q[s0:s1], k).relu_()
                scores = (scores * weights[s0:s1].unsqueeze(-1)).sum(dim=2)  # [c, b, n_c]
            compress_lens = (
                torch.arange(global_start + s0 + 1, global_start + s1 + 1, device=x.device) // ratio
            ).view(-1, 1, 1)
            scores.masked_fill_(positions >= compress_lens, -torch.inf)
            if self.is_candidate_source:
                candidate_blocks.append(
                    select_candidate_block_ids(scores, compress_lens, self.candidate_topk_blocks, block_size)
                )
            elif self.uses_candidates:
                keep = candidate_blocks_to_mask(state.candidate_blocks[s0:s1], block_size, n_c)
                scores.masked_fill_(~keep, -torch.inf)
            vals, idxs = scores.topk(topk, dim=-1)
            out.append(torch.where(torch.isfinite(vals) & (idxs < compress_lens), idxs, -1))
        if self.is_candidate_source:
            state.candidate_blocks = torch.cat(candidate_blocks, dim=0)
        return torch.cat(out, dim=0).permute(1, 0, 2).to(torch.int32).contiguous()

    def _select_topk_candidates(self, q, raw_weights, head_weight_scale, state, sq, topk, global_start=0):
        """Top-k inside the published candidate blocks (NVIDIA/Megatron-LM#7285 ``candidate_scores`` +
        ``candidate_topk_ids``): ``[b, sq, topk]`` int32, ids ascending, -1 = none."""
        ratio, block_size = self.compress_ratio, self.candidate_block_size
        blocks = state.candidate_blocks  # [sq, b, n_blocks] int32
        q_lp, k_lp = q.to(torch.bfloat16), state.indexer_k.to(torch.bfloat16)
        width = blocks.size(-1) * block_size
        out = []
        chunk = _rows_within_budget(width * q.size(1))
        for s0 in range(0, sq, chunk):
            s1 = min(s0 + chunk, sq)
            visible = torch.arange(global_start + s0 + 1, global_start + s1 + 1, device=q.device) // ratio
            per_batch = []
            for i in range(q.size(1)):
                scores = candidate_scores(
                    q_lp[s0:s1, i],
                    k_lp[:, i],
                    raw_weights[s0:s1, i],
                    head_weight_scale,
                    blocks[s0:s1, i],
                    block_size,
                    visible,
                )
                ids = candidate_topk_ids(scores, blocks[s0:s1, i], block_size, topk)
                per_batch.append(nn.functional.pad(ids, (0, topk - ids.size(-1)), value=-1))
            out.append(torch.stack(per_batch, dim=1))  # [c, b, topk]
        return torch.cat(out, dim=0).permute(1, 0, 2).contiguous()

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
        boundary_hidden=None,
        boundary_kv=None,
    ):
        assert attention_mask is None, "DeepSeek-V4.1 attention uses an implicit causal mask"
        cp = self.pg_collection.cp
        cp_size = cp.size()
        packed = packed_seq_params is not None
        if packed:
            # Packed THD (the only input megatron-core's DSv4 attention accepts under CP): q [t, h, d], k [t, 1, d],
            # hidden [t, 1, H], qr [t, r]. One sequence per microbatch (its padded length is the pack).
            cu = packed_seq_params.cu_seqlens_q_padded
            cu = packed_seq_params.cu_seqlens_q if cu is None else cu
            assert cu.numel() == 2, "DeepSeek-V4.1 packed attention supports one sequence per microbatch"
            query = query.unsqueeze(1) if query.dim() == 3 else query
            key = key.unsqueeze(1) if key.dim() == 3 else key
            qr = qr.unsqueeze(1) if qr is not None and qr.dim() == 2 else qr
        elif cp_size > 1:
            raise ValueError("DeepSeek-V4.1 context parallelism requires packed THD input")
        sq, b, _, _ = query.size()
        # Contiguous CP layout: this rank owns global positions [global_start, global_start + sq).
        global_start = cp.rank() * sq if cp_size > 1 else 0
        kv = key.squeeze(-2)
        if FAKE_QUANT_KV == "vllm":
            kv = fake_quant_fp8_ds_mla(kv, kv.size(-1) - self.config.qk_pos_emb_head_dim)
        elif FAKE_QUANT_KV == "reference":
            kv = fake_quant_fp8_block(kv)
        boundary_rows = 0
        if cp_size > 1:
            # The left neighbour's last rows (projected + RoPE'd by DSv4HybridSelfAttention) extend the window.
            assert boundary_kv is not None, "DeepSeek-V4.1 CP needs the left-boundary KV"
            bkv = boundary_kv.reshape(boundary_kv.size(0), b, kv.size(-1))
            if FAKE_QUANT_KV == "vllm":
                bkv = fake_quant_fp8_ds_mla(bkv, bkv.size(-1) - self.config.qk_pos_emb_head_dim)
            elif FAKE_QUANT_KV == "reference":
                bkv = fake_quant_fp8_block(bkv)
            boundary_rows = bkv.size(0)
            assert boundary_rows >= self.window_size, "CP boundary narrower than the sliding window"
            kv = torch.cat([bkv, kv], dim=0)
            window_idxs = (
                _window_idxs_cached(self.window_size, sq, global_start, boundary_rows, str(query.device))
                .unsqueeze(0)
                .expand(b, -1, -1)
            )
        else:
            window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)
        compressed_idxs = None
        kv_full = kv
        if self.compress_ratio:
            # Upstream's stack passes csa2_state; the pinned DSv4HybridSelfAttention does not, so use this forward's.
            state = csa2_state if csa2_state is not None else forward_state().csa2
            ratio = self.compress_ratio
            if self.is_kv_source:
                latent = self.compressor(x)
                if cp_size > 1:
                    # Each rank compresses its own rows (its chunk starts on a group boundary); every rank attends
                    # to all compressed positions, so gather them (backward: reduce-scatter to the owners).
                    assert latent is not None and sq % ratio == 0, "CP chunk must be a whole number of groups"
                    latent = gather_from_sequence_parallel_region(latent, tensor_parallel_output_grad=True, group=cp)
                state.compress_ratio = ratio
                if latent is None:
                    state.global_kv = state.indexer_k = None
                else:
                    n_c = latent.size(0)
                    rope = (lambda t: self._rope(t, n_c, ratio)) if cp_size == 1 else (
                        lambda t: self._rope_at(t, 0, n_c, ratio)
                    )
                    if self.indexer is not None:
                        with torch.no_grad():
                            k, _ = self.indexer.linear_wk(latent.detach())
                            state.indexer_k = rope(self.indexer.k_norm(k))
                    state.global_kv = rope(latent)
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
                    replayed = _replayed_selection()
                    if replayed is not None and replayed[0] is not None:
                        state.global_indices = replayed[0]
                        if self.is_candidate_source:
                            state.candidate_blocks = replayed[1]
                    else:
                        state.global_indices = self._select_topk(x, qr, state, sq, n_c, global_start)
                offset = kv.size(0)  # compressed rows follow the (boundary +) local rows
                compressed_idxs = torch.where(state.global_indices >= 0, state.global_indices + offset, -1)
                kv_full = torch.cat([kv, state.global_kv], dim=0)

        if _NANCHECK and self.compress_ratio and cp_size > 1:
            _nancheck(self.layer_id, cp.rank(), query=query, kv_ext=kv, global_kv=state.global_kv,
                      indexer_k=state.indexer_k, global_indices=state.global_indices, compressed_idxs=compressed_idxs,
                      window_idxs=window_idxs, kv_full_rows=kv_full.size(0))
        groups = (window_idxs,) if compressed_idxs is None else (window_idxs, compressed_idxs)
        if self.use_fused_kernels:
            flat_idxs, topk_length = build_flat_topk_idxs(*groups, batch_size=b, compact=True)
            out = csa_sparse_attn(
                query, kv_full, self.attn_sink.float(), flat_idxs, self.softmax_scale, topk_length=topk_length
            )
            if _SELDUMP and self.layer_id == 2 and self.compress_ratio:
                _dump_selection(self, cp, global_start, sq, state.global_indices, out)
            if _NANCHECK and self.compress_ratio and cp_size > 1 and (self.layer_id, cp.rank(), "out") not in _NANCHECK_SEEN:
                _NANCHECK_SEEN.add((self.layer_id, cp.rank(), "out"))
                rows = torch.isnan(out.float().flatten(1)).any(dim=1)
                _nanlog(f"[cp_nancheck] layer {self.layer_id} cp_rank {cp.rank()} attn_out nan_rows={int(rows.sum())} "
                        f"flat_idxs {list(flat_idxs.shape)} max={int(flat_idxs.max())} topk_length "
                        f"min={int(topk_length.min())} max={int(topk_length.max())}")
            return out
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
