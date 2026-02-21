"""Qwen3-VL vision-language model implementation.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import jax
from flax import nnx
from jax import numpy as jnp

from tx.layers.layernorm import RMSNorm
from tx.layers.util import Param
from tx.models.configs import Qwen3VLModelConfig
from tx.models.qwen3_vl_configs import Qwen3VLConfig
from tx.models.types import CausalLMOutput, ModelForCausalLM, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache
from tx.utils.logits_processor import LogitsProcessorMixin, LMHead

DType = jnp.dtype


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class VisionEmbeddings:
    """Container for vision tower outputs: tokens + optional deepstack features."""

    tokens: jax.Array
    deepstack: tuple[jax.Array, ...] = ()

    def cast(self, dtype: jnp.dtype) -> "VisionEmbeddings":
        return VisionEmbeddings(
            tokens=self.tokens.astype(dtype),
            deepstack=tuple(f.astype(dtype) for f in self.deepstack),
        )

    def with_batch_dim(self, batch: int) -> "VisionEmbeddings":
        """Ensure batch dimension matches expected size."""
        tokens = self.tokens if self.tokens.ndim == 3 else self.tokens[None, ...]
        if tokens.shape[0] == 1 and batch > 1:
            tokens = jnp.tile(tokens, (batch, 1, 1))
        deepstack = []
        for feat in self.deepstack:
            if feat.ndim == 2:
                feat = feat[None, ...]
            if feat.shape[0] == 1 and batch > 1:
                feat = jnp.tile(feat, (batch, 1, 1))
            deepstack.append(feat)
        return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))


@dataclass
class Qwen3VLSpec:
    """Spec for Qwen3-VL model built from config."""

    text_hidden_size: int
    text_num_heads: int
    text_num_layers: int
    text_num_kv_heads: int
    text_head_dim: int
    text_intermediate_size: int
    text_rope_theta: float
    text_rope_section: tuple[int, ...]
    text_mrope_interleaved: bool
    text_rms_norm_eps: float
    text_vocab_size: int
    vision_hidden_size: int
    vision_out_hidden_size: int
    vision_depth: int
    vision_num_heads: int
    vision_intermediate_size: int
    vision_patch_size: int
    vision_temporal_patch_size: int
    vision_spatial_merge_size: int
    vision_in_channels: int
    vision_num_position_embeddings: int | None
    vision_deepstack_indexes: tuple[int, ...]
    vision_fullatt_block_indexes: tuple[int, ...]
    vision_window_size: int
    image_token_id: int
    vision_start_token_id: int
    tie_word_embeddings: bool


# ============================================================================
# RoPE / mRoPE utilities
# ============================================================================


def _rotate_half(x: jax.Array) -> jax.Array:
    """Rotate half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_multimodal_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    rope_section: Sequence[int],
    unsqueeze_dim: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to q/k with optional interleaved mRoPE.

    Args:
        q, k: [B, Hq or Hkv, T, Dh]
        cos, sin: cos/sin tables shaped for mRoPE sections
        rope_section: tuple of section sizes
        unsqueeze_dim: where to broadcast cos/sin over heads

    Returns:
        (q_embed, k_embed) with rotation applied.
    """
    if cos.ndim == 3:
        cos_embed = jnp.expand_dims(cos, axis=unsqueeze_dim).astype(q.dtype)
        sin_embed = jnp.expand_dims(sin, axis=unsqueeze_dim).astype(q.dtype)
        q_embed = q * cos_embed + _rotate_half(q) * sin_embed
        k_embed = k * cos_embed + _rotate_half(k) * sin_embed
        return q_embed, k_embed

    sections = tuple(int(x) for x in rope_section)

    def _reorder(table: jax.Array) -> jax.Array:
        chunks = []
        for axis_idx, sec in enumerate(sections):
            axis_table = table[axis_idx, ...]
            offset = sum(sections[:axis_idx])
            chunk = axis_table[..., offset : offset + sec]
            chunks.append(chunk)
        reordered = jnp.concatenate(chunks, axis=-1)
        return jnp.concatenate([reordered, reordered], axis=-1)

    cos_flat = _reorder(cos).astype(q.dtype)
    sin_flat = _reorder(sin).astype(q.dtype)
    cos_embed = jnp.expand_dims(cos_flat, axis=unsqueeze_dim)
    sin_embed = jnp.expand_dims(sin_flat, axis=unsqueeze_dim)

    rope_dim = sum(sections) * 2
    if rope_dim > q.shape[-1]:
        rotated_dim = sum(sections)
        q_rot, q_pass = q[..., :rotated_dim], q[..., rotated_dim:]
        k_rot, k_pass = k[..., :rotated_dim], k[..., rotated_dim:]
        cos_rot = cos_embed[..., :rotated_dim]
        sin_rot = sin_embed[..., :rotated_dim]
        q_embed = jnp.concatenate(
            [q_rot * cos_rot + _rotate_half(q_rot) * sin_rot, q_pass], axis=-1
        )
        k_embed = jnp.concatenate(
            [k_rot * cos_rot + _rotate_half(k_rot) * sin_rot, k_pass], axis=-1
        )
    else:
        q_embed = q * cos_embed + _rotate_half(q) * sin_embed
        k_embed = k * cos_embed + _rotate_half(k) * sin_embed
    return q_embed, k_embed


def _apply_interleaved_mrope(freqs: jax.Array, rope_section: Sequence[int]) -> jax.Array:
    """Interleave (t,h,w) rotary freqs into a single axis layout."""
    sections = tuple(rope_section)
    if freqs.shape[0] < 3 or len(sections) < 3:
        return freqs[0]
    freqs_t = freqs[0]
    for axis_idx, offset in enumerate((1, 2), start=1):
        length = int(sections[axis_idx]) * 3
        if length <= offset:
            continue
        idx = jnp.arange(offset, length, 3)
        mask = jnp.zeros((freqs_t.shape[-1],), dtype=jnp.bool_).at[idx].set(True)
        freqs_t = jnp.where(mask[None, None, :], freqs[axis_idx], freqs_t)
    return freqs_t


def build_mrope(
    position_ids_axes: jax.Array,
    rope_section: Sequence[int],
    rope_theta: float,
    dtype: DType = jnp.bfloat16,
    rope_scaling_type: Optional[str] = None,
    rope_scaling_factor: Optional[float] = None,
    mrope_interleaved: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Build 3D mRoPE tables for (t, h, w) axes.

    Args:
        position_ids_axes: [3, B, T] integer positions per axis
        rope_section: sizes for each axis subspace
        rope_theta: RoPE base
        dtype: output dtype

    Returns:
        (cos, sin) each shaped [3, B, T, 2*sum(rope_section)] or [B, T, 2*sum] for 1D
    """
    sections = tuple(int(x) for x in rope_section)
    pos = position_ids_axes.astype(jnp.float32)
    if rope_scaling_factor and rope_scaling_type in (None, "linear", "dynamic", "finetuned"):
        pos = pos / jnp.float32(rope_scaling_factor)

    total_dim = sum(sections)
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(total_dim, dtype=jnp.float32) / total_dim))
    freqs = jnp.einsum(
        "sbn,k->sbnk", pos, inv_freq, precision=jax.lax.Precision.HIGHEST
    )
    if mrope_interleaved:
        freqs = _apply_interleaved_mrope(freqs, sections)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)

    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def build_text_rope(
    positions: jax.Array,
    rope_section: Sequence[int],
    rope_theta: float,
    dtype: DType = jnp.bfloat16,
    rope_scaling_type: Optional[str] = None,
    rope_scaling_factor: Optional[float] = None,
    mrope_interleaved: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Classic 1D RoPE for text tokens. Broadcasts to 3 axes to share codepath with mRoPE."""
    axes = len(tuple(rope_section))
    pos_axes = jnp.broadcast_to(positions[None, ...], (axes,) + positions.shape)
    return build_mrope(
        pos_axes,
        rope_section,
        rope_theta,
        dtype,
        rope_scaling_type,
        rope_scaling_factor,
        mrope_interleaved,
    )


def get_rope_index(
    spatial_merge_size: int = 2,
    input_ids: Optional[jax.Array] = None,
    image_grid_thw: Optional[jax.Array] = None,
    attention_mask: Optional[jax.Array] = None,
    image_token_id: Optional[int] = None,
    vision_start_id: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Compute per-token mRoPE indices for mixed text+vision sequences.

    Returns position_ids [3, B, T] and per-batch offsets `deltas` to align
    decode-time positions with prefill length. Text tokens get 1D positions
    broadcast to 3 axes; vision tokens use true (t,h,w) grid indices.
    """
    if input_ids is not None:
        batch, seq_len = input_ids.shape
    elif attention_mask is not None:
        batch, seq_len = attention_mask.shape[0], attention_mask.shape[1]
    else:
        batch, seq_len = 1, 1

    if input_ids is None or image_grid_thw is None:
        if attention_mask is not None:
            mask = attention_mask.astype(jnp.int32)
            positions = jnp.cumsum(mask, axis=-1) - 1
            positions = jnp.where(mask == 0, 0, positions)
            position_ids = jnp.tile(positions[None, ...], (3, 1, 1))
            deltas = (
                position_ids.max(axis=0).max(axis=-1, keepdims=True) + 1 - seq_len
            ).astype(jnp.int32)
        else:
            position_ids = jnp.tile(
                jnp.arange(seq_len, dtype=jnp.int32)[None, None, :], (3, batch, 1)
            )
            deltas = jnp.zeros((batch, 1), dtype=jnp.int32)
        return position_ids, deltas

    attention_mask = (
        attention_mask if attention_mask is not None else jnp.ones_like(input_ids)
    )
    grid_2d = image_grid_thw if image_grid_thw.ndim == 2 else image_grid_thw[:, 0, :]

    max_valid = seq_len

    def _single_seq(ids: jax.Array, mask: jax.Array, grid: jax.Array) -> jax.Array:
        n_valid = jnp.sum(mask).astype(jnp.int32)
        t, h, w = grid[0], grid[1], grid[2]
        grid_h = h // spatial_merge_size
        grid_w = w // spatial_merge_size
        num_vision = t * grid_h * grid_w
        num_text = n_valid - num_vision

        text_pos = jnp.tile(
            jnp.arange(num_text, dtype=jnp.int32)[None, :], (3, 1)
        )
        t_idx = jnp.tile(
            jnp.arange(t, dtype=jnp.int32)[:, None], (1, grid_h * grid_w)
        ).reshape(-1)
        h_idx = jnp.tile(
            jnp.arange(grid_h, dtype=jnp.int32)[None, :, None], (t, 1, grid_w)
        ).reshape(-1)
        w_idx = jnp.tile(
            jnp.arange(grid_w, dtype=jnp.int32)[None, None, :], (t, grid_h, 1)
        ).reshape(-1)
        spatial = jnp.stack([t_idx, h_idx, w_idx], axis=0) + num_text
        positions = jnp.concatenate([text_pos, spatial], axis=1)
        pad_len = max_valid - positions.shape[1]
        positions = jnp.pad(
            positions, ((0, 0), (0, pad_len)), constant_values=0
        )
        return positions

    positions_batched = jax.vmap(_single_seq, in_axes=(0, 0, 0))(
        input_ids, attention_mask, grid_2d
    )
    position_ids = jnp.transpose(positions_batched, (1, 0, 2))
    masked_positions = positions_batched * attention_mask[:, None, :].astype(
        positions_batched.dtype
    )
    max_per_batch = jnp.max(masked_positions, axis=(1, 2))
    deltas = (max_per_batch + 1 - seq_len).reshape(batch, 1).astype(jnp.int32)
    return position_ids, deltas


# ============================================================================
# Vision encoder
# ============================================================================


class VisionPatchEmbed(nnx.Module):
    """Patch embedding for vision (linear projection of flattened patches)."""

    def __init__(self, embed_dim: int, patch_volume: int, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.embed_dim = embed_dim
        self.patch_volume = patch_volume
        self.dtype = dtype
        self.proj = nnx.Linear(
            patch_volume,
            embed_dim,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj(x.astype(self.dtype))


class VisionAttention(nnx.Module):
    """Window-based self-attention for vision tokens."""

    def __init__(self, hidden_size: int, num_heads: int, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.dtype = dtype
        self.qkv = nnx.Linear(
            hidden_size,
            3 * hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        seq_len = x.shape[0]
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)

        cos = cos[:, : self.head_dim].astype(self.dtype)[:, None, :]
        sin = sin[:, : self.head_dim].astype(self.dtype)[:, None, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        num_windows = cu_seqlens.shape[0] - 1
        chunks = []
        for i in range(num_windows):
            start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            if start >= end:
                continue
            q_w, k_w, v_w = q[start:end], k[start:end], v[start:end]
            q_w = jnp.transpose(q_w, (1, 0, 2))
            k_w = jnp.transpose(k_w, (1, 0, 2))
            v_w = jnp.transpose(v_w, (1, 0, 2))
            scores = (
                jnp.einsum("hqd,hkd->hqk", q_w.astype(jnp.float32), k_w.astype(jnp.float32))
                * self.scale
            )
            weights = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("hqk,hkd->hqd", weights, v_w.astype(jnp.float32)).astype(self.dtype)
            chunks.append(jnp.transpose(out, (1, 0, 2)))

        out = jnp.concatenate(chunks, axis=0).reshape(seq_len, self.hidden_size)
        return self.proj(out)


class VisionMLP(nnx.Module):
    """MLP for vision blocks."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.fc1 = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=True)
        return self.fc2(x)


class VisionLayerNorm(nnx.Module):
    """LayerNorm for vision (with bias)."""

    def __init__(self, hidden_size: int, eps: float, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Param(
            hidden_size,
            dtype=dtype,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.bias = Param(
            hidden_size,
            dtype=dtype,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        normed = (x_f32 - mean) * jax.lax.rsqrt(var + self.eps)
        return (normed * self.weight + self.bias).astype(x.dtype)


class VisionBlock(nnx.Module):
    """Single vision transformer block."""

    def __init__(self, spec: Qwen3VLSpec, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.norm1 = VisionLayerNorm(spec.vision_hidden_size, 1e-6, dtype=dtype, rngs=rngs)
        self.norm2 = VisionLayerNorm(spec.vision_hidden_size, 1e-6, dtype=dtype, rngs=rngs)
        self.attn = VisionAttention(
            spec.vision_hidden_size,
            spec.vision_num_heads,
            dtype=dtype,
            rngs=rngs,
        )
        self.mlp = VisionMLP(
            spec.vision_hidden_size,
            spec.vision_intermediate_size,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nnx.Module):
    """Merge patches with optional spatial shuffle."""

    def __init__(
        self,
        context_dim: int,
        out_dim: int,
        spatial_merge_size: int,
        *,
        use_postshuffle_norm: bool = False,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.unit = spatial_merge_size**2
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = context_dim * self.unit if use_postshuffle_norm else context_dim
        self.norm = VisionLayerNorm(norm_dim, 1e-6, dtype=dtype, rngs=rngs)
        self.fc1 = nnx.Linear(
            context_dim * self.unit,
            context_dim * self.unit,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            context_dim * self.unit,
            out_dim,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_postshuffle_norm:
            x = x.reshape(-1, self.unit * self.context_dim)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.reshape(-1, self.unit * self.context_dim)
        x = jax.nn.gelu(self.fc1(x))
        return self.fc2(x)


class Qwen3VisionTransformer(nnx.Module):
    """Vision encoder (ViT) for Qwen3-VL."""

    def __init__(self, spec: Qwen3VLSpec, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.spec = spec
        patch_vol = (
            spec.vision_in_channels
            * spec.vision_temporal_patch_size
            * spec.vision_patch_size**2
        )
        self.patch_embed = VisionPatchEmbed(
            spec.vision_hidden_size,
            patch_vol,
            dtype=dtype,
            rngs=rngs,
        )
        self.pos_embed = None
        if spec.vision_num_position_embeddings:
            self.pos_embed = nnx.Embed(
                spec.vision_num_position_embeddings,
                spec.vision_hidden_size,
                dtype=dtype,
                embedding_init=nnx.initializers.normal(stddev=0.02),
                rngs=rngs,
            )
        self.blocks = [
            VisionBlock(spec, dtype=dtype, rngs=rngs)
            for _ in range(spec.vision_depth)
        ]
        self.merger = VisionPatchMerger(
            spec.vision_hidden_size,
            spec.vision_out_hidden_size,
            spec.vision_spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
        )
        self.deepstack_mergers = [
            VisionPatchMerger(
                spec.vision_hidden_size,
                spec.vision_out_hidden_size,
                spec.vision_spatial_merge_size,
                use_postshuffle_norm=True,
                dtype=dtype,
                rngs=rngs,
            )
            for _ in spec.vision_deepstack_indexes
        ]

    def _rot_pos_emb(self, grid_thw: jax.Array) -> jax.Array:
        """Compute rotary position embeddings for vision tokens."""
        rotary_dim = (self.spec.vision_hidden_size // self.spec.vision_num_heads) // 2
        theta = 10000.0
        inv_freq = 1.0 / (
            theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
        )
        pos_chunks = []
        for idx in range(grid_thw.shape[0]):
            t, h, w = grid_thw[idx]
            merge = self.spec.vision_spatial_merge_size
            hpos = jnp.arange(h)[:, None].repeat(w, axis=1)
            wpos = jnp.arange(w)[None, :].repeat(h, axis=0)
            hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(
                (0, 2, 1, 3)
            ).reshape(-1)
            wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(
                (0, 2, 1, 3)
            ).reshape(-1)
            pos = jnp.stack([hpos, wpos], axis=-1)
            pos = jnp.tile(pos, (int(t), 1))
            pos_chunks.append(pos)
        pos_ids = jnp.concatenate(pos_chunks, axis=0)
        max_grid = int(jnp.max(grid_thw[:, 1:]))
        seq_len = pos_ids.shape[0]
        freqs = jnp.outer(
            jnp.arange(max_grid * max_grid, dtype=jnp.float32)[:seq_len],
            inv_freq,
        )
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return emb

    def _get_cu_seqlens(self, grid_thw: jax.Array) -> jax.Array:
        """Cumulative sequence lengths per image."""
        merge = self.spec.vision_spatial_merge_size
        frame_sizes = jnp.repeat(
            grid_thw[:, 1] * grid_thw[:, 2] * (merge**2), grid_thw[:, 0]
        )
        return jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(frame_sizes, dtype=jnp.int32)]
        )

    def __call__(
        self,
        pixel_values: jax.Array,
        grid_thw: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """Forward pass. Returns (merged_tokens, deepstack_features)."""
        x = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            pos_ids = jnp.arange(x.shape[0], dtype=jnp.int32)
            pos_emb = self.pos_embed(pos_ids)
            x = x + pos_emb.astype(x.dtype)
        rotary_emb = self._rot_pos_emb(grid_thw)
        cos = jnp.cos(rotary_emb).astype(x.dtype)
        sin = jnp.sin(rotary_emb).astype(x.dtype)
        cu_seqlens = self._get_cu_seqlens(grid_thw)

        deepstack_feats = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cu_seqlens)
            if i in self.spec.vision_deepstack_indexes:
                idx = self.spec.vision_deepstack_indexes.index(i)
                feat = self.deepstack_mergers[idx](x)
                deepstack_feats.append(feat)

        x = self.merger(x)
        return x, tuple(deepstack_feats)


# ============================================================================
# Text decoder (VL-specific with mRoPE support)
# ============================================================================


class Qwen3VLAttention(nnx.Module):
    """Multi-head attention with RoPE for VL."""

    def __init__(self, spec: Qwen3VLSpec, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.spec = spec
        self.q_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_num_heads * spec.text_head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_num_kv_heads * spec.text_head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_num_kv_heads * spec.text_head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            spec.text_num_heads * spec.text_head_dim,
            spec.text_hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.q_norm = RMSNorm(
            spec.text_head_dim,
            eps=spec.text_rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )
        self.k_norm = RMSNorm(
            spec.text_head_dim,
            eps=spec.text_rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attention_mask: jax.Array,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        positions: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.spec.text_num_heads, self.spec.text_head_dim)
        k = self.k_proj(x).reshape(B, T, self.spec.text_num_kv_heads, self.spec.text_head_dim)
        v = self.v_proj(x).reshape(B, T, self.spec.text_num_kv_heads, self.spec.text_head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [B, H, T, Dh] for apply_multimodal_rotary_pos_emb
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        q, k = apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, self.spec.text_rope_section
        )
        # Keep [B, H, T, D] for einsum (no transpose back)

        # Handle KV cache (decode step)
        if kv_cache is not None and positions is not None:
            k, v = KVCache.update_layer(kv_cache, k, v, positions)
            k = jnp.transpose(k, (0, 2, 1, 3))  # [B, seq, Hkv, D] -> [B, Hkv, seq, D]
            v = jnp.transpose(v, (0, 2, 1, 3))
        else:
            v = jnp.transpose(v, (0, 2, 1, 3))  # [B, T, Hkv, D] -> [B, Hkv, T, D]

        scale = self.spec.text_head_dim**-0.5
        attn_mask = attention_mask[:, None, None, :].astype(jnp.float32)
        attn_mask = (1.0 - attn_mask) * -1e9

        kv_len = k.shape[2]
        if self.spec.text_num_heads != self.spec.text_num_kv_heads:
            repeats = self.spec.text_num_heads // self.spec.text_num_kv_heads
            q_grouped = q.reshape(
                B, self.spec.text_num_kv_heads, repeats, T, self.spec.text_head_dim
            )
            scores = (
                jnp.einsum(
                    "bhgqd,bhkd->bhgqk",
                    q_grouped.astype(jnp.float32),
                    k.astype(jnp.float32),
                )
                * scale
            )
            scores = scores.reshape(B, self.spec.text_num_heads, T, kv_len)
        else:
            scores = (
                jnp.einsum(
                    "bhqd,bhkd->bhqk",
                    q.astype(jnp.float32),
                    k.astype(jnp.float32),
                )
                * scale
            )

        scores = scores + attn_mask
        if T > 1 or kv_cache is None:
            causal_mask = jnp.tril(jnp.ones((T, kv_len), dtype=jnp.float32))
            scores = scores + (1.0 - causal_mask)[None, None, :, :] * -1e9
        weights = jax.nn.softmax(scores, axis=-1)

        if self.spec.text_num_heads != self.spec.text_num_kv_heads:
            weights_grouped = weights.reshape(
                B, self.spec.text_num_kv_heads, repeats, T, kv_len
            )
            out = jnp.einsum(
                "bhgqk,bhkd->bhgqd",
                weights_grouped,
                v.astype(jnp.float32),
            )
            out = out.reshape(B, self.spec.text_num_heads, T, self.spec.text_head_dim)
        else:
            out = jnp.einsum(
                "bhqk,bhkd->bhqd",
                weights,
                v.astype(jnp.float32),
            )
        out = jnp.transpose(out, (0, 2, 1, 3)).astype(x.dtype).reshape(B, T, -1)
        return self.o_proj(out), (k, v)


class Qwen3VLMLP(nnx.Module):
    """MLP for VL decoder."""

    def __init__(self, spec: Qwen3VLSpec, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.gate_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            spec.text_intermediate_size,
            spec.text_hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLDecoderLayer(nnx.Module):
    """Single decoder layer for VL."""

    def __init__(self, spec: Qwen3VLSpec, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_norm = RMSNorm(
            spec.text_hidden_size,
            eps=spec.text_rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )
        self.post_norm = RMSNorm(
            spec.text_hidden_size,
            eps=spec.text_rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )
        self.attn = Qwen3VLAttention(spec, dtype=dtype, rngs=rngs)
        self.mlp = Qwen3VLMLP(spec, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attention_mask: jax.Array,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        positions: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        attn_out, cache = self.attn(
            self.input_norm(x), cos, sin, attention_mask,
            kv_cache=kv_cache, positions=positions,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache


# ============================================================================
# Main model
# ============================================================================


def spec_from_config(config: Qwen3VLConfig | Qwen3VLModelConfig) -> Qwen3VLSpec:
    """Build Qwen3VLSpec from config."""
    text_cfg = config.text_config
    vision_cfg = config.vision_config
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads

    rope_params = getattr(text_cfg, "rope_parameters", None)
    if isinstance(rope_params, dict):
        rope_section = rope_params.get("mrope_section", [head_dim // 2])
        mrope_interleaved = bool(rope_params.get("mrope_interleaved", False))
    else:
        rope_section = [head_dim // 2]
        mrope_interleaved = False
    rope_section = tuple(int(x) for x in rope_section)
    rope_theta = getattr(text_cfg, "rope_theta", 500000.0)

    vision_fullatt = list(range(vision_cfg.depth)) if vision_cfg else []
    vision_deepstack = tuple(getattr(vision_cfg, "deepstack_visual_indexes", [8, 16, 24]) or [8, 16, 24])
    patch_sz = vision_cfg.patch_size if vision_cfg else 16
    window_sz = patch_sz * getattr(vision_cfg, "spatial_merge_size", 2)

    return Qwen3VLSpec(
        text_hidden_size=text_cfg.hidden_size,
        text_num_heads=text_cfg.num_attention_heads,
        text_num_layers=text_cfg.num_hidden_layers,
        text_num_kv_heads=text_cfg.num_key_value_heads,
        text_head_dim=head_dim,
        text_intermediate_size=text_cfg.intermediate_size,
        text_rope_theta=rope_theta,
        text_rope_section=rope_section,
        text_mrope_interleaved=mrope_interleaved,
        text_rms_norm_eps=text_cfg.rms_norm_eps,
        text_vocab_size=text_cfg.vocab_size,
        vision_hidden_size=vision_cfg.hidden_size if vision_cfg else 0,
        vision_out_hidden_size=vision_cfg.out_hidden_size if vision_cfg else 0,
        vision_depth=vision_cfg.depth if vision_cfg else 0,
        vision_num_heads=vision_cfg.num_heads if vision_cfg else 0,
        vision_intermediate_size=vision_cfg.intermediate_size if vision_cfg else 0,
        vision_patch_size=patch_sz,
        vision_temporal_patch_size=getattr(vision_cfg, "temporal_patch_size", 2) if vision_cfg else 2,
        vision_spatial_merge_size=getattr(vision_cfg, "spatial_merge_size", 2) if vision_cfg else 2,
        vision_in_channels=getattr(vision_cfg, "in_channels", 3) if vision_cfg else 3,
        vision_num_position_embeddings=getattr(vision_cfg, "num_position_embeddings", None)
        if vision_cfg
        else None,
        vision_deepstack_indexes=vision_deepstack,
        vision_fullatt_block_indexes=tuple(vision_fullatt),
        vision_window_size=window_sz,
        image_token_id=config.image_token_id,
        vision_start_token_id=config.vision_start_token_id,
        tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
    )


class Qwen3VLModel(nnx.Module):
    """Qwen3-VL model (vision + text backbone)."""

    def __init__(self, config: Qwen3VLModelConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.spec = spec_from_config(config)

        self.embed_tokens = nnx.Embed(
            self.spec.text_vocab_size,
            self.spec.text_hidden_size,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            rngs=rngs,
        )
        self.layers = [
            Qwen3VLDecoderLayer(self.spec, dtype=dtype, rngs=rngs)
            for _ in range(self.spec.text_num_layers)
        ]
        self.norm = RMSNorm(
            self.spec.text_hidden_size,
            eps=self.spec.text_rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )
        self.visual = (
            Qwen3VisionTransformer(self.spec, dtype=dtype, rngs=rngs)
            if self.spec.vision_depth > 0
            else None
        )

    def _apply_deepstack(
        self,
        hidden: jax.Array,
        visual_mask: jax.Array | None,
        features: jax.Array,
    ) -> jax.Array:
        """Add deepstack vision features at vision token positions."""
        if visual_mask is None or features.size == 0:
            return hidden

        def _add(h: jax.Array, mask: jax.Array, feat: jax.Array) -> jax.Array:
            idx = jnp.where(mask.ravel(), size=feat.shape[0], fill_value=-1)[0]
            valid = idx >= 0
            idx = jnp.where(valid, idx, 0)
            updates = jnp.where(
                valid[:, None],
                feat.astype(h.dtype),
                jnp.zeros_like(feat, dtype=h.dtype),
            )
            return h.at[idx.ravel()].add(updates.reshape(-1, h.shape[-1]))

        return jax.vmap(_add)(hidden, visual_mask.astype(bool), features)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        positions: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        hidden = self.embed_tokens(input_ids)
        batch = hidden.shape[0]
        is_decode = kv_cache is not None

        visual_mask = None
        deepstack = ()
        if (
            not is_decode
            and pixel_values is not None
            and self.visual is not None
            and image_grid_thw is not None
        ):
            vision_tokens, deepstack = self.visual(pixel_values, image_grid_thw)
            vision_emb = vision_tokens
            if vision_emb.ndim == 2:
                vision_emb = vision_emb[None, ...]
            if vision_emb.shape[0] == 1 and batch > 1:
                vision_emb = jnp.tile(vision_emb, (batch, 1, 1))
            image_pad_id = self.spec.image_token_id
            visual_mask = input_ids == image_pad_id

            def inject_vision(hidden_b, tokens_b, vis_b):
                mask = tokens_b == image_pad_id
                all_indices = jnp.where(mask)[0]
                n = min(all_indices.shape[0], vis_b.shape[0])
                indices = all_indices[:n]
                return hidden_b.at[indices].set(vis_b[:n])

            hidden = jax.vmap(inject_vision)(hidden, input_ids, vision_emb)

        if is_decode and positions is not None:
            cos, sin = build_text_rope(
                positions,
                self.spec.text_rope_section,
                self.spec.text_rope_theta,
                dtype=hidden.dtype,
                rope_scaling_type=None,
                rope_scaling_factor=None,
                mrope_interleaved=self.spec.text_mrope_interleaved,
            )
        else:
            position_ids, _ = get_rope_index(
                spatial_merge_size=self.spec.vision_spatial_merge_size,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                image_token_id=self.spec.image_token_id,
                vision_start_id=self.spec.vision_start_token_id,
            )
            cos, sin = build_mrope(
                position_ids,
                self.spec.text_rope_section,
                self.spec.text_rope_theta,
                dtype=hidden.dtype,
                rope_scaling_type=None,
                rope_scaling_factor=None,
                mrope_interleaved=self.spec.text_mrope_interleaved,
            )

        all_hidden = [] if output_hidden_states else None
        layer_caches: list[tuple[jax.Array, jax.Array]] = []
        for i, layer in enumerate(self.layers):
            layer_kv_tuple = (
                (kv_cache.keys[i], kv_cache.values[i]) if kv_cache else None
            )
            hidden, cache = layer(
                hidden,
                cos,
                sin,
                attention_mask,
                kv_cache=layer_kv_tuple,
                positions=positions,
            )
            layer_caches.append(cache)
            if deepstack and i < len(deepstack) and visual_mask is not None:
                hidden = self._apply_deepstack(hidden, visual_mask, deepstack[i])
            if output_hidden_states:
                all_hidden.append(hidden)

        hidden = self.norm(hidden)
        if output_hidden_states:
            all_hidden.append(hidden)

        # Transpose caches from [B, Hkv, T, D] to [B, T, Hkv, D] for KVCache
        keys = [jnp.transpose(c[0], (0, 2, 1, 3)) for c in layer_caches]
        values = [jnp.transpose(c[1], (0, 2, 1, 3)) for c in layer_caches]
        pos_for_cache = (
            positions
            if positions is not None
            else jnp.broadcast_to(
                jnp.arange(attention_mask.shape[1], dtype=jnp.int32)[None, :],
                (batch, attention_mask.shape[1]),
            )
        )
        new_kv_cache = KVCache.update(
            kv_cache,
            keys=keys,
            values=values,
            positions=pos_for_cache,
            attention_mask=attention_mask,
        )

        return ModelOutput(
            last_hidden_state=hidden,
            kv_cache=new_kv_cache,
            hidden_states=all_hidden,
        )


class Qwen3VLForCausalLM(nnx.Module, ModelForCausalLM, GeneratorMixin, LogitsProcessorMixin):
    """Qwen3-VL for causal language modeling (vision + text generation)."""

    def __init__(self, config: Qwen3VLModelConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Qwen3VLModel(config, dtype=dtype, rngs=rngs)

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens.T
        else:
            self.lm_head = nnx.Linear(
                self.model.spec.text_hidden_size,
                self.model.spec.text_vocab_size,
                use_bias=False,
                dtype=dtype,
                kernel_init=nnx.initializers.lecun_normal(),
                rngs=rngs,
            )

    def get_lm_head(self) -> LMHead:
        """Return lm_head callable: (hidden_states, adapter_indices) -> logits."""
        if self.config.tie_word_embeddings:
            emb = self.model.embed_tokens.embedding
            return lambda h, a=None: h @ emb[...].T
        return lambda h, a=None: self.lm_head(h)

    def get_model_config(self):
        return self.config

    @staticmethod
    def is_lora_param(path: tuple, _value: Any) -> bool:
        return False

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        positions: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
    ) -> CausalLMOutput:
        if positions is None and kv_cache is None:
            positions = jnp.broadcast_to(
                jnp.arange(attention_mask.shape[1], dtype=jnp.int32)[None, :],
                (attention_mask.shape[0], attention_mask.shape[1]),
            )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            positions=positions,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states or False,
        )
        return CausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )
