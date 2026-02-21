"""Qwen3-VL vision-language model implementation.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from tx.layers.attention import dot_product_attention
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
    text_attention_bias: bool
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
    video_token_id: int
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
    if cos.ndim == 4:
        # Legacy path: [3, B, T, D]. Collapse into interleaved [B, T, D].
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
    else:
        cos_flat = cos.astype(q.dtype)
        sin_flat = sin.astype(q.dtype)

    cos_embed = jnp.expand_dims(cos_flat, axis=unsqueeze_dim)
    sin_embed = jnp.expand_dims(sin_flat, axis=unsqueeze_dim)

    rotated_dim = min(int(cos_embed.shape[-1]), int(q.shape[-1]))
    if rotated_dim != q.shape[-1]:
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


def _apply_interleaved_mrope(
    freqs: jax.Array, rope_section: Sequence[int]
) -> jax.Array:
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
    if rope_scaling_factor and rope_scaling_type in (
        None,
        "linear",
        "dynamic",
        "finetuned",
    ):
        pos = pos / jnp.float32(rope_scaling_factor)

    total_dim = sum(sections)
    inv_freq = 1.0 / (
        rope_theta ** (jnp.arange(total_dim, dtype=jnp.float32) / total_dim)
    )
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


def _get_rope_index_batch_py(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grid_thw: np.ndarray,
    video_grid_thw: np.ndarray,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """HF-aligned multimodal RoPE index computation over the full batch."""
    batch, seq_len = input_ids.shape

    image_grid = np.asarray(image_grid_thw, dtype=np.int32)
    if image_grid.size == 0:
        image_grid = np.zeros((0, 3), dtype=np.int32)
    elif image_grid.ndim == 1:
        image_grid = image_grid.reshape(1, 3)

    video_grid = np.asarray(video_grid_thw, dtype=np.int32)
    if video_grid.size == 0:
        video_grid = np.zeros((0, 3), dtype=np.int32)
    elif video_grid.ndim == 1:
        video_grid = video_grid.reshape(1, 3)

    if video_grid.shape[0] > 0:
        expanded = []
        for t, h, w in video_grid.tolist():
            expanded.extend([[1, h, w]] * int(t))
        video_grid = (
            np.asarray(expanded, dtype=np.int32)
            if expanded
            else np.zeros((0, 3), dtype=np.int32)
        )

    position_ids = np.zeros((3, batch, seq_len), dtype=np.int32)
    mrope_position_deltas = []

    image_index = 0
    video_index = 0
    for b in range(batch):
        valid_tokens = input_ids[b][attention_mask[b].astype(bool)]
        input_tokens = valid_tokens.tolist()

        if len(input_tokens) == 0:
            mrope_position_deltas.append(0)
            continue

        vision_start_indices = np.where(valid_tokens == vision_start_id)[0]
        if len(vision_start_indices) > 0:
            next_indices = np.clip(vision_start_indices + 1, 0, len(valid_tokens) - 1)
            vision_tokens = valid_tokens[next_indices]
            image_nums = int(np.sum(vision_tokens == image_token_id))
            video_nums = int(np.sum(vision_tokens == video_token_id))
        else:
            image_nums = 0
            video_nums = 0

        llm_pos_ids_list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums

        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1

            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = image_grid[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size

            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                np.arange(text_len, dtype=np.int32)[None, :].repeat(3, axis=0) + st_idx
            )

            t_index = (
                np.arange(llm_grid_t, dtype=np.int32)[:, None]
                .repeat(llm_grid_h * llm_grid_w, axis=1)
                .reshape(-1)
            )
            h_index = (
                np.arange(llm_grid_h, dtype=np.int32)[None, :, None]
                .repeat(llm_grid_t, axis=0)
                .repeat(llm_grid_w, axis=2)
                .reshape(-1)
            )
            w_index = (
                np.arange(llm_grid_w, dtype=np.int32)[None, None, :]
                .repeat(llm_grid_t, axis=0)
                .repeat(llm_grid_h, axis=1)
                .reshape(-1)
            )
            llm_pos_ids_list.append(
                np.stack([t_index, h_index, w_index], axis=0) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.arange(text_len, dtype=np.int32)[None, :].repeat(3, axis=0) + st_idx
            )

        llm_positions = (
            np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            if len(llm_pos_ids_list) > 0
            else np.zeros((3, 0), dtype=np.int32)
        )

        valid_sel = attention_mask[b].astype(bool)
        position_ids[:, b, valid_sel] = llm_positions
        delta = (
            int(llm_positions.max()) + 1 - int(seq_len) if llm_positions.size > 0 else 0
        )
        mrope_position_deltas.append(delta)

    deltas = np.asarray(mrope_position_deltas, dtype=np.int32)[:, None]
    return position_ids, deltas


def get_rope_index(
    spatial_merge_size: int = 2,
    input_ids: Optional[jax.Array] = None,
    image_grid_thw: Optional[jax.Array] = None,
    video_grid_thw: Optional[jax.Array] = None,
    attention_mask: Optional[jax.Array] = None,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    vision_start_id: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Compute per-token mRoPE indices (HF-aligned segment parsing).

    Scans for vision_start_token_id, identifies image/video per segment,
    builds positions for interleaved text/vision. Returns position_ids [3,B,T]
    and rope_deltas [B,1] for decode alignment.
    """
    if input_ids is not None:
        batch, seq_len = input_ids.shape
    elif attention_mask is not None:
        batch, seq_len = attention_mask.shape[0], attention_mask.shape[1]
    else:
        batch, seq_len = 1, 1

    IMG_ID = int(image_token_id) if image_token_id is not None else 151655
    VID_ID = int(video_token_id) if video_token_id is not None else 151656
    VSTART_ID = int(vision_start_id) if vision_start_id is not None else 151652

    if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
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
    ig = (
        jnp.asarray(image_grid_thw, dtype=jnp.int32)
        if image_grid_thw is not None
        else jnp.zeros((0, 3), dtype=jnp.int32)
    )
    vg = (
        jnp.asarray(video_grid_thw, dtype=jnp.int32)
        if video_grid_thw is not None
        else jnp.zeros((0, 3), dtype=jnp.int32)
    )

    def _rope_callback(ids, msk, ig_all, vg_all):
        return _get_rope_index_batch_py(
            np.asarray(ids),
            np.asarray(msk),
            np.asarray(ig_all),
            np.asarray(vg_all),
            spatial_merge_size,
            IMG_ID,
            VID_ID,
            VSTART_ID,
        )

    result_shape = (
        jax.ShapeDtypeStruct((3, batch, seq_len), jnp.int32),
        jax.ShapeDtypeStruct((batch, 1), jnp.int32),
    )
    position_ids, deltas = jax.pure_callback(
        _rope_callback,
        result_shape,
        input_ids,
        attention_mask,
        ig,
        vg,
    )
    return position_ids, deltas


# ============================================================================
# Vision encoder
# ============================================================================


class VisionPatchEmbed(nnx.Module):
    """Patch embedding for vision (linear projection of flattened patches)."""

    def __init__(
        self, embed_dim: int, patch_volume: int, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
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

    def __init__(
        self, hidden_size: int, num_heads: int, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
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
                jnp.einsum(
                    "hqd,hkd->hqk", q_w.astype(jnp.float32), k_w.astype(jnp.float32)
                )
                * self.scale
            )
            weights = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("hqk,hkd->hqd", weights, v_w.astype(jnp.float32)).astype(
                self.dtype
            )
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

    def __init__(
        self, hidden_size: int, eps: float, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
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
        self.norm1 = VisionLayerNorm(
            spec.vision_hidden_size, 1e-6, dtype=dtype, rngs=rngs
        )
        self.norm2 = VisionLayerNorm(
            spec.vision_hidden_size, 1e-6, dtype=dtype, rngs=rngs
        )
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
            VisionBlock(spec, dtype=dtype, rngs=rngs) for _ in range(spec.vision_depth)
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
        """Compute rotary position embeddings (PyTorch-aligned freq_table lookup)."""
        rotary_dim = (self.spec.vision_hidden_size // self.spec.vision_num_heads) // 2
        theta = 10000.0
        inv_freq = 1.0 / (
            theta ** (jnp.arange(rotary_dim, dtype=jnp.float32) / rotary_dim)
        )
        max_hw = int(jnp.max(grid_thw[:, 1:]))
        freq_table = jnp.outer(jnp.arange(max_hw, dtype=jnp.float32), inv_freq)

        merge = self.spec.vision_spatial_merge_size
        pos_chunks = []
        for idx in range(grid_thw.shape[0]):
            t, h, w = grid_thw[idx]
            merged_h, merged_w = h // merge, w // merge
            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge)
            intra_col = jnp.arange(merge)
            row_idx = (
                block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]
            )
            row_idx = jnp.broadcast_to(
                row_idx, (merged_h, merged_w, merge, merge)
            ).reshape(-1)
            col_idx = jnp.broadcast_to(
                col_idx, (merged_h, merged_w, merge, merge)
            ).reshape(-1)
            coords = jnp.stack([row_idx, col_idx], axis=-1)
            if t > 1:
                coords = jnp.tile(coords, (t, 1))
            pos_chunks.append(coords)

        pos_ids = jnp.concatenate(pos_chunks, axis=0)
        row_emb = freq_table[pos_ids[:, 0]]
        col_emb = freq_table[pos_ids[:, 1]]
        embeddings = jnp.concatenate([row_emb, col_emb], axis=-1)
        return embeddings

    def _get_cu_seqlens(self, grid_thw: jax.Array) -> jax.Array:
        """Cumulative sequence lengths per frame (PyTorch-aligned)."""
        frame_sizes = jnp.repeat(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0].astype(jnp.int32)
        )
        return jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(frame_sizes, dtype=jnp.int32)]
        )

    def _fast_pos_embed_interpolate(self, grid_thw: jax.Array) -> jax.Array:
        """Bilinear interpolation of position embeddings (PyTorch-aligned)."""
        if self.pos_embed is None:
            return jnp.zeros((0, self.spec.vision_hidden_size))
        num_pos = int(self.spec.vision_num_position_embeddings or 0)
        num_grid = int(num_pos**0.5)
        grid = jnp.asarray(grid_thw)
        grid_ts = grid[:, 0].astype(jnp.int32)
        grid_hs = grid[:, 1].astype(jnp.int32)
        grid_ws = grid[:, 2].astype(jnp.int32)

        idx_arrays = [[], [], [], []]
        weight_arrays = [[], [], [], []]
        for i in range(grid_thw.shape[0]):
            h, w = int(grid_hs[i]), int(grid_ws[i])
            h_idxs = jnp.linspace(0, num_grid - 1, h)
            w_idxs = jnp.linspace(0, num_grid - 1, w)
            h_floor = h_idxs.astype(jnp.int32)
            w_floor = w_idxs.astype(jnp.int32)
            h_ceil = jnp.minimum(h_floor + 1, num_grid - 1)
            w_ceil = jnp.minimum(w_floor + 1, num_grid - 1)
            dh = h_idxs - h_floor.astype(h_idxs.dtype)
            dw = w_idxs - w_floor.astype(w_idxs.dtype)
            base_h = h_floor * num_grid
            base_h_ceil = h_ceil * num_grid
            indices = [
                (base_h[:, None] + w_floor[None]).reshape(-1),
                (base_h[:, None] + w_ceil[None]).reshape(-1),
                (base_h_ceil[:, None] + w_floor[None]).reshape(-1),
                (base_h_ceil[:, None] + w_ceil[None]).reshape(-1),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None]).reshape(-1),
                ((1 - dh)[:, None] * dw[None]).reshape(-1),
                (dh[:, None] * (1 - dw)[None]).reshape(-1),
                (dh[:, None] * dw[None]).reshape(-1),
            ]
            for j in range(4):
                idx_arrays[j].append(indices[j])
                weight_arrays[j].append(weights[j])

        idx_concat = [
            jnp.concatenate(arrs) if arrs else jnp.array([], dtype=jnp.int32)
            for arrs in idx_arrays
        ]
        weight_concat = [
            jnp.concatenate(arrs) if arrs else jnp.array([], dtype=jnp.float32)
            for arrs in weight_arrays
        ]
        if idx_concat[0].shape[0] == 0:
            return jnp.zeros((0, self.spec.vision_hidden_size))

        idx_tensor = jnp.stack(idx_concat, axis=0)
        weight_tensor = jnp.stack(weight_concat, axis=0)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[..., None]
        patch_pos_embeds = jnp.sum(pos_embeds, axis=0)

        merge = self.spec.vision_spatial_merge_size
        out_chunks = []
        offset = 0
        for i in range(grid_thw.shape[0]):
            t, h, w = int(grid_ts[i]), int(grid_hs[i]), int(grid_ws[i])
            count = h * w
            pos_embed = patch_pos_embeds[offset : offset + count]
            offset += count
            if t > 1:
                pos_embed = jnp.repeat(pos_embed, t, axis=0)
            pos_embed = pos_embed.reshape(
                t, h // merge, merge, w // merge, merge, -1
            ).transpose(0, 1, 3, 2, 4, 5)
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])
            out_chunks.append(pos_embed)
        return (
            jnp.concatenate(out_chunks, axis=0)
            if out_chunks
            else jnp.zeros((0, self.spec.vision_hidden_size))
        )

    def __call__(
        self,
        pixel_values: jax.Array,
        grid_thw: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """Forward pass. Returns (merged_tokens, deepstack_features)."""
        x = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
            x = x + pos_embeds.astype(x.dtype)
        rotary_emb = self._rot_pos_emb(grid_thw)
        cos = jnp.cos(rotary_emb).astype(x.dtype)
        sin = jnp.sin(rotary_emb).astype(x.dtype)
        cu_seqlens = self._get_cu_seqlens(grid_thw)

        deepstack_feats = []
        for i, block in enumerate(self.blocks):
            cu = (
                jnp.array([0, x.shape[0]], dtype=jnp.int32)
                if i in self.spec.vision_fullatt_block_indexes
                else cu_seqlens
            )
            x = block(x, cos, sin, cu)
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
            use_bias=spec.text_attention_bias,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_num_kv_heads * spec.text_head_dim,
            use_bias=spec.text_attention_bias,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            spec.text_hidden_size,
            spec.text_num_kv_heads * spec.text_head_dim,
            use_bias=spec.text_attention_bias,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            spec.text_num_heads * spec.text_head_dim,
            spec.text_hidden_size,
            use_bias=spec.text_attention_bias,
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
        q = self.q_proj(x).reshape(
            B, T, self.spec.text_num_heads, self.spec.text_head_dim
        )
        k = self.k_proj(x).reshape(
            B, T, self.spec.text_num_kv_heads, self.spec.text_head_dim
        )
        v = self.v_proj(x).reshape(
            B, T, self.spec.text_num_kv_heads, self.spec.text_head_dim
        )
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
        repeats = 1
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
            self.input_norm(x),
            cos,
            sin,
            attention_mask,
            kv_cache=kv_cache,
            positions=positions,
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
    hidden_size = int(text_cfg.hidden_size)
    num_attention_heads = int(text_cfg.num_attention_heads)
    num_hidden_layers = int(text_cfg.num_hidden_layers)
    num_kv_heads = int(text_cfg.num_key_value_heads)
    intermediate_size = int(text_cfg.intermediate_size)
    vocab_size = int(text_cfg.vocab_size)
    rms_norm_eps = float(text_cfg.rms_norm_eps)
    head_dim = int(
        getattr(text_cfg, "head_dim", None) or (hidden_size // num_attention_heads)
    )

    rope_params = getattr(text_cfg, "rope_parameters", None)
    if isinstance(rope_params, dict):
        rope_section = rope_params.get("mrope_section", [head_dim // 2])
        mrope_interleaved = bool(rope_params.get("mrope_interleaved", False))
    else:
        rope_section = [head_dim // 2]
        mrope_interleaved = False
    rope_section = tuple(int(x) for x in rope_section)
    rope_theta = getattr(text_cfg, "rope_theta", 500000.0)

    vision_fullatt = tuple(getattr(vision_cfg, "fullatt_block_indexes", ()) or ())
    vision_deepstack = tuple(
        getattr(vision_cfg, "deepstack_visual_indexes", [8, 16, 24]) or [8, 16, 24]
    )
    patch_sz = vision_cfg.patch_size if vision_cfg else 16
    window_sz = patch_sz * getattr(vision_cfg, "spatial_merge_size", 2)

    return Qwen3VLSpec(
        text_hidden_size=hidden_size,
        text_num_heads=num_attention_heads,
        text_num_layers=num_hidden_layers,
        text_num_kv_heads=num_kv_heads,
        text_head_dim=head_dim,
        text_intermediate_size=intermediate_size,
        text_rope_theta=rope_theta,
        text_rope_section=rope_section,
        text_mrope_interleaved=mrope_interleaved,
        text_rms_norm_eps=rms_norm_eps,
        text_vocab_size=vocab_size,
        text_attention_bias=getattr(text_cfg, "attention_bias", False),
        vision_hidden_size=vision_cfg.hidden_size if vision_cfg else 0,
        vision_out_hidden_size=vision_cfg.out_hidden_size if vision_cfg else 0,
        vision_depth=vision_cfg.depth if vision_cfg else 0,
        vision_num_heads=vision_cfg.num_heads if vision_cfg else 0,
        vision_intermediate_size=vision_cfg.intermediate_size if vision_cfg else 0,
        vision_patch_size=patch_sz,
        vision_temporal_patch_size=getattr(vision_cfg, "temporal_patch_size", 2)
        if vision_cfg
        else 2,
        vision_spatial_merge_size=getattr(vision_cfg, "spatial_merge_size", 2)
        if vision_cfg
        else 2,
        vision_in_channels=getattr(vision_cfg, "in_channels", 3) if vision_cfg else 3,
        vision_num_position_embeddings=getattr(
            vision_cfg, "num_position_embeddings", None
        )
        if vision_cfg
        else None,
        vision_deepstack_indexes=vision_deepstack,
        vision_fullatt_block_indexes=vision_fullatt,
        vision_window_size=window_sz,
        image_token_id=config.image_token_id,
        video_token_id=getattr(config, "video_token_id", 151656),
        vision_start_token_id=config.vision_start_token_id,
        tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
    )


class Qwen3VLModel(nnx.Module):
    """Qwen3-VL model (vision + text backbone)."""

    def __init__(
        self, config: Qwen3VLModelConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
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

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_mask = visual_mask.reshape(-1).astype(bool)
        idx = jnp.where(flat_mask, size=features.shape[0], fill_value=-1)[0]
        valid = idx >= 0
        safe_idx = jnp.where(valid, idx, 0)
        updates = jnp.where(
            valid[:, None],
            features.astype(hidden.dtype),
            jnp.zeros_like(features, dtype=hidden.dtype),
        )
        flat_hidden = flat_hidden.at[safe_idx].add(updates)
        return flat_hidden.reshape(hidden.shape)

    @staticmethod
    def _check_placeholder_match(
        n_placeholders: jax.Array,
        n_features: jax.Array,
        modality: str,
    ) -> None:
        n_placeholders_int = int(np.asarray(n_placeholders))
        n_features_int = int(np.asarray(n_features))
        if n_placeholders_int != n_features_int:
            raise ValueError(
                f"{modality.capitalize()} features and {modality} tokens do not match, "
                f"tokens: {n_placeholders_int}, features: {n_features_int}"
            )

    def _inject_modal_embeddings(
        self,
        hidden: jax.Array,
        input_ids: jax.Array,
        token_id: int,
        features: jax.Array,
        modality: str,
    ) -> tuple[jax.Array, jax.Array]:
        mask = input_ids == token_id
        n_placeholders = jnp.sum(mask).astype(jnp.int32)
        n_features = jnp.array(features.shape[0], dtype=jnp.int32)
        jax.debug.callback(
            lambda n_p, n_f: self._check_placeholder_match(n_p, n_f, modality),
            n_placeholders,
            n_features,
        )

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_mask = mask.reshape(-1).astype(bool)
        idx = jnp.where(flat_mask, size=features.shape[0], fill_value=-1)[0]
        valid = idx >= 0
        safe_idx = jnp.where(valid, idx, 0)
        updates = jnp.where(
            valid[:, None],
            features.astype(hidden.dtype),
            jnp.zeros_like(features, dtype=hidden.dtype),
        )
        flat_hidden = flat_hidden.at[safe_idx].set(updates)
        return flat_hidden.reshape(hidden.shape), mask

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        pixel_values: jax.Array | None = None,
        pixel_values_videos: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        video_grid_thw: jax.Array | None = None,
        positions: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        hidden = self.embed_tokens(input_ids)
        batch = hidden.shape[0]
        is_decode = kv_cache is not None

        image_mask = None
        video_mask = None
        deepstack_image: tuple[jax.Array, ...] | None = None
        deepstack_video: tuple[jax.Array, ...] | None = None
        if not is_decode and self.visual is not None:
            if pixel_values is not None and image_grid_thw is not None:
                image_embeds, deepstack_image = self.visual(
                    pixel_values, image_grid_thw
                )
                hidden, image_mask = self._inject_modal_embeddings(
                    hidden,
                    input_ids,
                    self.spec.image_token_id,
                    image_embeds,
                    modality="image",
                )

            if pixel_values_videos is not None and video_grid_thw is not None:
                video_embeds, deepstack_video = self.visual(
                    pixel_values_videos,
                    video_grid_thw,
                )
                hidden, video_mask = self._inject_modal_embeddings(
                    hidden,
                    input_ids,
                    self.spec.video_token_id,
                    video_embeds,
                    modality="video",
                )

        rope_deltas = None
        if is_decode and positions is not None:
            rope_deltas_from_cache = (
                kv_cache.rope_deltas if kv_cache is not None else None
            )
            if rope_deltas_from_cache is not None:
                pos_1d = positions.astype(jnp.int32) + rope_deltas_from_cache
                position_ids = jnp.broadcast_to(
                    pos_1d[None, :, :],
                    (3, batch, pos_1d.shape[-1]),
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
            else:
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
            position_ids, rope_deltas = get_rope_index(
                spatial_merge_size=self.spec.vision_spatial_merge_size,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                image_token_id=self.spec.image_token_id,
                video_token_id=self.spec.video_token_id,
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

        all_hidden: list[jax.Array] | None = [] if output_hidden_states else None
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
            if (
                deepstack_image is not None
                and image_mask is not None
                and i < len(deepstack_image)
            ):
                hidden = self._apply_deepstack(hidden, image_mask, deepstack_image[i])
            if (
                deepstack_video is not None
                and video_mask is not None
                and i < len(deepstack_video)
            ):
                hidden = self._apply_deepstack(hidden, video_mask, deepstack_video[i])
            if output_hidden_states:
                assert all_hidden is not None
                all_hidden.append(hidden)

        hidden = self.norm(hidden)
        if output_hidden_states:
            assert all_hidden is not None
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
        rope_deltas_for_cache = rope_deltas
        new_kv_cache = KVCache.update(
            kv_cache,
            keys=keys,
            values=values,
            positions=pos_for_cache,
            attention_mask=attention_mask,
            rope_deltas=rope_deltas_for_cache,
        )

        return ModelOutput(
            last_hidden_state=hidden,
            kv_cache=new_kv_cache,
            hidden_states=all_hidden,
        )


class Qwen3VLForCausalLM(
    nnx.Module, ModelForCausalLM, GeneratorMixin, LogitsProcessorMixin
):
    """Qwen3-VL for causal language modeling (vision + text generation)."""

    def __init__(
        self, config: Qwen3VLModelConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
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
        pixel_values_videos: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        video_grid_thw: jax.Array | None = None,
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
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            positions=positions,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states or False,
        )
        return CausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )
