import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
)

from tx.models.configs import Qwen3VLModelConfig
from tx.models.qwen3_vl import (
    Qwen3VLModel,
    build_additive_causal_mask,
    get_rope_index,
    spec_from_config,
)


def _hf_reference_rope_index(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grid_thw: np.ndarray,
    video_grid_thw: np.ndarray,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if video_grid_thw.size > 0:
        video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
        video_grid_thw[:, 0] = 1

    image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw.size > 0 else []
    video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw.size > 0 else []

    batch, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch, seq_len), dtype=np.int32)
    mrope_deltas = []

    image_index = 0
    video_index = 0
    for i in range(batch):
        ids = input_ids[i][attention_mask[i] == 1]
        vision_start_indices = np.argwhere(ids == vision_start_token_id).reshape(-1)
        vision_tokens = (
            ids[vision_start_indices + 1]
            if vision_start_indices.size > 0
            else np.array([], dtype=ids.dtype)
        )
        image_nums = int(np.sum(vision_tokens == image_token_id))
        video_nums = int(np.sum(vision_tokens == video_token_id))

        input_tokens = ids.tolist()
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
                t, h, w = image_grid_thw_list[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw_list[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

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
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.arange(text_len, dtype=np.int32)[None, :].repeat(3, axis=0) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        position_ids[:, i, attention_mask[i] == 1] = llm_positions
        mrope_deltas.append(int(llm_positions.max()) + 1 - seq_len)

    return position_ids, np.asarray(mrope_deltas, dtype=np.int32)[:, None]


def _make_tiny_vl_model() -> Qwen3VLModel:
    base_cfg = Qwen3VLMoeConfig(
        image_token_id=7,
        video_token_id=8,
        vision_start_token_id=6,
        text_config={
            "vocab_size": 128,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rope_parameters": {"mrope_section": [2, 1, 1], "mrope_interleaved": False},
            "attention_bias": False,
        },
        vision_config={
            "depth": 0,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 8,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "num_position_embeddings": 4,
        },
    )
    cfg = Qwen3VLModelConfig(
        base_cfg,
        max_lora_adapters=0,
        max_lora_rank=0,
        shard_attention_heads=True,
        gradient_checkpointing=False,
    )
    return Qwen3VLModel(cfg, dtype=jnp.float32, rngs=nnx.Rngs(0))


def test_qwen3_vl_get_rope_index_parity_image_video_mixed():
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652

    input_ids = np.array(
        [
            [
                11,
                vision_start_token_id,
                image_token_id,
                image_token_id,
                12,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                21,
                vision_start_token_id,
                video_token_id,
                video_token_id,
                22,
                vision_start_token_id,
                video_token_id,
                video_token_id,
                23,
                0,
                0,
                0,
            ],
            [
                31,
                vision_start_token_id,
                image_token_id,
                image_token_id,
                32,
                vision_start_token_id,
                video_token_id,
                video_token_id,
                33,
                0,
                0,
                0,
            ],
        ],
        dtype=np.int32,
    )
    attention_mask = (input_ids != 0).astype(np.int32)

    image_grid_thw = np.array([[1, 2, 4], [1, 2, 4]], dtype=np.int32)
    video_grid_thw = np.array([[2, 2, 4], [1, 2, 4]], dtype=np.int32)

    ref_pos, ref_delta = _hf_reference_rope_index(
        input_ids,
        attention_mask,
        image_grid_thw,
        video_grid_thw,
        spatial_merge_size=2,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
    )

    pos, delta = get_rope_index(
        spatial_merge_size=2,
        input_ids=jnp.asarray(input_ids),
        image_grid_thw=jnp.asarray(image_grid_thw),
        video_grid_thw=jnp.asarray(video_grid_thw),
        attention_mask=jnp.asarray(attention_mask),
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_id=vision_start_token_id,
    )

    pos_np = np.asarray(pos)
    delta_np = np.asarray(delta)
    assert pos_np.shape == (3, 3, input_ids.shape[1])
    assert delta_np.shape == (3, 1)
    np.testing.assert_array_equal(pos_np, ref_pos)
    np.testing.assert_array_equal(delta_np, ref_delta)


def test_qwen3_vl_placeholder_injection_image_video_and_mismatch():
    model = _make_tiny_vl_model()

    hidden = jnp.zeros((1, 6, 8), dtype=jnp.float32)
    input_ids = jnp.array([[5, 7, 7, 9, 8, 10]], dtype=jnp.int32)
    image_features = jnp.array([[1.0] * 8, [2.0] * 8], dtype=jnp.float32)
    video_features = jnp.array([[3.0] * 8], dtype=jnp.float32)

    hidden, image_mask = model._inject_modal_embeddings(
        hidden, input_ids, 7, image_features, modality="image"
    )
    hidden, video_mask = model._inject_modal_embeddings(
        hidden, input_ids, 8, video_features, modality="video"
    )

    hidden_np = np.asarray(hidden)
    assert int(np.asarray(image_mask).sum()) == 2
    assert int(np.asarray(video_mask).sum()) == 1
    np.testing.assert_array_equal(hidden_np[0, 1], np.asarray(image_features[0]))
    np.testing.assert_array_equal(hidden_np[0, 2], np.asarray(image_features[1]))
    np.testing.assert_array_equal(hidden_np[0, 4], np.asarray(video_features[0]))

    with pytest.raises(
        ValueError, match="Image features and image tokens do not match"
    ):
        out_hidden, _ = model._inject_modal_embeddings(
            hidden,
            input_ids,
            7,
            jnp.array([[9.0] * 8], dtype=jnp.float32),
            modality="image",
        )
        jax.block_until_ready(out_hidden)


def test_qwen3_vl_deepstack_addition_mixed_visual_masks():
    model = _make_tiny_vl_model()

    hidden = jnp.zeros((1, 6, 8), dtype=jnp.float32)
    image_mask = jnp.array([[False, True, True, False, False, False]])
    video_mask = jnp.array([[False, False, False, False, True, False]])

    image_deepstack = jnp.array([[0.5] * 8, [1.0] * 8], dtype=jnp.float32)
    video_deepstack = jnp.array([[2.0] * 8], dtype=jnp.float32)

    hidden = model._apply_deepstack(hidden, image_mask, image_deepstack)
    hidden = model._apply_deepstack(hidden, video_mask, video_deepstack)

    hidden_np = np.asarray(hidden)
    np.testing.assert_array_equal(hidden_np[0, 1], np.asarray(image_deepstack[0]))
    np.testing.assert_array_equal(hidden_np[0, 2], np.asarray(image_deepstack[1]))
    np.testing.assert_array_equal(hidden_np[0, 4], np.asarray(video_deepstack[0]))
    np.testing.assert_array_equal(hidden_np[0, 0], np.zeros((8,), dtype=np.float32))
    np.testing.assert_array_equal(hidden_np[0, 3], np.zeros((8,), dtype=np.float32))
    np.testing.assert_array_equal(hidden_np[0, 5], np.zeros((8,), dtype=np.float32))


def test_qwen3_vl_additive_causal_mask_matches_expected_pattern():
    attention_mask = jnp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=jnp.int32)
    query_positions = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)

    mask = build_additive_causal_mask(attention_mask, query_positions, kv_len=5)
    mask_np = np.asarray(mask)

    assert mask_np.shape == (2, 1, 3, 5)

    # Batch 0, query at pos=2 can attend keys 0..2, cannot attend pad/future.
    assert np.all(mask_np[0, 0, 2, :3] == 0.0)
    assert np.all(mask_np[0, 0, 2, 3:] < -1e8)

    # Batch 1, query at pos=1 can attend keys 0..1 only.
    assert np.all(mask_np[1, 0, 0, :2] == 0.0)
    assert np.all(mask_np[1, 0, 0, 2:] < -1e8)


def test_qwen3_vl_spec_forces_interleaved_mrope_like_hf():
    base_cfg = Qwen3VLMoeConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            # HF implementation interleaves regardless; verify our spec does too.
            "rope_parameters": {"mrope_section": [2, 1, 1], "mrope_interleaved": False},
        },
        vision_config={
            "depth": 0,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 8,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "num_position_embeddings": 4,
        },
    )
    cfg = Qwen3VLModelConfig(
        base_cfg,
        max_lora_adapters=0,
        max_lora_rank=0,
        shard_attention_heads=True,
        gradient_checkpointing=False,
    )
    spec = spec_from_config(cfg)
    assert spec.text_mrope_interleaved is True


def test_qwen3_vl_accepts_4_plane_position_ids_branch():
    model = _make_tiny_vl_model()
    input_ids = jnp.array([[11, 12, 13]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    text_pos = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    mrope_pos = jnp.stack([text_pos, text_pos, text_pos], axis=0)
    position_ids = jnp.concatenate([text_pos[None, ...], mrope_pos], axis=0)

    out = model(
        input_ids,
        attention_mask=attention_mask,
        positions=position_ids,
    )
    assert out.last_hidden_state.shape == (1, 3, model.spec.text_hidden_size)
