from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
)

from tx.models.configs import Qwen3VLModelConfig
from tx.models.qwen3_vl_moe import Qwen3VLForCausalLM


def _make_test_mesh() -> jax.sharding.Mesh:
    return jax.make_mesh(
        (1, 1, 1),
        ("fsdp", "ep", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )


def _make_tiny_hf_vl_moe_config() -> Qwen3VLMoeConfig:
    # Keep dimensions tiny for CI speed while exercising MoE + mRoPE codepaths.
    return Qwen3VLMoeConfig(
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        text_config={
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "hidden_act": "silu",
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 8,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "mlp_only_layers": [],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
            },
            # HF Qwen3VLMoeTextRotaryEmbedding currently reads rope_scaling.
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
            },
        },
        # Vision tower is unused in these text-only parity tests, but config must exist.
        vision_config={
            "depth": 0,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "num_position_embeddings": 4,
            "deepstack_visual_indexes": [],
        },
        tie_word_embeddings=False,
    )


def _load_text_weights_from_hf(
    jax_model: Qwen3VLForCausalLM, hf_model: Qwen3VLMoeForConditionalGeneration
) -> None:
    # Embeddings + final norm + lm_head
    jax_model.model.embed_tokens.embedding[...] = hf_model.model.language_model.embed_tokens.weight.detach().cpu().numpy()
    jax_model.model.norm.weight[...] = hf_model.model.language_model.norm.weight.detach().cpu().numpy()
    jax_model.lm_head.kernel[...] = hf_model.lm_head.weight.detach().cpu().numpy().T

    # Decoder layers (text-only parity path)
    num_layers = len(jax_model.model.layers)
    for i in range(num_layers):
        jax_layer = jax_model.model.layers[i]
        hf_layer = hf_model.model.language_model.layers[i]

        # Layer norms
        jax_layer.input_norm.weight[...] = hf_layer.input_layernorm.weight.detach().cpu().numpy()
        jax_layer.post_norm.weight[...] = hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()

        # Attention
        jax_layer.attn.q_proj.kernel[...] = hf_layer.self_attn.q_proj.weight.detach().cpu().numpy().T
        jax_layer.attn.k_proj.kernel[...] = hf_layer.self_attn.k_proj.weight.detach().cpu().numpy().T
        jax_layer.attn.v_proj.kernel[...] = hf_layer.self_attn.v_proj.weight.detach().cpu().numpy().T
        jax_layer.attn.o_proj.kernel[...] = hf_layer.self_attn.o_proj.weight.detach().cpu().numpy().T
        jax_layer.attn.q_norm.weight[...] = hf_layer.self_attn.q_norm.weight.detach().cpu().numpy()
        jax_layer.attn.k_norm.weight[...] = hf_layer.self_attn.k_norm.weight.detach().cpu().numpy()

        # MoE (router + experts)
        jax_layer.mlp.router.kernel[...] = hf_layer.mlp.gate.weight.detach().cpu().numpy().T
        gate_up = hf_layer.mlp.experts.gate_up_proj.detach().cpu().numpy()
        inter = jax_layer.mlp.experts.gate_proj.weight.shape[2]
        # HF gate_up_proj packs [gate, up] in out_features; split then transpose to [in, out].
        jax_layer.mlp.experts.gate_proj.weight[...] = gate_up[:, :inter, :].transpose(0, 2, 1)
        jax_layer.mlp.experts.up_proj.weight[...] = gate_up[:, inter:, :].transpose(0, 2, 1)
        hf_down = hf_layer.mlp.experts.down_proj.detach().cpu().numpy()
        assert hf_down.shape == jax_layer.mlp.experts.down_proj.weight.shape
        jax_layer.mlp.experts.down_proj.weight[...] = hf_down


def _build_tiny_models() -> tuple[Qwen3VLMoeForConditionalGeneration, Qwen3VLForCausalLM, jax.sharding.Mesh]:
    torch.manual_seed(0)
    hf_config = _make_tiny_hf_vl_moe_config()
    hf_model = Qwen3VLMoeForConditionalGeneration(hf_config).eval()

    jax_config = Qwen3VLModelConfig(
        hf_config,
        max_lora_adapters=0,
        max_lora_rank=0,
        shard_attention_heads=True,
        gradient_checkpointing=False,
    )

    mesh = _make_test_mesh()
    with jax.set_mesh(mesh):
        jax_model = Qwen3VLForCausalLM(jax_config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        _load_text_weights_from_hf(jax_model, hf_model)

    return hf_model, jax_model, mesh


def test_qwen3_vl_moe_text_prefill_parity_with_hf():
    hf_model, jax_model, mesh = _build_tiny_models()

    input_ids = torch.tensor(
        [
            [11, 12, 13, 14, 0, 0],
            [21, 22, 23, 24, 25, 26],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    with jax.set_mesh(mesh):
        jax_outputs = jax_model(
            np.asarray(input_ids, dtype=np.int32),
            attention_mask=np.asarray(attention_mask, dtype=np.int32),
            output_hidden_states=True,
        )
        assert jax_outputs.hidden_states is not None
        assert hf_outputs.hidden_states is not None
        jax_logits = jax_model.compute_logits(jax_outputs.last_hidden_state)

    np.testing.assert_allclose(
        np.asarray(hf_outputs.hidden_states[0], dtype=np.float32),
        np.asarray(jax_outputs.hidden_states[0], dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(hf_outputs.hidden_states[1], dtype=np.float32),
        np.asarray(jax_outputs.hidden_states[1], dtype=np.float32),
        rtol=1.5e-2,
        atol=1.5e-2,
    )
    # HF VL-MoE exposes pre-final-norm hidden states here, while JAX includes
    # final norm in hidden_states. Align by stage instead of raw index.
    hf_last = np.asarray(hf_outputs.hidden_states[-1], dtype=np.float32)
    if len(jax_outputs.hidden_states) == len(hf_outputs.hidden_states):
        jax_last_aligned = np.asarray(jax_outputs.hidden_states[-1], dtype=np.float32)
    else:
        jax_last_aligned = np.asarray(jax_outputs.hidden_states[-2], dtype=np.float32)
    np.testing.assert_allclose(hf_last, jax_last_aligned, rtol=1.5e-2, atol=1.5e-2)
    valid = np.asarray(attention_mask, dtype=bool)
    hf_logits = np.asarray(hf_outputs.logits, dtype=np.float32)
    jax_logits_np = np.asarray(jax_logits, dtype=np.float32)
    np.testing.assert_allclose(
        hf_logits[valid],
        jax_logits_np[valid],
        rtol=1.5e-2,
        atol=1e-2,
    )


def test_qwen3_vl_moe_text_decode_step_parity_with_hf():
    hf_model, jax_model, mesh = _build_tiny_models()

    # Prefill 4 tokens, then decode 1 token.
    prefill_ids = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.long)
    prefill_mask = torch.ones_like(prefill_ids, dtype=torch.long)
    decode_ids = torch.tensor([[15], [25]], dtype=torch.long)
    decode_mask = torch.ones((2, 5), dtype=torch.long)

    with torch.no_grad():
        hf_prefill = hf_model(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
        hf_decode = hf_model(
            input_ids=decode_ids,
            attention_mask=decode_mask,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )

    with jax.set_mesh(mesh):
        jax_prefill = jax_model(
            np.asarray(prefill_ids, dtype=np.int32),
            attention_mask=np.asarray(prefill_mask, dtype=np.int32),
        )
        assert jax_prefill.kv_cache is not None
        # Match generation runtime behavior: decode updates into a pre-allocated KV cache.
        jax_prefill_cache = jax_prefill.kv_cache.pad_to_length(int(decode_mask.shape[1]))

        decode_positions = np.asarray(jax_prefill_cache.cache_position[:, None], dtype=np.int32)
        jax_decode = jax_model(
            np.asarray(decode_ids, dtype=np.int32),
            attention_mask=np.asarray(decode_mask, dtype=np.int32),
            kv_cache=jax_prefill_cache,
            positions=decode_positions,
        )
        jax_decode_logits = jax_model.compute_logits(jax_decode.last_hidden_state)

    np.testing.assert_allclose(
        np.asarray(hf_decode.logits, dtype=np.float32),
        np.asarray(jax_decode_logits, dtype=np.float32),
        rtol=2e-1,
        atol=9e-2,
    )

