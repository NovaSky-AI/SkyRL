import os
import threading


_PATCH_LOCK = threading.Lock()
_MLA_FORWARD_PATCH_APPLIED = False


def _patch_mode_for_model(model_path: str) -> str:
    """
    Patch mode:
    - off/0/false/no: disable
    - force/1/true/on: enable for any MLA model
    - auto (default): enable for Moonlight/DeepSeek model names only
    """
    mode = os.environ.get("SKYRL_MLA_PATCH_MODE", "auto").strip().lower()
    if mode in {"off", "0", "false", "no"}:
        return "off"
    if mode in {"force", "1", "true", "on"}:
        return "force"
    if mode == "auto":
        model_path_lc = (model_path or "").lower()
        if "moonlight" in model_path_lc or "deepseek" in model_path_lc:
            return "auto"
        return "off"
    # Unknown mode: be conservative and do not patch.
    return "off"


def maybe_apply_mla_forward_patch(model_path: str, has_mla: bool) -> bool:
    """
    Apply the MLA forward monkey patch (adapted from verl) once per process.

    Returns True if patching happened in this call, False otherwise.
    """
    if not has_mla:
        return False

    mode = _patch_mode_for_model(model_path)
    if mode == "off":
        return False

    global _MLA_FORWARD_PATCH_APPLIED
    with _PATCH_LOCK:
        if _MLA_FORWARD_PATCH_APPLIED:
            return False

        try:
            import megatron.core
            import torch.nn.functional as F
            from packaging import version
            from megatron.core.transformer.multi_latent_attention import (
                MultiLatentAttention,
                deprecate_inference_params,
            )
        except Exception as exc:
            print(f"[SkyRL] WARNING: Failed to import MLA patch dependencies: {exc}")
            return False

        mcore_ge_013 = version.parse(megatron.core.__version__) >= version.parse("0.13.0")
        original_forward = MultiLatentAttention.forward

        def patch_forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_context=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            rotary_pos_cos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            position_ids=None,
            sequence_len_offset=None,
            *,
            inference_params=None,
            **kwargs,
        ):
            """
            Forward pass for multi-latent attention (legacy verl-compatible path).

            We intentionally fall back to the original implementation for dynamic
            caching paths because this patch targets the static MLA execution used
            in SkyRL replay debugging.
            """
            # Dynamic/cache paths in newer mcore contain logic absent from the
            # legacy patch. Keep upstream behavior there.
            patched_inference_context = deprecate_inference_params(inference_context, inference_params)
            if getattr(self.config, "cache_mla_latents", False):
                return original_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    key_value_states=key_value_states,
                    inference_context=patched_inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    rotary_pos_cos_sin=rotary_pos_cos_sin,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    position_ids=position_ids,
                    sequence_len_offset=sequence_len_offset,
                    inference_params=inference_params,
                )
            if patched_inference_context is not None and not patched_inference_context.is_static_batching():
                return original_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    key_value_states=key_value_states,
                    inference_context=patched_inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    rotary_pos_cos_sin=rotary_pos_cos_sin,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    position_ids=position_ids,
                    sequence_len_offset=sequence_len_offset,
                    inference_params=inference_params,
                )

            assert attention_bias is None, "Attention bias should not be passed into MLA."
            assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"
            assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."

            # hidden_states: [sq, b, h]
            inference_context = patched_inference_context

            # =====================
            # Query, Key, and Value
            # =====================
            qkv = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
            query, key, value = qkv[:3]
            q_compressed = None
            if len(qkv) > 4:
                q_compressed = qkv[3]

            # ===================================================
            # Adjust key, value for inference
            # ===================================================
            if mcore_ge_013:
                query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
                    inference_context, query, key, value, rotary_pos_emb=None
                )
            else:
                query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
                    inference_context, query, key, value, rotary_pos_emb=None
                )

            # TODO: Currently, TE can only accept contiguous tensors for MLA
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            # ==================================
            # core attention computation
            # ==================================
            non_dsa_thd_qkv_format = (
                packed_seq_params
                and packed_seq_params.qkv_format == "thd"
                and getattr(self.config, "experimental_attention_variant", None) is None
            )
            v_dim = value.shape[-1]
            if non_dsa_thd_qkv_format and query.shape[-1] != v_dim:
                value = F.pad(value, [0, query.shape[-1] - v_dim])
                self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
            if self.checkpoint_core_attention and self.training:
                core_attn_out = self._checkpointed_attention_forward(
                    query, key, value, attention_mask, packed_seq_params=packed_seq_params
                )
            else:
                extra_kwargs = {}
                if getattr(self.config, "experimental_attention_variant", None) == "dsa":
                    extra_kwargs["x"] = hidden_states
                    extra_kwargs["qr"] = q_compressed
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=attn_mask_type,
                    **extra_kwargs,
                )
            if non_dsa_thd_qkv_format:
                if core_attn_out.ndim == 2:
                    core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, value.shape[-1])
                if query.shape[-1] != v_dim:
                    core_attn_out = core_attn_out[..., :v_dim]
                core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

            if self.recompute_up_proj:
                assert self.qkv_up_checkpoint is not None
                self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
                self.qkv_up_checkpoint = None

            # =================
            # Output. [sq, b, h]
            # =================
            output, bias = self.linear_proj(core_attn_out)

            return output, bias

        try:
            MultiLatentAttention.forward = patch_forward
            _MLA_FORWARD_PATCH_APPLIED = True
            return True
        except Exception as exc:
            print(f"[SkyRL] WARNING: Failed to apply MLA forward patch: {exc}")
            return False
