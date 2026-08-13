"""Skip Megatron's MLA THD value pad on Blackwell (sm100+).

For packed (``qkv_format="thd"``) execution, Megatron-core pads the MLA value
tensor from ``v_head_dim`` (e.g. 128) up to the QK head dim (e.g. 192) in
``_prepare_mla_core_attention_value`` and trims the attention output back
afterwards.

On Blackwell that pad is fatal for training: cuDNN fused attention has no
backward support for ``head_dim > 128`` on sm100+, so with the padded
``head_dim_v == head_dim_qk == 192`` TransformerEngine disables FusedAttention
for training-mode forwards. FlashAttention 2 does not support MLA at all,
FlashAttention 3 is sm90-only, and UnfusedDotProductAttention does not support
context parallelism - so MLA + CP training raises
``ValueError: No dot product attention backend is available``. Inference-mode
forwards (logprob computation) are unaffected, which makes the failure appear
only at the first ``forward_backward``.

cuDNN fused attention natively supports MLA's unequal QK/V head dims
(192/128), including THD + context parallelism with the ``p2p`` exchange, for
both forward and backward. Skipping the pad simply selects that native path
(and saves the pad/trim memory traffic). Behavior on pre-Blackwell devices is
left unchanged.
"""

from loguru import logger

_APPLIED = False


def apply_megatron_mla_thd_v_pad_patch() -> None:
    """Patch ``_prepare_mla_core_attention_value`` to skip the V pad on sm100+."""
    global _APPLIED
    if _APPLIED:
        return

    import torch
    from megatron.core.transformer import multi_latent_attention as mla

    orig_prepare = mla._prepare_mla_core_attention_value

    def patched_prepare(parallel_attention, query, value, packed_seq_params):
        if (
            value is not None
            and packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
            and query.shape[-1] != value.shape[-1]
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (10, 0)
        ):
            orig_v_dim = value.shape[-1]
            return value, False, orig_v_dim, orig_v_dim
        return orig_prepare(parallel_attention, query, value, packed_seq_params)

    mla._prepare_mla_core_attention_value = patched_prepare
    _APPLIED = True
    logger.info("Applied Megatron MLA THD V-pad skip for sm100+ (native unequal-head-dim attention)")
