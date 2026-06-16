"""Runtime patch: compute the vLLM LM-head projection (logits) in FP32.

Computing lmhead matmul in fp32 reduces the inference-vs-trainer logprob mismatch that destabilizes training in RL.

vLLM has no native knob for this, so we monkey-patch
``vllm.model_executor.layers.logits_processor.LogitsProcessor`` inside each vLLM
worker process. The patch is installed unconditionally (see
``new_inference_worker_wrap``) but is a strict no-op unless the owning
``LogitsProcessor`` was built with
``VllmConfig.additional_config["skyrl_enable_fp32_lm_head"]`` set. SkyRL sets
that flag in ``build_vllm_cli_args`` when
``generator.inference_engine.enable_fp32_lm_head=true``.

The flag is read in the patched ``__init__`` via ``get_current_vllm_config()``,
which is active during model build, and cached per-instance; the patched
``_get_logits`` then upcasts the unquantized lm-head GEMM to fp32. Quantized,
LoRA-wrapped, and bias'd heads fall back to the original implementation
(matching sglang, which only special-cases the plain ``.weight`` and GGUF
paths).

TODO: remove once vLLM supports fp32 logits natively (server/engine arg +
LogitsProcessor branch).
"""

import torch
from loguru import logger

_PATCHED = False

# Key under VllmConfig.additional_config that gates the fp32 lm-head behavior.
# Must match the key set in
# skyrl.backends.skyrl_train.inference_servers.utils.build_vllm_cli_args.
ADDITIONAL_CONFIG_KEY = "skyrl_enable_fp32_lm_head"


def apply_fp32_lm_head_patch() -> None:
    """Install the fp32 lm-head patch on ``LogitsProcessor`` (idempotent).

    Safe to call in every worker process: installs unconditionally but only
    changes behavior for ``LogitsProcessor`` instances whose ``additional_config``
    has ``skyrl_enable_fp32_lm_head=True``.
    """
    global _PATCHED
    if _PATCHED:
        return

    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    orig_init = LogitsProcessor.__init__
    orig_get_logits = LogitsProcessor._get_logits

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        enabled = False
        try:
            cfg = get_current_vllm_config()
            additional_config = getattr(cfg, "additional_config", None) or {}
            enabled = bool(additional_config.get(ADDITIONAL_CONFIG_KEY, False))
        except Exception:
            # get_current_vllm_config() may be unavailable in some contexts
            # (e.g. unit tests building a bare LogitsProcessor); default to off.
            enabled = False
        self._skyrl_fp32_lm_head = enabled
        if enabled:
            logger.info("SkyRL: fp32 LM head enabled for this LogitsProcessor")

    def patched_get_logits(self, hidden_states, lm_head, embedding_bias):
        if not getattr(self, "_skyrl_fp32_lm_head", False):
            return orig_get_logits(self, hidden_states, lm_head, embedding_bias)

        # Only the plain unquantized ``.weight`` head is upcast; anything else
        # (quantized / LoRA-wrapped / bias'd) falls back to the original path.
        weight = getattr(lm_head, "weight", None)
        can_fp32 = weight is not None and embedding_bias is None and not hasattr(lm_head, "apply_lora")
        if not can_fp32:
            return orig_get_logits(self, hidden_states, lm_head, embedding_bias)

        logits = torch.matmul(hidden_states.to(torch.float32), weight.to(torch.float32).t())

        # Gather logits for TP (dtype-agnostic; carries fp32 through).
        logits = self._gather_logits(logits)

        # Remove vocab padding (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    LogitsProcessor.__init__ = patched_init
    LogitsProcessor._get_logits = patched_get_logits
    _PATCHED = True
    logger.info("SkyRL: installed fp32 LM head patch on vLLM LogitsProcessor")
