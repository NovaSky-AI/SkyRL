"""``vllm.general_plugins`` entry point: serve DeepSeek-V4.1 checkpoints with vLLM's text-only class.

vLLM registers ``DeepseekV41ForCausalLM`` to its vision-language wrapper, which is not ``SupportsLoRA``,
treats the model as multimodal in the frontend, and finalizes weights only once (it skips the layerwise
reload a weight sync needs). The text-only ``DeepseekV41LLMForCausalLM`` (LoRA, drops ``vision.*`` /
``aligner.*`` / ``image_*`` / ``mtp.*`` on load) has no registry entry of its own. Other vLLM code keys
engram, cudagraph and tokenizer handling on the architecture string, so the text class is registered under
the same name rather than a new one.

Installed in every vLLM process (the frontend and the engine core must agree on the class). Set
``SKYRL_DEEPSEEK_V41_TEXT_ONLY=0`` to keep vLLM's vision-language class.

TODO: remove once the pinned vLLM's own V4.1 class covers all three reasons, then serve it with
``language_model_only=True`` (already in vLLM 0.32.0; skips the vision tower and multimodal input handling):
- LoRA on the vision-language class: vllm-project/vllm#57245 (open; also adapts the compressor and grouped
  output projections, which the text class's LoRA leaves out);
- layerwise reload: vllm-project/vllm#64357 (open; drops the VL wrappers' finalize-during-load workaround and runs
  the model-level hook once per reload, the same fix patch_deepseek_v4_reload.py backports).
Check that the pinned ``vl_model.py`` class lists ``SupportsLoRA`` and no longer sets
``finalizes_weights_during_load``, then drop this module, its entry point in ``pyproject.toml`` and
``SKYRL_DEEPSEEK_V41_TEXT_ONLY``, and re-run the V4.1 roundtrip test with weight sync plus a LoRA step.
"""

import os

ENV = "SKYRL_DEEPSEEK_V41_TEXT_ONLY"
TEXT_CLASS = "vllm.models.deepseek_v41.nvidia.model:DeepseekV41LLMForCausalLM"


def vllm_general_plugin() -> None:
    if os.environ.get(ENV, "1") == "0":
        return
    try:
        from vllm import ModelRegistry
    except ImportError:
        return
    ModelRegistry.register_model("DeepseekV41ForCausalLM", TEXT_CLASS)
