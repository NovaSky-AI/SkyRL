"""``vllm.general_plugins`` entry point: serve DeepSeek-V4.1 checkpoints with vLLM's text-only class.

vLLM registers ``DeepseekV41ForCausalLM`` to its vision-language wrapper, which is not ``SupportsLoRA``,
treats the model as multimodal in the frontend, and finalizes weights only once (it skips the layerwise
reload a weight sync needs). The text-only ``DeepseekV41LLMForCausalLM`` (LoRA, drops ``vision.*`` /
``aligner.*`` / ``image_*`` / ``mtp.*`` on load) has no registry entry of its own. Other vLLM code keys
engram, cudagraph and tokenizer handling on the architecture string, so the text class is registered under
the same name rather than a new one.

Installed in every vLLM process (the frontend and the engine core must agree on the class). Set
``SKYRL_DEEPSEEK_V41_TEXT_ONLY=0`` to keep vLLM's vision-language class.

Remove once vLLM registers a text-only DeepSeek-V4.1 architecture.
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
