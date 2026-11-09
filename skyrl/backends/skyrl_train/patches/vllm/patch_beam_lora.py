"""``vllm.general_plugins`` entry point: serve Beam (``BeamForCausalLM``) with LoRA.

vLLM 0.33.0's ``vllm.models.beam`` model does not declare ``SupportsLoRA`` ("LoRA support is deferred to a
follow-up change"), so ``--enable-lora`` is rejected at config time. Everything else LoRA needs is already in
vLLM 0.33.0, as it was in the vLLM build SkyRL served Beam with before (``vllm 0.1.0+beam2``, whose Beam
LoRA support this module reproduces):

- ``packed_modules_mapping``: ``q/k/v/output_gate_proj`` pack into ``qkv_gate_proj`` (4 slices, wrapped by
  ``MergedColumnParallelLinearVariableSliceWithLoRA``) when the KV heads cover the TP size, otherwise into
  ``qkv_proj`` plus a separate ``output_gate_proj``; the dense MLP's ``gate/up_proj`` pack into ``gate_up_proj``.
- The attention ``self_attn.gate_proj`` is renamed to ``output_gate_proj`` by ``hf_to_vllm_mapper``, which the
  LoRA loaders apply as a rename-only mapper, so an adapter's attention gate never matches the MLP's
  ``gate_proj``.
- Routed experts take vLLM's FusedMoE LoRA with per-expert adapter keys (``mlp.experts.<n>.<proj>``).

What the class below adds on top of vLLM's:

- ``SupportsLoRA`` with ``embedding_modules = {}`` (Beam has no LoRA on its embeddings or LM head).
- No low-latency packed ``qkv_gate_proj`` GEMM under LoRA. ``BeamAttention.warmup_projection`` turns it on
  for unquantized weights on SM100, and ``forward`` then multiplies by ``qkv_gate_proj.weight`` directly,
  which skips a LoRA wrapper. With LoRA the projection always runs through the (wrapped) module.

The architecture is registered to this module rather than patched on vLLM's class: from vLLM 0.33 the
config-time LoRA check reads the class's model info, inspected in a subprocess and cached on disk keyed by
the hash of the class's module file.

TODO: remove once the pinned vLLM's ``BeamForCausalLM`` lists ``SupportsLoRA`` (drop this module and its entry
point in ``pyproject.toml``), and re-run the ``beam-tiny`` adapter rows of
``gpu_ci/megatron/test_megatron_lora_models.py``.
"""

from __future__ import annotations

ARCH = "BeamForCausalLM"
_LORA_CLASS = None


def build_beam_lora_class():
    """Return the LoRA-capable subclass of vLLM's ``BeamForCausalLM`` (built once)."""
    global _LORA_CLASS
    if _LORA_CLASS is not None:
        return _LORA_CLASS

    from vllm.model_executor.models.interfaces import SupportsLoRA
    from vllm.models.beam.nvidia.attention import BeamAttention
    from vllm.models.beam.nvidia.model import BeamForCausalLM

    class BeamForCausalLMWithLoRA(BeamForCausalLM, SupportsLoRA):
        """vLLM's ``BeamForCausalLM`` with LoRA; see the module docstring."""

        embedding_modules: dict[str, str] = {}

        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._skyrl_lora_enabled = vllm_config.lora_config is not None

        def process_weights_after_loading(self) -> None:
            if not self._skyrl_lora_enabled:
                super().process_weights_after_loading()
                return
            for module in self.model.modules():
                if isinstance(module, BeamAttention):
                    module.low_latency_projection = False

    BeamForCausalLMWithLoRA.__name__ = ARCH
    BeamForCausalLMWithLoRA.__qualname__ = ARCH
    _LORA_CLASS = BeamForCausalLMWithLoRA
    return _LORA_CLASS


def __getattr__(name: str):
    # Resolved lazily by vLLM's registry (see ``vllm_general_plugin``): importing this module must not import
    # vLLM, so the class is only built when the registry asks for it.
    if name == ARCH:
        return build_beam_lora_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def vllm_general_plugin() -> None:
    """Register the LoRA-capable class under vLLM's Beam architecture name, in every vLLM process."""
    try:
        from vllm import ModelRegistry
    except ImportError:
        return
    if getattr(ModelRegistry, "register_model", None) is None:
        return
    ModelRegistry.register_model(ARCH, f"{__name__}:{ARCH}")
