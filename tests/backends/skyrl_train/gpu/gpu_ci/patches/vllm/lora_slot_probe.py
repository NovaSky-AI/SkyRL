"""Test-only vLLM worker extension that reports the LoRA slot ranks of the loaded model."""

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)


class LoRASlotProbeWorkerWrap(NewInferenceWorkerWrap):
    def skyrl_test_lora_slot_ranks(self) -> dict:
        """Map each MoE / linear LoRA wrapper to the sorted ranks of its stacked slot buffers."""
        from vllm.lora.layers import FusedMoEWithLoRA

        manager = self.model_runner.lora_manager._adapter_manager
        ranks = {}
        for name, module in manager.modules.items():
            if isinstance(module, FusedMoEWithLoRA):
                a = module.w13_lora_a_stacked + module.w2_lora_a_stacked
                b = module.w13_lora_b_stacked + module.w2_lora_b_stacked
                ranks[name] = sorted({t.shape[-2] for t in a} | {t.shape[-1] for t in b})
            elif name.endswith(("qkv_proj", "o_proj", "gate_up_proj", "down_proj")):
                ranks[name] = sorted({t.shape[-2] for t in module.lora_a_stacked})
        return ranks


LORA_SLOT_PROBE_WORKER_EXTENSION_CLS = f"{__name__}.LoRASlotProbeWorkerWrap"
