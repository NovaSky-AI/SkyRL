"""CPU tests for ``patch_mxfp4_trtllm_lora_moe``: env gating and MXFP4 MoE backend registration.

The kernel itself needs a Blackwell GPU and a model; its throughput and numerics were checked end to end with
the vLLM inference bench (see the patch docstring). Here: the patch is inert without the env var, and with it the
LoRA-capable experts class is offered under FLASHINFER_TRTLLM_MXFP4_BF16, which the generic priority list now tries
just before Marlin.
"""

import pytest


@pytest.mark.vllm
class TestMxfp4TrtllmLoraMoePatch:
    @pytest.fixture(autouse=True)
    def fresh(self, monkeypatch):
        pytest.importorskip("vllm")
        oracle = pytest.importorskip("vllm.model_executor.layers.fused_moe.oracle.mxfp4")
        fi_moe = pytest.importorskip("flashinfer.fused_moe")
        if not hasattr(fi_moe, "trtllm_fp4_block_scale_routed_moe"):
            pytest.skip("FlashInfer without the FP4 routed MoE")
        from skyrl.backends.skyrl_train.patches.vllm import patch_mxfp4_trtllm_lora_moe as mod

        # Restore vLLM's selection functions and the patch's applied flag after each test.
        monkeypatch.setattr(oracle, "backend_to_kernel_cls", oracle.backend_to_kernel_cls)
        monkeypatch.setattr(oracle, "_get_priority_backends", oracle._get_priority_backends)
        monkeypatch.setattr(mod, "_APPLIED", False)
        self.oracle, self.mod = oracle, mod

    def test_off_without_env(self, monkeypatch):
        monkeypatch.delenv(self.mod.ENV, raising=False)
        before = list(self.oracle._get_priority_backends())
        assert self.mod.apply_mxfp4_trtllm_lora_moe_patch() is False
        assert list(self.oracle._get_priority_backends()) == before

    def test_registers_lora_experts_ahead_of_marlin(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        assert self.mod.apply_mxfp4_trtllm_lora_moe_patch() is True
        assert self.mod.apply_mxfp4_trtllm_lora_moe_patch() is True  # idempotent
        B = self.oracle.Mxfp4MoeBackend

        bf16 = self.oracle.backend_to_kernel_cls(B.FLASHINFER_TRTLLM_MXFP4_BF16)
        assert bf16[-1].__name__ == "TrtLlmMxfp4LoRAExperts" and bf16[-1].supports_lora()
        assert sum(c.__name__ == "TrtLlmMxfp4LoRAExperts" for c in bf16) == 1
        # Not offered under the MXFP8-activation variant: activations stay BF16, as with Marlin.
        mxfp8 = self.oracle.backend_to_kernel_cls(B.FLASHINFER_TRTLLM_MXFP4_MXFP8)
        assert all(c.__name__ != "TrtLlmMxfp4LoRAExperts" for c in mxfp8)

        priority = list(self.oracle._get_priority_backends())
        if B.MARLIN in priority:  # CUDA platforms
            assert priority.index(B.FLASHINFER_TRTLLM_MXFP4_BF16) == priority.index(B.MARLIN) - 1
            assert priority.count(B.FLASHINFER_TRTLLM_MXFP4_BF16) == 1
