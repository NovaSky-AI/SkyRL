"""CPU test for ``patch_moe_lora_two_kernel``: env gating of vLLM's one-shot fused MoE-LoRA cutoff."""

import pytest


@pytest.mark.vllm
class TestMoeLoraTwoKernelPatch:
    @pytest.fixture(autouse=True)
    def fresh(self, monkeypatch):
        pytest.importorskip("vllm")
        op = pytest.importorskip("vllm.lora.ops.triton_ops.fused_moe_lora_op")
        if not hasattr(op, "_FUSED_MOE_LORA_ONE_SHOT_MAX_RANK"):
            pytest.skip("vLLM without the one-shot MoE-LoRA cutoff")
        from skyrl.backends.skyrl_train.patches.vllm import patch_moe_lora_two_kernel as mod

        monkeypatch.setattr(op, "_FUSED_MOE_LORA_ONE_SHOT_MAX_RANK", op._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK)
        monkeypatch.setattr(mod, "_APPLIED", False)
        self.op, self.mod = op, mod

    def test_off_without_env(self, monkeypatch):
        monkeypatch.delenv(self.mod.ENV, raising=False)
        before = self.op._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK
        assert self.mod.apply_moe_lora_two_kernel_patch() is False
        assert self.op._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK == before

    def test_forces_two_kernel_path(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        assert self.mod.apply_moe_lora_two_kernel_patch() is True
        assert self.mod.apply_moe_lora_two_kernel_patch() is True  # idempotent
        assert self.op._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK == 0
