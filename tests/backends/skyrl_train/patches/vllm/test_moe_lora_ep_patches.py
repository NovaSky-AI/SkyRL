"""CPU tests for the DP-attention + EP MoE-LoRA patches: ``patch_moe_lora_ep_global_meta`` and
``patch_moe_lora_int64_offsets`` (env gating + the helper swap; the kernel itself needs a GPU)."""

import pytest
import torch


@pytest.mark.vllm
class TestMoeLoraEpGlobalMeta:
    @pytest.fixture(autouse=True)
    def fresh(self, monkeypatch):
        pytest.importorskip("vllm")
        meta_mod = pytest.importorskip("vllm.lora.ops.triton_ops.lora_kernel_metadata")
        from skyrl.backends.skyrl_train.patches.vllm import patch_moe_lora_ep_global_meta as mod

        monkeypatch.setattr(meta_mod.LoRAKernelMeta, "prepare_tensors", meta_mod.LoRAKernelMeta.prepare_tensors)
        monkeypatch.setattr(mod, "_APPLIED", False)
        self.meta_mod, self.mod = meta_mod, mod

    def _meta(self):
        return self.meta_mod.LoRAKernelMeta.make(max_loras=2, max_num_tokens=8, device="cpu")

    def test_off_without_env(self, monkeypatch):
        monkeypatch.delenv(self.mod.ENV, raising=False)
        assert self.mod.apply_moe_lora_ep_global_meta_patch() is False

    def test_no_lora_batch_still_writes_mapping(self, monkeypatch):
        from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

        for name in ("moe_lora_align_block_size", "add_lora_fused_moe"):
            monkeypatch.setattr(PunicaWrapperGPU, name, getattr(PunicaWrapperGPU, name))
        monkeypatch.setenv(self.mod.ENV, "1")
        assert self.mod.apply_moe_lora_ep_global_meta_patch() is True
        meta = self._meta()
        meta.prepare_tensors(torch.tensor([0, 0, 1, 1], dtype=torch.int32))
        meta.prepare_tensors(torch.full((4,), -1, dtype=torch.int32))  # all no-LoRA: vLLM skips the copy
        assert bool(meta.no_lora_flag_cpu[0])
        assert meta.token_lora_mapping[:4].tolist() == [-1, -1, -1, -1]

    def test_dispatched_mapping_uses_every_slot(self):
        meta = self._meta()
        meta.prepare_tensors(torch.full((4,), -1, dtype=torch.int32))  # this rank's own batch: no LoRA
        seen = {}

        class Wrapper:
            token_mapping_meta = meta

            @self.mod._with_global_meta
            def add_lora_fused_moe(self, x, token_lora_mapping=None):
                seen["args"] = self.token_mapping_meta.meta_args(x, False)

        w = Wrapper()
        w.add_lora_fused_moe(4, token_lora_mapping=torch.tensor([0, 1, -1, 0], dtype=torch.int32))
        _, _, _, _, lora_ids, no_lora, num_active = seen["args"]
        assert lora_ids.tolist() == [0, 1, -1]
        assert not bool(no_lora[0])
        assert num_active.tolist() == [3]
        assert w.token_mapping_meta is meta  # restored

        # Without EP (no mapping, or a view of the local buffer) the local metadata is used unchanged.
        for mapping in (None, meta.token_lora_mapping[:4]):
            w.add_lora_fused_moe(4, token_lora_mapping=mapping)
            assert bool(seen["args"][5][0])


@pytest.mark.vllm
class TestMoeLoraInt64Offsets:
    @pytest.fixture(autouse=True)
    def fresh(self, monkeypatch):
        pytest.importorskip("vllm")
        op = pytest.importorskip("vllm.lora.ops.triton_ops.fused_moe_lora_op")
        from skyrl.backends.skyrl_train.patches.vllm import patch_moe_lora_int64_offsets as mod

        monkeypatch.setattr(op, "_get_token_offs", op._get_token_offs)
        monkeypatch.setattr(mod, "_APPLIED", False)
        self.op, self.mod = op, mod

    def test_off_without_env(self, monkeypatch):
        monkeypatch.delenv(self.mod.ENV, raising=False)
        before = self.op._get_token_offs
        assert self.mod.apply_moe_lora_int64_offsets_patch() is False
        assert self.op._get_token_offs is before

    def test_swaps_helper(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        before = self.op._get_token_offs
        assert self.mod.apply_moe_lora_int64_offsets_patch() is True
        assert self.op._get_token_offs is not before
        assert "int64" in self.op._get_token_offs.src
