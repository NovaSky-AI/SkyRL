"""CPU tests for ``patch_deepseek_v41_prefill_workspace``.

A fake workspace manager stands in for vLLM's, so the patch runs without a GPU or a model.
"""

import inspect

import pytest
import torch


class _FakeWorkspaceManager:
    def __init__(self):
        self.calls = 0

    def get_simultaneous(self, *specs):
        self.calls += 1
        return [torch.full(shape, float("nan"), dtype=dtype) for shape, dtype in specs]

    def other(self):
        return "passed through"


@pytest.mark.vllm
class TestPrefillWorkspacePatch:
    @pytest.fixture(autouse=True)
    def patched(self, monkeypatch):
        pytest.importorskip("vllm")
        flashmla = pytest.importorskip("vllm.models.deepseek_v41.nvidia.flashmla")

        from skyrl.backends.skyrl_train.patches.vllm import (
            patch_deepseek_v41_prefill_workspace,
        )

        self.manager = _FakeWorkspaceManager()
        # Restored by monkeypatch after the test, together with the wrapper installed below.
        monkeypatch.setattr(flashmla, "current_workspace_manager", lambda: self.manager)
        monkeypatch.setattr(patch_deepseek_v41_prefill_workspace, "_PATCHED", False)
        self.flashmla = flashmla
        self.patch = patch_deepseek_v41_prefill_workspace

    def test_zeroes_only_the_gathered_kv_buffer(self):
        assert self.patch.apply_deepseek_v41_prefill_workspace_patch()
        kv, other = self.flashmla.current_workspace_manager().get_simultaneous(
            ((4, 8), torch.bfloat16), ((3,), torch.float32)
        )
        assert self.manager.calls == 1
        assert torch.equal(kv, torch.zeros(4, 8, dtype=torch.bfloat16))
        # Only the gather buffer is read before being fully written; the rest are left alone.
        assert torch.isnan(other).all()

    def test_forwards_other_attributes_and_installs_once(self):
        assert self.patch.apply_deepseek_v41_prefill_workspace_patch()
        assert self.flashmla.current_workspace_manager().other() == "passed through"
        assert not self.patch.apply_deepseek_v41_prefill_workspace_patch()

    def test_prefill_looks_the_manager_up_through_the_patched_name(self):
        # The patch replaces the module-level name and zeroes the first buffer: it is a no-op if prefill stops
        # looking the manager up there, and wrong if the gathered KV stops being the first buffer.
        source = inspect.getsource(self.flashmla.DeepseekV4FlashMLAAttention._forward_prefill)
        assert "= current_workspace_manager()" in source
        assert "kv, combined_indices_out, combined_lens_out = workspace" in source
