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


@pytest.mark.vllm
class TestCombineIndicesPatch:
    @pytest.fixture(autouse=True)
    def patched(self, monkeypatch):
        pytest.importorskip("vllm")
        cache_utils = pytest.importorskip("vllm.models.deepseek_v41.common.ops.cache_utils")
        import importlib

        from skyrl.backends.skyrl_train.patches.vllm import (
            patch_deepseek_v41_prefill_workspace as patch,
        )

        self.calls = []

        def fake_combine(*args, out=None, **kwargs):
            # Stands in for the Triton kernel: writes only the valid prefix of each row.
            self.calls.append(out)
            out[0][:, :2] = 7
            return out

        modules = []
        for name in patch._COMBINE_TOPK_SWA_INDICES_MODULES:
            try:
                modules.append(importlib.import_module(name))
            except ImportError:
                continue
        # Restored by monkeypatch after the test, together with the wrapper installed below.
        for module in modules:
            if hasattr(module, "combine_topk_swa_indices"):
                monkeypatch.setattr(module, "combine_topk_swa_indices", fake_combine)
        monkeypatch.setattr(patch, "_COMBINE_PATCHED", False)
        self.modules = modules
        self.cache_utils = cache_utils
        self.patch = patch

    def test_seeds_minus_one_before_the_kernel_writes_the_prefix(self):
        assert self.patch.apply_deepseek_v41_combine_indices_patch()
        stale = torch.full((3, 8), 123, dtype=torch.int32)
        lens = torch.zeros(3, dtype=torch.int32)
        idx, _ = self.cache_utils.combine_topk_swa_indices(out=(stale, lens))
        assert len(self.calls) == 1
        assert (idx[:, :2] == 7).all()
        # The stale tail a previous batch left is gone.
        assert (idx[:, 2:] == -1).all()

    def test_rebinds_every_importer_and_installs_once(self):
        assert self.patch.apply_deepseek_v41_combine_indices_patch()
        wrappers = {m.combine_topk_swa_indices for m in self.modules if hasattr(m, "combine_topk_swa_indices")}
        assert len(wrappers) == 1
        assert not self.patch.apply_deepseek_v41_combine_indices_patch()

    def test_mega_attn_prefill_passes_the_reused_workspace_as_out(self):
        # The patch matters because callers hand a reused workspace slice to ``out``; if vLLM stops doing that
        # (or seeds -1 itself, see vllm-project/vllm#60613) the patch can go.
        mega = pytest.importorskip("vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn")
        source = inspect.getsource(mega)
        assert "combine_topk_swa_indices(" in source
        assert "out=(idx_ws[" in source
