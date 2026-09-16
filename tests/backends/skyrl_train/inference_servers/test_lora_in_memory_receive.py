"""Receive side of the LoRA weight-sync target.

Covers the worker-extension sink (start with a LoRA ``receive_target`` ->
stage chunks -> finish hands the expanded adapter to vLLM's LoRA manager) and
the vLLM runtime patch that builds a ``LoRAModel`` from staged tensors.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers import (
    new_inference_worker_wrap as wrap,
)
from skyrl.backends.skyrl_train.patches.vllm import patch_lora_in_memory as patch
from skyrl.backends.skyrl_train.weight_sync.lora_target import (
    build_lora_receive_target,
    in_memory_lora_path,
)


def _worker():
    """A bare NewInferenceWorkerWrap with only the attributes the sink touches."""
    worker = wrap.NewInferenceWorkerWrap.__new__(wrap.NewInferenceWorkerWrap)
    worker.model_runner = SimpleNamespace(lora_manager=object(), model=SimpleNamespace())
    worker.weight_transfer_engine = object()
    worker.vllm_config = None
    worker.device = "cpu"
    return worker


@pytest.fixture(autouse=True)
def _clear_staged():
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    yield
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)


class TestWorkerSink:
    def test_start_stage_finish_hands_expanded_adapter_to_vllm(self, monkeypatch):
        worker = _worker()
        staged_calls = []
        monkeypatch.setattr(
            wrap, "stage_in_memory_adapter", lambda name, tensors, cfg: staged_calls.append((name, tensors, cfg))
        )

        target = build_lora_receive_target("tenant", {"r": 4}, {"experts.1.lora_A.weight": "experts.0.lora_A.weight"})
        worker.skyrl_start_weight_update(is_checkpoint_format=True, receive_target=target)
        assert worker._skyrl_weight_update_active and not worker._skyrl_is_checkpoint_format

        buffer = torch.arange(6, dtype=torch.float32)
        loaded = worker._skyrl_stage_lora_weights([("experts.0.lora_A.weight", buffer[:3]), ("dense", buffer[3:])])
        assert loaded == {"experts.0.lora_A.weight", "dense"}
        buffer.fill_(-1.0)  # the transport buffer is reused; staged copies must not change

        worker.skyrl_finish_weight_update()
        assert not worker._skyrl_weight_update_active and worker._skyrl_lora_target is None
        ((name, tensors, cfg),) = staged_calls
        assert name == "tenant" and cfg == {"r": 4}
        assert set(tensors) == {"experts.0.lora_A.weight", "experts.1.lora_A.weight", "dense"}
        assert torch.equal(tensors["dense"], torch.tensor([3.0, 4.0, 5.0]))
        assert tensors["experts.1.lora_A.weight"].data_ptr() == tensors["experts.0.lora_A.weight"].data_ptr()

    def test_duplicate_name_in_one_update_is_rejected(self):
        worker = _worker()
        worker.skyrl_start_weight_update(receive_target=build_lora_receive_target("t", {}, {}))
        worker._skyrl_stage_lora_weights([("a", torch.ones(1))])
        with pytest.raises(ValueError, match="received twice"):
            worker._skyrl_stage_lora_weights([("a", torch.ones(1))])

    def test_finish_without_tensors_fails(self):
        worker = _worker()
        worker.skyrl_start_weight_update(receive_target=build_lora_receive_target("t", {}, {}))
        with pytest.raises(RuntimeError, match="without receiving any tensors"):
            worker.skyrl_finish_weight_update()
        assert not worker._skyrl_weight_update_active  # state is reset even on failure

    def test_requires_lora_enabled_engine(self):
        worker = _worker()
        worker.model_runner = SimpleNamespace(lora_manager=None)
        with pytest.raises(RuntimeError, match="enable-lora"):
            worker.skyrl_start_weight_update(receive_target=build_lora_receive_target("t", {}, {}))

    def test_model_target_delegates_to_layerwise_reload(self, monkeypatch):
        worker = _worker()
        calls = []
        monkeypatch.setattr(
            wrap.LayerwiseReloadWorkerMixin,
            "skyrl_start_weight_update",
            lambda self, is_checkpoint_format=True: calls.append(("start", is_checkpoint_format)),
        )
        monkeypatch.setattr(
            wrap.LayerwiseReloadWorkerMixin, "skyrl_finish_weight_update", lambda self: calls.append(("finish",))
        )
        worker.skyrl_start_weight_update(is_checkpoint_format=False)
        worker.skyrl_finish_weight_update()
        assert calls == [("start", False), ("finish",)]

    def test_second_start_while_active_is_rejected(self):
        worker = _worker()
        worker.skyrl_start_weight_update(receive_target=build_lora_receive_target("t", {}, {}))
        with pytest.raises(RuntimeError, match="already active"):
            worker.skyrl_start_weight_update(receive_target=build_lora_receive_target("t", {}, {}))


class TestPatchRegistry:
    def test_stage_and_discard(self):
        patch.stage_in_memory_adapter("a", {"k": torch.ones(1)}, {"r": 1})
        assert patch.staged_adapter_names() == ["a"]
        assert patch.discard_in_memory_adapter("a")
        assert not patch.discard_in_memory_adapter("a")

    def test_empty_stage_rejected(self):
        with pytest.raises(ValueError):
            patch.stage_in_memory_adapter("a", {}, {})
        with pytest.raises(ValueError):
            patch.stage_in_memory_adapter("", {"k": torch.ones(1)}, {})

    def test_ordinary_paths_go_to_the_original_loader(self, monkeypatch):
        seen = []
        monkeypatch.setattr(patch, "_ORIGINAL_LOAD_ADAPTER", lambda self, req: seen.append(req) or "original")
        request = SimpleNamespace(lora_path="/tmp/adapter", lora_name="x")
        assert patch._patched_load_adapter(object(), request) == "original"
        assert seen == [request]

    def test_memory_path_without_staged_tensors_is_an_error(self):
        request = SimpleNamespace(lora_path=in_memory_lora_path("ghost"), lora_name="ghost")
        with pytest.raises(RuntimeError, match="no tensors are staged"):
            patch._patched_load_adapter(object(), request)

    def test_stage_survives_loads_until_discarded(self, monkeypatch):
        """vLLM may evict and re-add an adapter (LRU); the rebuild must not need a resend."""
        patch.stage_in_memory_adapter("t", {"k": torch.ones(1)}, {"r": 1})
        monkeypatch.setattr(patch, "_load_in_memory_adapter", lambda manager, request, staged: ("lora", staged))
        request = SimpleNamespace(lora_path=in_memory_lora_path("t"), lora_name="t")
        for _ in range(2):
            result = patch._patched_load_adapter(object(), request)
            assert result[0] == "lora" and result[1].peft_config == {"r": 1}
        assert patch.staged_adapter_names() == ["t"]
        worker = _worker()
        assert worker.skyrl_discard_in_memory_lora("t")
        assert patch.staged_adapter_names() == []

    def test_resync_replaces_the_stage(self):
        patch.stage_in_memory_adapter("t", {"k": torch.ones(1)}, {"r": 1})
        patch.stage_in_memory_adapter("t", {"k": torch.zeros(1)}, {"r": 2})
        assert patch.staged_adapter_names() == ["t"]


@pytest.mark.vllm
class TestPatchBuildsLoRAModel:
    """Drives the staged-tensor loader against real vLLM LoRA classes."""

    def test_builds_lora_model_from_gpu_or_cpu_tensors_with_ep_filter(self):
        pytest.importorskip("vllm")
        from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec

        rank, hidden, inter = 4, 16, 8
        tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, hidden),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(hidden, rank),
        }
        for expert in range(4):
            tensors[f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_A.weight"] = torch.randn(
                rank, inter
            )
            tensors[f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_B.weight"] = torch.randn(
                hidden, rank
            )
        peft_config = {
            "r": rank,
            "lora_alpha": 8,
            "target_modules": ["q_proj", "down_proj"],
            "bias": "none",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        }
        patch.stage_in_memory_adapter("t", tensors, peft_config)

        class _Manager:
            # A 2D MoE model lists each expert's projections under "experts".
            supported_lora_modules = ["qkv_proj", "experts"]
            packed_modules_mapping = {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                "experts": [f"experts.{e}.down_proj" for e in range(4)],
            }
            model = SimpleNamespace(hf_to_vllm_mapper=None, lora_skip_prefixes=None)
            # EP rank 1 of 2 with 2 local experts owns experts 2 and 3.
            moe_ep_load_spec = MoEEPLoadSpec(ep_rank=1, local_num_experts=2, global_num_experts=4)

        manager = SimpleNamespace(
            _adapter_manager=_Manager(),
            _lora_model_cls=LoRAModel,
            lora_config=SimpleNamespace(
                lora_dtype=torch.float32,
                max_lora_rank=rank,
                lora_extra_vocab_size=0,
                fully_sharded_loras=False,
            ),
            device="cpu",
            vocab_size=32000,
        )
        request = SimpleNamespace(
            lora_path=in_memory_lora_path("t"), lora_name="t", lora_int_id=7, is_3d_lora_weight=False
        )
        lora = patch._patched_load_adapter(manager, request)

        assert lora.id == 7 and lora.rank == rank
        modules = set(lora.loras)  # vLLM strips the ``base_model.model.`` prefix
        assert "model.layers.0.self_attn.q_proj" in modules
        assert any(".experts.2." in m for m in modules) and any(".experts.3." in m for m in modules)
        assert not any(".experts.0." in m or ".experts.1." in m for m in modules)
        # On CPU vLLM pins a copy per key; on the GPU (the deployed case) the
        # per-key ``.to()`` is a no-op and the LoRAModel shares staged storage.
        q_a = lora.loras["model.layers.0.self_attn.q_proj"].lora_a
        assert torch.equal(q_a, tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"])
        assert patch.staged_adapter_names() == ["t"]
