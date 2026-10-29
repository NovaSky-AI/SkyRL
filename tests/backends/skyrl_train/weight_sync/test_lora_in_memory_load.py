"""The in-memory LoRA loader against vLLM's real LoRA classes.

The staged tensors are handed to ``LoRAModel.from_lora_tensors`` by reference
and kept for LRU rebuilds, while vLLM scales ``lora_b`` in place at load. The
trainer therefore publishes ``lora_alpha == r`` with the scale folded into
``lora_B``; this file tests that the loader refuses anything else.

It also tests when the patched ``add_adapter`` moves an adapter to the host:
once after an adapter is built on an accelerator, never for a disk adapter or a
repeat ``add_adapter`` call. A
``meta``-device stage stands in for GPU tensors; the copy itself is tested on
GPU in ``tests/backends/skyrl_train/gpu/gpu_ci/patches/vllm/test_lora_in_memory_offload.py``.

Run with: uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/test_lora_in_memory_load.py
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.patches.vllm import patch_lora_in_memory as patch
from skyrl.backends.skyrl_train.weight_sync.lora_target import in_memory_lora_path

pytest.importorskip("vllm", reason="drives vLLM's LoRAModel and LoRALayerWeights directly")
pytestmark = pytest.mark.vllm

from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec  # noqa: E402

RANK, HIDDEN, INTER = 4, 16, 8


@pytest.fixture(autouse=True)
def _clear_staged():
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    patch._AWAITING_OFFLOAD.clear()
    yield
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    patch._AWAITING_OFFLOAD.clear()


def _tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    t = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(RANK, HIDDEN, generator=g),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(HIDDEN, RANK, generator=g),
    }
    for expert in range(4):
        t[f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_A.weight"] = torch.randn(
            RANK, INTER, generator=g
        )
        t[f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_B.weight"] = torch.randn(
            HIDDEN, RANK, generator=g
        )
    return t


def _peft_config(lora_alpha: int) -> dict:
    return {
        "r": RANK,
        "lora_alpha": lora_alpha,
        "target_modules": ["q_proj", "down_proj"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }


def _manager(device: str = "cpu"):
    class _AdapterManager:
        # A 2D MoE model lists each expert's projections under "experts".
        supported_lora_modules = ["qkv_proj", "experts"]
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "experts": [f"experts.{e}.down_proj" for e in range(4)],
        }
        model = SimpleNamespace(hf_to_vllm_mapper=None, lora_skip_prefixes=None)
        # EP rank 1 of 2 with 2 local experts owns experts 2 and 3.
        moe_ep_load_spec = MoEEPLoadSpec(ep_rank=1, local_num_experts=2, global_num_experts=4)

    return SimpleNamespace(
        _adapter_manager=_AdapterManager(),
        _lora_model_cls=LoRAModel,
        lora_config=SimpleNamespace(
            lora_dtype=torch.float32, max_lora_rank=RANK, lora_extra_vocab_size=0, fully_sharded_loras=False
        ),
        device=device,
        vocab_size=32000,
    )


def _request(name: str, lora_int_id: int = 7):
    return SimpleNamespace(
        lora_path=in_memory_lora_path(name), lora_name=name, lora_int_id=lora_int_id, is_3d_lora_weight=False
    )


def _add_adapter_counting_offloads(monkeypatch):
    """Returns the patched ``add_adapter`` over a stand-in for vLLM's, and the offloaded ids.

    The stand-in builds an adapter only when its id is not registered yet, as
    ``WorkerLoRAManager.add_adapter`` does.
    """
    offloaded: list[int] = []
    monkeypatch.setattr(patch, "offload_in_memory_adapter", lambda manager, req: offloaded.append(req.lora_int_id))
    registered: set[int] = set()

    def original_add_adapter(manager, lora_request):
        if lora_request.lora_int_id in registered:
            return False
        patch._patched_load_adapter(manager, lora_request)
        registered.add(lora_request.lora_int_id)
        return True

    return patch._make_patched_add_adapter(original_add_adapter), offloaded


def test_unfolded_alpha_is_refused_before_anything_is_built():
    tensors = _tensors()
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=2 * RANK))
    with pytest.raises(ValueError, match="lora_alpha=8 != r=4"):
        patch._patched_load_adapter(_manager(), _request("t"))
    # Refusal leaves the stage in place for a corrected resync.
    assert patch.staged_adapter_names() == ["t"]


def test_device_built_adapter_is_offloaded_once(monkeypatch):
    add_adapter, offloaded = _add_adapter_counting_offloads(monkeypatch)
    patch.stage_in_memory_adapter("t", {k: v.to("meta") for k, v in _tensors().items()}, _peft_config(RANK))
    manager = _manager(device="meta")

    assert add_adapter(manager, _request("t")) is True
    # vLLM calls add_adapter for every scheduled batch; an activated adapter is not offloaded again.
    assert add_adapter(manager, _request("t")) is False
    assert offloaded == [7]
    assert not patch._AWAITING_OFFLOAD


def test_disk_adapter_is_not_offloaded(monkeypatch):
    add_adapter, offloaded = _add_adapter_counting_offloads(monkeypatch)
    disk_loads = []
    monkeypatch.setattr(patch, "_ORIGINAL_LOAD_ADAPTER", lambda manager, req: disk_loads.append(req.lora_int_id))
    disk_request = SimpleNamespace(lora_path="/tmp/adapter", lora_name="d", lora_int_id=3, is_3d_lora_weight=False)

    assert add_adapter(_manager(device="meta"), disk_request) is True
    assert disk_loads == [3]
    assert offloaded == []
    assert not patch._AWAITING_OFFLOAD


def test_patch_wraps_add_adapter_of_both_worker_managers(monkeypatch):
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager

    # Installs on fresh patch state; monkeypatch restores vLLM's classes afterwards.
    monkeypatch.setattr(WorkerLoRAManager, "_load_adapter", WorkerLoRAManager.__dict__["_load_adapter"])
    for cls in (WorkerLoRAManager, LRUCacheWorkerLoRAManager):
        monkeypatch.setattr(cls, "add_adapter", cls.__dict__["add_adapter"])
    monkeypatch.setattr(patch, "_PATCHED", False)
    monkeypatch.setattr(patch, "_ORIGINAL_LOAD_ADAPTER", None)
    monkeypatch.setattr(patch, "_ORIGINAL_ADD_ADAPTER", {})

    patch.apply_lora_in_memory_patch()

    # LRUCacheWorkerLoRAManager overrides add_adapter without calling super(), so both are wrapped.
    for cls in (WorkerLoRAManager, LRUCacheWorkerLoRAManager):
        assert cls.__dict__["add_adapter"].__name__ == "_patched_add_adapter"
