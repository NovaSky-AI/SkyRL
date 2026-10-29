"""In-memory LoRA adapters move to pinned host memory once vLLM has activated them.

``patch_lora_in_memory`` builds an in-memory adapter on the GPU, and the patched
``add_adapter`` then moves the registered ``LoRAModel`` (including vLLM's
``pack_moe`` stack of every local expert) and the staged tensors to pinned host
memory. These tests drive vLLM's real ``LoRAModel`` and
``PackedLoRALayerWeights`` through that path, with a stand-in for vLLM's
adapter manager, and check that nothing stays on the GPU, values are
unchanged, aliased tensors share one host copy, and a rebuild from the
offloaded stage happens on the host.

Run with:

    uv run --isolated --extra dev --extra fsdp pytest -s \
      tests/backends/skyrl_train/gpu/gpu_ci/patches/vllm/test_lora_in_memory_offload.py
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.patches.vllm import patch_lora_in_memory as patch
from skyrl.backends.skyrl_train.weight_sync.lora_target import in_memory_lora_path

pytest.importorskip("vllm", reason="drives vLLM's LoRAModel and PackedLoRALayerWeights directly")

from vllm.lora.lora_model import LoRAModel  # noqa: E402
from vllm.lora.lora_weights import PackedLoRALayerWeights  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="moves CUDA tensors to the host")

RANK, HIDDEN, INTER, NUM_EXPERTS = 4, 16, 8, 4
LORA_ID = 7
PREFIX = "base_model.model.model.layers.0"
EXPERTS = "model.layers.0.mlp.experts"


@pytest.fixture(autouse=True)
def _clear_staged():
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    patch._AWAITING_OFFLOAD.clear()
    yield
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    patch._AWAITING_OFFLOAD.clear()


def _staged_tensors() -> dict[str, torch.Tensor]:
    """A dense q_proj adapter and a non-gated 4-expert MoE adapter whose down_proj lora_A
    is one tensor aliased across experts, as ``expand_lora_aliases`` produces."""
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device="cuda", dtype=torch.bfloat16)

    t = {
        f"{PREFIX}.self_attn.q_proj.lora_A.weight": rand(RANK, HIDDEN),
        f"{PREFIX}.self_attn.q_proj.lora_B.weight": rand(HIDDEN, RANK),
    }
    shared_down_a = rand(RANK, INTER)
    for e in range(NUM_EXPERTS):
        t[f"{PREFIX}.mlp.experts.{e}.gate_proj.lora_A.weight"] = rand(RANK, HIDDEN)
        t[f"{PREFIX}.mlp.experts.{e}.gate_proj.lora_B.weight"] = rand(INTER, RANK)
        t[f"{PREFIX}.mlp.experts.{e}.down_proj.lora_A.weight"] = shared_down_a
        t[f"{PREFIX}.mlp.experts.{e}.down_proj.lora_B.weight"] = rand(HIDDEN, RANK)
    return t


def _manager():
    registered: dict[int, LoRAModel] = {}

    class _AdapterManager:
        supported_lora_modules = ["qkv_proj", "experts"]
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "experts": [f"experts.{e}.{p}" for e in range(NUM_EXPERTS) for p in ("gate_proj", "down_proj")],
        }
        model = SimpleNamespace(hf_to_vllm_mapper=None, lora_skip_prefixes=None)
        moe_ep_load_spec = None

        def get_adapter(self, lora_id):
            return registered.get(lora_id)

    return SimpleNamespace(
        _adapter_manager=_AdapterManager(),
        _lora_model_cls=LoRAModel,
        lora_config=SimpleNamespace(
            lora_dtype=torch.bfloat16, max_lora_rank=RANK, lora_extra_vocab_size=0, fully_sharded_loras=False
        ),
        device=torch.device("cuda"),
        vocab_size=32000,
        registered=registered,
    )


def _request():
    return SimpleNamespace(
        lora_path=in_memory_lora_path("t"), lora_name="t", lora_int_id=LORA_ID, is_3d_lora_weight=False
    )


def _vllm_add_adapter(manager, lora_request):
    """Stands in for vLLM's ``add_adapter``: load, pack the experts as
    ``LoRAModelManager._create_merged_loras_inplace`` does, and register."""
    lora = patch._patched_load_adapter(manager, lora_request)
    expert_loras = []
    for e in range(NUM_EXPERTS):
        expert_loras += [
            lora.loras.pop(f"{EXPERTS}.{e}.gate_proj"),
            lora.loras.pop(f"{EXPERTS}.{e}.down_proj"),
            None,
        ]
    lora.loras[EXPERTS] = PackedLoRALayerWeights.pack_moe(expert_loras, EXPERTS, is_non_gated_moe=True)
    manager.registered[lora_request.lora_int_id] = lora
    return True


def _sync(manager) -> dict[str, torch.Tensor]:
    """Stages and activates the adapter; returns host copies of the packed values to compare against."""
    peft_config = {
        "r": RANK,
        "lora_alpha": RANK,
        "target_modules": ["q_proj", "gate_proj", "down_proj"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    patch.stage_in_memory_adapter("t", _staged_tensors(), peft_config)
    # Capture the expected values from inside the build, before the offload runs.
    expected: dict[str, torch.Tensor] = {}

    def add_and_capture(manager, lora_request):
        loaded = _vllm_add_adapter(manager, lora_request)
        packed = manager.registered[lora_request.lora_int_id].loras[EXPERTS]
        expected["w1_a"] = packed.lora_a[0].cpu()
        expected["w2_a"] = packed.lora_a[1].cpu()
        expected["w2_b"] = packed.lora_b[1].cpu()
        return loaded

    add_adapter = patch._make_patched_add_adapter(add_and_capture)
    assert add_adapter(manager, _request()) is True
    return expected


def test_offload_moves_built_adapter_and_stage_to_pinned_host():
    # Frees CUDA tensors left unreachable by earlier tests in this process first.
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    manager = _manager()

    expected = _sync(manager)

    lora = manager.registered[LORA_ID]
    staged = patch._STAGED["t"].tensors
    host = list(staged.values())
    for weights in lora.loras.values():
        for value in (weights.lora_a, weights.lora_b):
            host += value if isinstance(value, list) else [value]
    assert all(t.device.type == "cpu" and t.is_pinned() for t in host)
    assert torch.cuda.memory_allocated() == baseline

    packed = lora.loras[EXPERTS]
    for name, value in (("w1_a", packed.lora_a[0]), ("w2_a", packed.lora_a[1]), ("w2_b", packed.lora_b[1])):
        assert torch.equal(value, expected[name]), name
    # Tensors that shared GPU storage share one host copy.
    down_a = {staged[f"{PREFIX}.mlp.experts.{e}.down_proj.lora_A.weight"].data_ptr() for e in range(NUM_EXPERTS)}
    assert len(down_a) == 1


def test_rebuild_from_offloaded_stage_stays_on_host():
    manager = _manager()
    _sync(manager)
    staged = patch._STAGED["t"].tensors

    lora = patch._patched_load_adapter(manager, _request())

    down_a = [lora.get_lora(f"{EXPERTS}.{e}.down_proj").lora_a for e in range(NUM_EXPERTS)]
    assert all(t.device.type == "cpu" and t.is_pinned() for t in down_a)
    # An already pinned tensor is not copied again, so the aliases stay shared.
    assert {t.data_ptr() for t in down_a} == {staged[f"{PREFIX}.mlp.experts.0.down_proj.lora_A.weight"].data_ptr()}
    assert not patch._AWAITING_OFFLOAD


def test_views_with_different_strides_get_their_own_host_copy():
    square = torch.arange(16, dtype=torch.bfloat16, device="cuda").reshape(4, 4)
    moved: dict = {}

    host = patch._to_host(square, moved)
    host_t = patch._to_host(square.t(), moved)
    torch.cuda.synchronize()

    assert torch.equal(host, square.cpu())
    assert torch.equal(host_t, square.t().cpu())
    assert patch._to_host(square, moved) is host
