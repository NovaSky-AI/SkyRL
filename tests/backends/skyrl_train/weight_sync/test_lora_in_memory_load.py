"""The in-memory LoRA loader against vLLM's real LoRA classes.

The staged tensors are handed to ``LoRAModel.from_lora_tensors`` by reference
and kept for LRU rebuilds, while vLLM scales ``lora_b`` in place at load. These
tests pin the contract that keeps that safe: the trainer publishes
``lora_alpha == r`` with the scale folded into ``lora_B``, the loader refuses
anything else, and loading the same stage twice leaves it byte-identical.

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
    yield
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)


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


def _manager():
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
        device="cpu",
        vocab_size=32000,
    )


def _request(name: str):
    return SimpleNamespace(lora_path=in_memory_lora_path(name), lora_name=name, lora_int_id=7, is_3d_lora_weight=False)


def _optimize_all(lora: LoRAModel) -> None:
    """What LoRAModelManager._create_merged_loras_inplace does to every loaded adapter."""
    for layer in lora.loras.values():
        layer.optimize()


def test_folded_adapter_loads_twice_without_touching_the_stage():
    tensors = _tensors()
    before = {k: v.clone() for k, v in tensors.items()}
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=RANK))
    manager = _manager()

    first = patch._patched_load_adapter(manager, _request("t"))
    _optimize_all(first)  # vLLM's in-place scale step; a no-op at lora_alpha == r
    second = patch._patched_load_adapter(manager, _request("t"))
    _optimize_all(second)

    assert first.id == 7 and first.rank == RANK
    modules = set(first.loras)  # vLLM strips the ``base_model.model.`` prefix
    assert "model.layers.0.self_attn.q_proj" in modules
    assert any(".experts.2." in m for m in modules) and any(".experts.3." in m for m in modules)
    assert not any(".experts.0." in m or ".experts.1." in m for m in modules)

    # The stage is what LRU rebuilds read; it must be exactly what the trainer sent.
    for key, tensor in before.items():
        assert torch.equal(tensors[key], tensor), key
    # And both loads see the same, unscaled values.
    q = "model.layers.0.self_attn.q_proj"
    assert torch.equal(first.loras[q].lora_b, second.loras[q].lora_b)
    assert torch.equal(first.loras[q].lora_b, before["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"])
    assert patch.staged_adapter_names() == ["t"]


def test_unfolded_alpha_is_refused_before_anything_is_built():
    tensors = _tensors()
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=2 * RANK))
    with pytest.raises(ValueError, match="lora_alpha=8 != r=4"):
        patch._patched_load_adapter(_manager(), _request("t"))
    # Refusal leaves the stage in place for a corrected resync.
    assert patch.staged_adapter_names() == ["t"]


def test_vllm_scales_lora_b_in_place_which_is_why_the_guard_exists():
    """Documents the hazard the guard prevents: with lora_alpha != r, vLLM's
    optimize() multiplies the tensor it was handed, i.e. the stage itself."""
    tensors = _tensors()
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    original = tensors[key].clone()
    from vllm.lora.peft_helper import PEFTHelper

    lora = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors=tensors,
        peft_helper=PEFTHelper.from_dict(_peft_config(lora_alpha=2 * RANK)),
        device="cpu",
        dtype=torch.float32,
        model_vocab_size=32000,
    )
    q = "model.layers.0.self_attn.q_proj"
    if lora.loras[q].lora_b.data_ptr() != tensors[key].data_ptr():
        # On hosts where vLLM pins CPU tensors, from_lora_tensors copies and the
        # scale lands on the copy. The deployed case is GPU-resident staging,
        # where no copy is made and the hazard below is real.
        pytest.skip("from_lora_tensors copied the tensor on this host (pinned memory); sharing not exercised")
    _optimize_all(lora)
    assert torch.equal(tensors[key], original * 2.0), "vLLM's scale landed on the staged tensor"
