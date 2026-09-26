"""The in-memory LoRA loader against vLLM's real LoRA classes.

The staged tensors are handed to ``LoRAModel.from_lora_tensors`` by reference
and kept for LRU rebuilds, while vLLM scales ``lora_b`` in place at load. The
trainer therefore publishes ``lora_alpha == r`` with the scale folded into
``lora_B``; this test pins that the loader refuses anything else.

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


def test_unfolded_alpha_is_refused_before_anything_is_built():
    tensors = _tensors()
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=2 * RANK))
    with pytest.raises(ValueError, match="lora_alpha=8 != r=4"):
        patch._patched_load_adapter(_manager(), _request("t"))
    # Refusal leaves the stage in place for a corrected resync.
    assert patch.staged_adapter_names() == ["t"]
