"""CPU unit tests for the LoRA rank-scale fold applied on the vLLM sync path.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/distributed/test_lora_export_rank_scale.py

Background: with ``normalize_moe_lora`` the trainer runs expert adapters at
``rank // moe_router_topk`` and scales them by ``alpha / effective_rank``; vLLM
scales every module by the single ``lora_alpha / r`` in ``adapter_config.json``.
These tests pin the invariant that makes the two agree:

    (alpha / config_rank) * folded_B @ A == (alpha / effective_rank) * B @ A
"""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.lora_export import (
    fold_lora_rank_scale_for_vllm,
    lora_rank_from_tensors,
)

ALPHA = 32
CONFIG_RANK = 32
EXPERT_RANK = 4  # 32 // topk 8, as in the Kimi-K2.7 recipe


def _trainer_delta(lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
    """What megatron-bridge applies: alpha / dim with the module's own rank."""
    rank = lora_b.shape[-1]
    return (ALPHA / rank) * (lora_b.float() @ lora_a.float())


def _vllm_delta(lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
    """What vLLM applies: alpha / r from adapter_config.json, for every module."""
    return (ALPHA / CONFIG_RANK) * (lora_b.float() @ lora_a.float())


def _state_per_expert_layout(dtype=torch.bfloat16, n_experts=3):
    g = torch.Generator().manual_seed(0)
    state = {}
    prefix = "base_model.model.language_model.model.layers.5"
    # dense modules at the full rank
    state[f"{prefix}.self_attn.o_proj.lora_A.weight"] = torch.randn(CONFIG_RANK, 64, generator=g).to(dtype)
    state[f"{prefix}.self_attn.o_proj.lora_B.weight"] = torch.randn(48, CONFIG_RANK, generator=g).to(dtype) * 1e-3
    state[f"{prefix}.mlp.shared_experts.down_proj.lora_A.weight"] = torch.randn(CONFIG_RANK, 16, generator=g).to(dtype)
    state[f"{prefix}.mlp.shared_experts.down_proj.lora_B.weight"] = (
        torch.randn(64, CONFIG_RANK, generator=g).to(dtype) * 1e-3
    )
    # routed experts at the reduced rank (per-expert 2D keys, as DeepSeek/Kimi decoders export)
    for e in range(n_experts):
        for proj, (out, inp) in (("gate_proj", (24, 64)), ("up_proj", (24, 64)), ("down_proj", (64, 24))):
            state[f"{prefix}.mlp.experts.{e}.{proj}.lora_A.weight"] = torch.randn(EXPERT_RANK, inp, generator=g).to(
                dtype
            )
            state[f"{prefix}.mlp.experts.{e}.{proj}.lora_B.weight"] = (
                torch.randn(out, EXPERT_RANK, generator=g).to(dtype) * 1e-3
            )
    return state


def test_expert_lora_b_is_rescaled_so_vllm_matches_trainer():
    state = _state_per_expert_layout()
    folded, rescaled = fold_lora_rank_scale_for_vllm(state, config_rank=CONFIG_RANK)

    assert list(folded.keys()) == list(state.keys()), "keys and order must be preserved"
    assert rescaled == {EXPERT_RANK: 9}, "3 experts x 3 projections at rank 4"

    for key, tensor in state.items():
        if not key.endswith(".lora_B.weight"):
            # lora_A is never touched
            assert torch.equal(folded[key], tensor), key
            continue
        a_key = key.replace(".lora_B.weight", ".lora_A.weight")
        expected = _trainer_delta(state[a_key], tensor)
        got = _vllm_delta(folded[a_key], folded[key])
        # x8 only shifts exponents, so the two products agree bit-for-bit.
        torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=key)
        assert folded[key].dtype == tensor.dtype


def test_full_rank_modules_pass_through_untouched():
    state = _state_per_expert_layout(n_experts=0)
    folded, rescaled = fold_lora_rank_scale_for_vllm(state, config_rank=CONFIG_RANK)
    assert rescaled == {}
    for key, tensor in state.items():
        assert torch.equal(folded[key], tensor), key


def test_power_of_two_ratio_is_exact_in_bf16():
    state = _state_per_expert_layout(dtype=torch.bfloat16, n_experts=1)
    folded, _ = fold_lora_rank_scale_for_vllm(state, config_rank=CONFIG_RANK)
    key = "base_model.model.language_model.model.layers.5.mlp.experts.0.down_proj.lora_B.weight"
    ratio = CONFIG_RANK / EXPERT_RANK
    assert ratio == 8.0
    # x8 is a pure exponent shift: bf16(x) * 8 == bf16(8 * x) bit-for-bit.
    assert torch.equal(folded[key].float(), state[key].float() * ratio)


def test_fused_3d_expert_layout_is_rescaled_before_flattening():
    g = torch.Generator().manual_seed(1)
    n_experts, out, inp = 4, 12, 16
    prefix = "base_model.model.model.layers.2.mlp.experts"
    state = {
        f"{prefix}.gate_up_proj.lora_A.weight": torch.randn(n_experts, EXPERT_RANK, inp, generator=g).to(
            torch.bfloat16
        ),
        f"{prefix}.gate_up_proj.lora_B.weight": torch.randn(n_experts, 2 * out, EXPERT_RANK, generator=g).to(
            torch.bfloat16
        ),
        f"{prefix}.down_proj.lora_A.weight": torch.randn(n_experts, EXPERT_RANK, out, generator=g).to(torch.bfloat16),
        f"{prefix}.down_proj.lora_B.weight": torch.randn(n_experts, inp, EXPERT_RANK, generator=g).to(torch.bfloat16),
    }
    folded, rescaled = fold_lora_rank_scale_for_vllm(state, config_rank=CONFIG_RANK)
    assert rescaled == {EXPERT_RANK: 2}
    for proj in ("gate_up_proj", "down_proj"):
        a = state[f"{prefix}.{proj}.lora_A.weight"]
        b = state[f"{prefix}.{proj}.lora_B.weight"]
        fb = folded[f"{prefix}.{proj}.lora_B.weight"]
        assert fb.shape == b.shape, "shape is preserved; flattening is a later, separate step"
        for e in range(n_experts):
            torch.testing.assert_close(_vllm_delta(a[e], fb[e]), _trainer_delta(a[e], b[e]), rtol=0, atol=0)


def test_lora_rank_from_tensors_reads_both_layouts_and_rejects_mismatch():
    assert lora_rank_from_tensors(torch.zeros(4, 16), torch.zeros(8, 4)) == 4
    assert lora_rank_from_tensors(None, torch.zeros(8, 32)) == 32
    assert lora_rank_from_tensors(torch.zeros(3, 4, 16), torch.zeros(3, 8, 4)) == 4
    with pytest.raises(ValueError, match="rank"):
        lora_rank_from_tensors(torch.zeros(8, 16), torch.zeros(8, 4))
    with pytest.raises(ValueError, match="lora_B must be"):
        lora_rank_from_tensors(None, torch.zeros(4))


def test_invalid_config_rank_rejected():
    with pytest.raises(ValueError, match="config_rank"):
        fold_lora_rank_scale_for_vllm({}, config_rank=0)
