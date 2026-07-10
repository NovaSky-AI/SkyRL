"""Tests for fused-MoE expert LoRA layout conversion to vLLM's flat PEFT format.

Requires megatron-core (imported transitively by megatron_utils); skipped when
it is not installed.

Run with:
uv run --extra dev --extra megatron pytest tests/backends/skyrl_train/distributed/test_moe_lora_conversion.py
"""

import importlib.util

import pytest
import torch

_has_megatron = importlib.util.find_spec("megatron") is not None

if _has_megatron:
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        _convert_moe_experts_lora_to_vllm,
    )

NUM_EXPERTS = 4
RANK = 2
HIDDEN = 6
INTERMEDIATE = 3

GATE_UP_A = "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight"
GATE_UP_B = "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_B.weight"
DOWN_A = "base_model.model.model.layers.0.mlp.experts.down_proj.lora_A.weight"
DOWN_B = "base_model.model.model.layers.0.mlp.experts.down_proj.lora_B.weight"
CONVERTED_GATE_UP_A = "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight"
CONVERTED_GATE_UP_B = "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"
CONVERTED_DOWN_A = "base_model.model.model.layers.0.mlp.experts.lora_A.weight"
CONVERTED_DOWN_B = "base_model.model.model.layers.0.mlp.experts.lora_B.weight"


def _vllm_per_expert_view_a(flat_a: torch.Tensor, num_experts: int) -> torch.Tensor:
    """vLLM's `_stack_moe_lora_weights` lora_A reshape: (rank*E, in) -> (E, rank, in)."""
    return flat_a.reshape(num_experts, -1, flat_a.shape[-1])


def _vllm_per_expert_view_b(flat_b: torch.Tensor, num_experts: int) -> torch.Tensor:
    """vLLM's `_stack_moe_lora_weights` lora_B reshape: (out, rank*E) -> (E, out, rank)."""
    return flat_b.reshape(flat_b.shape[0], -1, num_experts).permute(2, 0, 1)


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestPackedExpertConversion:
    """Packed-HF MoE models (Qwen3.5/3.6 style): one 3D tensor per fused projection."""

    def test_per_expert_3d_round_trip(self):
        """(E, rank, in) / (E, out, rank) exports must survive the vLLM reshape inverse."""
        gate_up_a = torch.randn(NUM_EXPERTS, RANK, HIDDEN)
        gate_up_b = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, RANK)
        down_a = torch.randn(NUM_EXPERTS, RANK, INTERMEDIATE)
        down_b = torch.randn(NUM_EXPERTS, HIDDEN, RANK)

        converted = _convert_moe_experts_lora_to_vllm(
            {GATE_UP_A: gate_up_a, GATE_UP_B: gate_up_b, DOWN_A: down_a, DOWN_B: down_b},
            num_moe_experts=NUM_EXPERTS,
        )

        assert converted[CONVERTED_GATE_UP_A].shape == (RANK * NUM_EXPERTS, HIDDEN)
        assert converted[CONVERTED_GATE_UP_B].shape == (2 * INTERMEDIATE, RANK * NUM_EXPERTS)
        torch.testing.assert_close(_vllm_per_expert_view_a(converted[CONVERTED_GATE_UP_A], NUM_EXPERTS), gate_up_a)
        torch.testing.assert_close(_vllm_per_expert_view_b(converted[CONVERTED_GATE_UP_B], NUM_EXPERTS), gate_up_b)
        torch.testing.assert_close(_vllm_per_expert_view_a(converted[CONVERTED_DOWN_A], NUM_EXPERTS), down_a)
        torch.testing.assert_close(_vllm_per_expert_view_b(converted[CONVERTED_DOWN_B], NUM_EXPERTS), down_b)

    def test_shared_outer_sides_expanded_to_all_experts(self):
        """Shared-outer exports carry a (1, ...) shared side (gate_up lora_A, down
        lora_B); it must be expanded so every expert sees the same matrix after
        vLLM's per-expert reshape."""
        shared_gate_up_a = torch.randn(1, RANK, HIDDEN)
        per_expert_gate_up_b = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, RANK)
        per_expert_down_a = torch.randn(NUM_EXPERTS, RANK, INTERMEDIATE)
        shared_down_b = torch.randn(1, HIDDEN, RANK)

        converted = _convert_moe_experts_lora_to_vllm(
            {
                GATE_UP_A: shared_gate_up_a,
                GATE_UP_B: per_expert_gate_up_b,
                DOWN_A: per_expert_down_a,
                DOWN_B: shared_down_b,
            },
            num_moe_experts=NUM_EXPERTS,
        )

        expanded_a = _vllm_per_expert_view_a(converted[CONVERTED_GATE_UP_A], NUM_EXPERTS)
        assert expanded_a.shape == (NUM_EXPERTS, RANK, HIDDEN)
        for expert_idx in range(NUM_EXPERTS):
            torch.testing.assert_close(expanded_a[expert_idx], shared_gate_up_a[0])

        expanded_b = _vllm_per_expert_view_b(converted[CONVERTED_DOWN_B], NUM_EXPERTS)
        assert expanded_b.shape == (NUM_EXPERTS, HIDDEN, RANK)
        for expert_idx in range(NUM_EXPERTS):
            torch.testing.assert_close(expanded_b[expert_idx], shared_down_b[0])

        torch.testing.assert_close(
            _vllm_per_expert_view_b(converted[CONVERTED_GATE_UP_B], NUM_EXPERTS), per_expert_gate_up_b
        )
        torch.testing.assert_close(_vllm_per_expert_view_a(converted[CONVERTED_DOWN_A], NUM_EXPERTS), per_expert_down_a)

    def test_non_expert_tensors_pass_through(self):
        q_proj_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        state = {q_proj_a: torch.randn(RANK, HIDDEN)}
        converted = _convert_moe_experts_lora_to_vllm(state, num_moe_experts=NUM_EXPERTS)
        assert converted.keys() == state.keys()
        torch.testing.assert_close(converted[q_proj_a], state[q_proj_a])


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestPerExpertHFFormatSharedOuter:
    """Per-expert-HF MoE models (Qwen3 MoE style): shared-outer's expert-agnostic
    shared side must be replicated into the indexed keys vLLM's PEFT loader parses."""

    def _indexed(self, proj: str, expert_idx: int, side: str) -> str:
        return f"base_model.model.model.layers.0.mlp.experts.{expert_idx}.{proj}.lora_{side}.weight"

    def _shared(self, proj: str, side: str) -> str:
        return f"base_model.model.model.layers.0.mlp.experts.{proj}.lora_{side}.weight"

    def test_shared_sides_replicated_into_indexed_keys(self):
        shared_gate_a = torch.randn(1, RANK, HIDDEN)
        shared_down_b = torch.randn(1, HIDDEN, RANK)
        state = {
            self._shared("gate_proj", "A"): shared_gate_a,
            self._shared("up_proj", "A"): shared_gate_a.clone(),
            self._shared("down_proj", "B"): shared_down_b,
        }
        for expert_idx in range(NUM_EXPERTS):
            state[self._indexed("gate_proj", expert_idx, "B")] = torch.randn(INTERMEDIATE, RANK)
            state[self._indexed("up_proj", expert_idx, "B")] = torch.randn(INTERMEDIATE, RANK)
            state[self._indexed("down_proj", expert_idx, "A")] = torch.randn(RANK, INTERMEDIATE)

        converted = _convert_moe_experts_lora_to_vllm(state, num_moe_experts=NUM_EXPERTS)

        for expert_idx in range(NUM_EXPERTS):
            for proj, side, shared in (
                ("gate_proj", "A", shared_gate_a),
                ("up_proj", "A", shared_gate_a),
                ("down_proj", "B", shared_down_b),
            ):
                key = self._indexed(proj, expert_idx, side)
                assert key in converted, key
                torch.testing.assert_close(converted[key], shared[0])
            # Existing per-expert tensors are untouched.
            torch.testing.assert_close(
                converted[self._indexed("down_proj", expert_idx, "A")],
                state[self._indexed("down_proj", expert_idx, "A")],
            )
        # Expert-agnostic shared keys are consumed by the replication.
        assert self._shared("gate_proj", "A") not in converted
        assert self._shared("down_proj", "B") not in converted
