"""Shared test utilities for LoRA training tests."""

import jax
import jax.numpy as jnp

from tx.utils.models import is_stacked_lora_path


def get_adapter_params(params, adapter_idx: int):
    """Extract adapter params at a specific index.

    Decoder layer LoRA params have shape (num_layers, num_adapters, ...).
    Embed tokens LoRA params have shape (num_adapters, ...).
    """
    def extract(path, p):
        if is_stacked_lora_path(path):
            return p[:, adapter_idx].copy()
        return p[adapter_idx].copy()
    return jax.tree.map_with_path(extract, params)


def get_out_of_rank_params(params, adapter_idx: int, rank: int):
    """Extract out-of-rank params for an adapter.

    Returns the portion of LoRA weights beyond the effective rank,
    which should remain unchanged during training.
    """
    def slice_param(path, p):
        path_str = str(path)
        is_stacked = is_stacked_lora_path(path)
        if "lora_A" in path_str:
            if is_stacked:
                return p[:, adapter_idx, ..., rank:].copy()
            return p[adapter_idx, ..., rank:].copy()
        elif "lora_B" in path_str:
            if is_stacked:
                return p[:, adapter_idx, ..., rank:, :].copy()
            return p[adapter_idx, ..., rank:, :].copy()
        return p
    return jax.tree.map_with_path(slice_param, params)


def verify_params_unchanged(initial_params, final_params, error_msg_prefix: str):
    """Verify that params haven't changed between initial and final state."""
    for (path, initial), (_, final) in zip(
        jax.tree.leaves_with_path(initial_params), jax.tree.leaves_with_path(final_params)
    ):
        assert jnp.allclose(initial, final), f"{error_msg_prefix} for {path}"


def is_routed_expert_path(path) -> bool:
    """Check if path is for routed experts (not shared_experts)."""
    keys = []
    for p in path:
        if hasattr(p, "key"):
            keys.append(str(p.key))
        elif hasattr(p, "name"):
            keys.append(str(p.name))
    for i, key in enumerate(keys):
        if key == "experts" and i > 0 and keys[i - 1] == "mlp":
            return True
    return False


def get_moe_out_of_rank_params(params, adapter_idx: int, rank: int, num_experts: int):
    """Extract out-of-rank params for MoE models.

    For routed experts, uses effective rank = max(1, rank // num_experts).
    """
    def slice_param(path, p):
        path_str = str(path)
        effective_rank = max(1, rank // num_experts) if is_routed_expert_path(path) else rank
        is_stacked = is_stacked_lora_path(path)
        if "lora_A" in path_str:
            if is_stacked:
                return p[:, adapter_idx, ..., effective_rank:].copy()
            return p[adapter_idx, ..., effective_rank:].copy()
        elif "lora_B" in path_str:
            if is_stacked:
                return p[:, adapter_idx, ..., effective_rank:, :].copy()
            return p[adapter_idx, ..., effective_rank:, :].copy()
        return p
    return jax.tree.map_with_path(slice_param, params)
