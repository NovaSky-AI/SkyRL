import jax
import jax.numpy as jnp

from skyrl.tx.utils.models import get_adapter_idx


def get_adapter_params(params, adapter_idx):
    """Extract adapter params at specific index."""

    def extract(path, p):
        idx = get_adapter_idx(path, adapter_idx)
        return p[idx].copy()

    return jax.tree.map_with_path(extract, params)


def get_out_of_rank_params(params, adapter_idx, rank):
    """Extract out-of-rank params for an adapter."""

    def slice_param(path, p):
        path_str = str(path)
        idx = get_adapter_idx(path, adapter_idx)
        if "lora_A" in path_str:
            return p[*idx, ..., rank:].copy()
        elif "lora_B" in path_str:
            return p[*idx, ..., rank:, :].copy()
        return p

    return jax.tree.map_with_path(slice_param, params)


def verify_params_unchanged(initial_params, final_params, error_msg_prefix):
    """Verify that params have not changed between initial and final states."""
    for (path, initial), (_, final) in zip(
        jax.tree.leaves_with_path(initial_params), jax.tree.leaves_with_path(final_params)
    ):
        assert jnp.allclose(initial, final), f"{error_msg_prefix} for {path}"


def _is_routed_expert_path(path) -> bool:
    """Disambiguate shared_experts and experts"""
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
    """Extract out-of-rank params, using effective rank for routed expert layers."""

    def slice_param(path, p):
        path_str = str(path)

        if _is_routed_expert_path(path):
            effective_rank = max(1, rank // num_experts)
        else:
            effective_rank = rank

        idx = get_adapter_idx(path, adapter_idx)
        if "lora_A" in path_str:
            # lora_A shape: [adapters, ..., max_rank] - slice last dim
            return p[*idx, ..., effective_rank:].copy()
        elif "lora_B" in path_str:
            # lora_B shape: [adapters, ..., max_rank, out] - slice second-to-last dim
            return p[*idx, ..., effective_rank:, :].copy()
        return p

    return jax.tree.map_with_path(slice_param, params)
