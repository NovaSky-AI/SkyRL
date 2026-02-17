"""
Utility functions for MoE Router Replay.
"""

import torch
from typing import Optional, List
from skyrl_train.training_batch import TrainingInputBatch

def _split_replay_indices(rollout_inference_indices: torch.Tensor) -> List[torch.Tensor]:
    if rollout_inference_indices is None:
        return None
    if rollout_inference_indices.dim() != 4:
        raise ValueError(f"Expected 4D replay indices, got shape {rollout_inference_indices.shape}")
    per_layer = rollout_inference_indices.permute(2, 0, 1, 3).contiguous()
    return [per_layer[i] for i in range(per_layer.shape[0])]

def setup_router_replay_forward(data: TrainingInputBatch, enable_router_replay: bool) -> bool:
    """
    Set up router replay for forward pass (ref/policy inference).
    """
    if not enable_router_replay:
        return False
        
    rollout_inference_indices = data.get("rollout_inference_indices")
    if rollout_inference_indices is None:
        return False
    
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    
    RouterReplay.set_replay_data(_split_replay_indices(rollout_inference_indices))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    
    return True


def setup_router_replay_backward(data: TrainingInputBatch, enable_router_replay: bool) -> bool:
    """
    Set up router replay for training forward/backward pass.
    """
    if not enable_router_replay:
        return False
        
    rollout_inference_indices = data.get("rollout_inference_indices")
    if rollout_inference_indices is None:
        return False
    
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    
    RouterReplay.set_replay_data(_split_replay_indices(rollout_inference_indices))
    # Use REPLAY_FORWARD - Megatron handles REPLAY_BACKWARD automatically
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    
    return True


def clear_router_replay():
    """Clear all router replay state."""
    from megatron.core.transformer.moe.router_replay import RouterReplay
    
    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_router_replay_instances()

