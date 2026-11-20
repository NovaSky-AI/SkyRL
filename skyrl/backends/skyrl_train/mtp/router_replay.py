# Keep the MTP heads' MoE routers out of rollout routing replay (R3). Every router registers a ``RouterReplay``
# in ``RouterReplay.global_router_replay_instances``, the MTP block's too, but vLLM captures routes for the
# target's layers only: ``set_replay_data`` would see more instances than captured layers, and a layer-number
# match would hand a head a trunk layer's routes. The heads route on their own, as the drafter does.

from __future__ import annotations

from typing import Iterable


def detach_mtp_routers_from_replay(model_chunks: Iterable) -> int:
    """Remove the MTP blocks' routers from routing replay. Returns how many were removed on this rank."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    removed = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            replay = getattr(module, "router_replay", None)
            # Dot-padding the name matches "mtp" only as a whole path component ("mtp.layers.0..." or nested).
            if replay is None or ".mtp." not in f".{name}.":
                continue
            module.router_replay = None
            instances = RouterReplay.global_router_replay_instances
            for i, instance in enumerate(instances):
                if instance is replay:
                    del instances[i]
                    break
            removed += 1
    return removed
