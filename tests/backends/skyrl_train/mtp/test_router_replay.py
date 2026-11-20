"""CPU test: the MTP heads' MoE routers leave rollout routing replay (R3); the trunk's routers stay.

uv run --isolated --extra megatron --extra dev pytest tests/backends/skyrl_train/mtp/test_router_replay.py
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.transformer.moe.router_replay import RouterReplay  # noqa: E402

from skyrl.backends.skyrl_train.mtp.router_replay import (  # noqa: E402
    detach_mtp_routers_from_replay,
)

pytestmark = pytest.mark.megatron


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router_replay = RouterReplay()


def _model() -> torch.nn.Module:
    """Megatron-style names: one trunk router under ``decoder`` and one head router under ``mtp.layers.0``."""
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.router = _Router()
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.mtp.layers[0].router = _Router()
    return model


def test_mtp_routers_leave_routing_replay_and_trunk_routers_stay():
    saved = list(RouterReplay.global_router_replay_instances)
    model = _model()
    try:
        RouterReplay.global_router_replay_instances[:] = [
            model.decoder.router.router_replay,
            model.mtp.layers[0].router.router_replay,
        ]
        assert detach_mtp_routers_from_replay([model]) == 1
        assert model.mtp.layers[0].router.router_replay is None
        assert model.decoder.router.router_replay is not None
        assert RouterReplay.global_router_replay_instances == [model.decoder.router.router_replay]
        # A second call finds nothing left to remove.
        assert detach_mtp_routers_from_replay([model]) == 0
    finally:
        RouterReplay.global_router_replay_instances[:] = saved


def test_mtp_must_be_a_whole_path_component():
    """A module whose name only contains "mtp" (e.g. ``decoder.mtp_like``) is not an MTP head."""
    saved = list(RouterReplay.global_router_replay_instances)
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.mtp_like = _Router()
    try:
        RouterReplay.global_router_replay_instances[:] = [model.decoder.mtp_like.router_replay]
        assert detach_mtp_routers_from_replay([model]) == 0
        assert model.decoder.mtp_like.router_replay is not None
    finally:
        RouterReplay.global_router_replay_instances[:] = saved
