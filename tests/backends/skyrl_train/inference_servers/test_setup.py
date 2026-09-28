"""Tests for the deployment launcher in ``inference_servers/setup.py`` (no servers are started)."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

import skyrl.backends.skyrl_train.inference_servers.setup as inference_setup
from skyrl.backends.skyrl_train.inference_servers.common import (
    SERVER_PORT_STRIDE,
    VLLM_START_PORT,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.setup import (
    InferenceServerSetup,
    create_inference_servers,
    launch_remote_inference_client,
)
from skyrl.train.config import InferenceEngineConfig


class FakeServerGroup:
    """Records its constructor arguments and answers with canned server infos."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.index = len(FakeServerGroup.instances)
        FakeServerGroup.instances.append(self)

    def start(self, blocking=True):
        return [f"ref-{self.index}-{i}" for i in range(self.kwargs["num_servers"])]

    @property
    def server_infos(self):
        return [SimpleNamespace(url=f"http://server-{self.index}-{i}:8000") for i in range(self.kwargs["num_servers"])]


@pytest.fixture
def launcher_doubles(monkeypatch):
    """Replace everything that would touch Ray or start a process."""
    import skyrl.backends.skyrl_train.inference_servers.server_group as server_group_module
    import skyrl.backends.skyrl_train.inference_servers.vllm_router as vllm_router_module

    FakeServerGroup.instances = []
    monkeypatch.setattr(server_group_module, "ServerGroup", FakeServerGroup)

    class FakeRouter:
        def __init__(self, router_args, log_path):
            self.router_args = router_args
            self.log_path = log_path

        def start(self):
            return "http://router:7000"

    monkeypatch.setattr(vllm_router_module, "VLLMRouter", FakeRouter)
    monkeypatch.setattr(inference_setup, "build_router_args", lambda ie_cfg, **kwargs: kwargs)
    monkeypatch.setattr(inference_setup.ray, "get", lambda refs: refs)

    pg_requests = []

    def fake_placement_group(bundles, strategy):
        pg_requests.append((len(bundles), strategy))
        return f"raw-pg-{len(pg_requests)}"

    monkeypatch.setattr(inference_setup, "ray_placement_group", fake_placement_group)
    monkeypatch.setattr(inference_setup, "get_ray_pg_ready_with_timeout", lambda pg, timeout: None)
    monkeypatch.setattr(inference_setup, "ResolvedPlacementGroup", lambda pg: SimpleNamespace(pg=pg))
    return pg_requests


def test_create_inference_servers_offsets_port_windows_from_start_port(launcher_doubles):
    ie_cfg = InferenceEngineConfig(num_engines=2, tensor_parallel_size=2, data_parallel_size=1)
    cli_args = Namespace(tensor_parallel_size=2, pipeline_parallel_size=1)

    result = create_inference_servers(ie_cfg, cli_args, log_path="/tmp/logs", start_port=8800)

    assert [g.kwargs["start_port"] for g in FakeServerGroup.instances] == [8800, 8800 + SERVER_PORT_STRIDE]
    assert [g.kwargs["placement_group_bundle_offset"] for g in FakeServerGroup.instances] == [0, 2]
    assert result.server_urls == ["http://server-0-0:8000", "http://server-1-0:8000"]
    assert result.proxy_url == "http://router:7000"
    assert launcher_doubles == [(4, "PACK")]  # one PG of num_engines * tp * pp * dp single-GPU bundles


def test_create_inference_servers_default_port_and_supplied_pg(launcher_doubles):
    ie_cfg = InferenceEngineConfig(num_engines=1)
    external_pg = SimpleNamespace(pg="colocate-pg")

    result = create_inference_servers(
        ie_cfg,
        Namespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        log_path="/tmp/logs",
        placement_group=external_pg,
    )

    assert FakeServerGroup.instances[0].kwargs["start_port"] == VLLM_START_PORT
    assert FakeServerGroup.instances[0].kwargs["placement_group"] is external_pg
    assert launcher_doubles == []  # nothing created when the caller supplied the PG
    assert result.server_urls == ["http://server-0-0:8000"]


def test_launch_remote_inference_client_wraps_the_deployment(monkeypatch):
    ie_cfg = InferenceEngineConfig(num_engines=2, data_parallel_size=2)
    canned = InferenceServerSetup(
        proxy_url="http://router:7000",
        server_urls=[f"http://s{i}:8000" for i in range(4)],
    )
    calls = []

    def fake_create(ie_cfg_arg, cli_args, log_path, placement_group, start_port):
        calls.append((ie_cfg_arg, cli_args, log_path, placement_group, start_port))
        return canned

    monkeypatch.setattr(inference_setup, "create_inference_servers", fake_create)

    client, server_setup = launch_remote_inference_client(
        ie_cfg,
        "cli-args",
        model_name="teacher",
        log_path="/tmp/logs",
        start_port=8800,
        enable_return_routed_experts=True,
        enable_return_sample_support_set=True,
        uses_lora_weight_sync=True,
    )

    assert server_setup is canned
    assert calls == [(ie_cfg, "cli-args", "/tmp/logs", None, 8800)]
    assert isinstance(client, RemoteInferenceClient)
    assert client.proxy_url == "http://router:7000"
    assert client.server_urls == canned.server_urls
    assert client.model_name == "teacher"
    assert client.data_parallel_size == 2
    assert client.enable_return_routed_experts is True
    assert client.enable_return_sample_support_set is True
    assert client.uses_lora_weight_sync is True
    assert client.tokenizer is None
