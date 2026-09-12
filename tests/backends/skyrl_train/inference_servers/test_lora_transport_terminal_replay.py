import asyncio
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

pytest.importorskip(
    "vllm", reason="native LoRA terminal replay is installed on the vLLM server actor"
)
pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (  # noqa: E402
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (  # noqa: E402
    VLLMServerActor,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport import (  # noqa: E402
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRAUpdateRequest,
)


def _add_custom_endpoints(app, engine):
    gate = VLLMServerActor._prepare_lora_transport_admission(Namespace(middleware=[]))
    VLLMServerActor._bind_lora_transport_admission(app, gate)
    VLLMServerActor._add_custom_endpoints(
        app,
        engine,
        Namespace(),
        lora_transport_admission_gate=gate,
    )


def _create_layout():
    source = LoRABridgeSource(
        key="adapter.weight",
        source_rank=0,
        hf_param_names=("adapter.weight",),
        component="linear_out",
        transform="identity",
        shape=(1, 2),
        tensor_parallel_axis=0,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=(),
    )
    layout = LoRABridgeSourceLayout("adapter", (source,))
    return layout


@pytest.mark.asyncio
async def test_terminal_http_replays_preserve_restored_route_and_newer_transaction():
    app = FastAPI()
    calls = []

    async def collective_rpc(method, kwargs):
        calls.append((method, kwargs))

    _add_custom_endpoints(app, SimpleNamespace(collective_rpc=collective_rpc))
    adapter_ids = iter(range(10, 20))
    models = SimpleNamespace(
        lora_requests={},
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_id_counter=SimpleNamespace(inc=lambda amount: next(adapter_ids)),
    )
    app.state.openai_serving_models = models
    layout = _create_layout()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://example.com"
    ) as client:

        async def stage(generation):
            body = {
                "lora_name": "adapter",
                "request": LoRAUpdateRequest.from_layout(
                    layout, generation
                ).to_json_dict(),
                "transport": "nccl",
            }
            response = await client.post("/skyrl/v1/stage_lora_nccl_adapter", json=body)
            assert response.status_code == 200
            return {**body, "adapter_id": response.json()["lora_int_id"]}

        async def phase(name, body):
            return await client.post(
                f"/skyrl/v1/{name}_lora_transport_adapter", json=body
            )

        initial = await stage(0)
        assert (await phase("activate", initial)).status_code == 200
        assert (await phase("commit", initial)).status_code == 200
        original = models.lora_requests["adapter"]

        failed = await stage(1)
        assert (await phase("activate", failed)).status_code == 200
        assert (await phase("rollback", failed)).status_code == 200
        assert models.lora_requests["adapter"] is original
        calls_after_rollback = list(calls)
        assert (await phase("rollback", failed)).status_code == 200
        assert calls == calls_after_rollback
        assert models.lora_requests["adapter"] is original

        replacement = await stage(2)
        previous_requests = dict(models._skyrl_lora_transport_previous_requests)
        assert (await phase("rollback", failed)).status_code == 200
        assert models._skyrl_lora_transport_previous_requests == previous_requests
        assert (await phase("activate", replacement)).status_code == 200
        assert (await phase("commit", replacement)).status_code == 200
        new_route = models.lora_requests["adapter"]

        later = await stage(3)
        calls_before_replay = list(calls)
        previous_requests = dict(models._skyrl_lora_transport_previous_requests)
        assert (await phase("commit", replacement)).status_code == 200
        assert calls == calls_before_replay
        assert models._skyrl_lora_transport_previous_requests == previous_requests
        assert models.lora_requests["adapter"] is new_route

        assert (await phase("rollback", replacement)).status_code == 500
        assert (await phase("rollback", failed)).status_code == 500
        assert models.lora_requests["adapter"] is new_route
        assert models._skyrl_lora_transport_previous_requests == previous_requests
        assert (
            models._skyrl_lora_transport_lifecycle._staged["adapter"][1]
            == later["adapter_id"]
        )

        assert (await phase("rollback", later)).status_code == 200
        response = await client.post(
            "/skyrl/v1/unload_lora_transport_adapter", json={"lora_name": "adapter"}
        )
        assert response.status_code == 200
        assert "adapter" not in models.lora_requests
        assert "adapter" not in models._skyrl_lora_transport_previous_requests
        calls_after_unload = list(calls)
        assert (
            await client.post(
                "/skyrl/v1/unload_lora_transport_adapter", json={"lora_name": "adapter"}
            )
        ).status_code == 200
        assert calls == calls_after_unload


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lost_response", "partial_stage_cleanup"])
async def test_fleet_recovers_unacknowledged_stage_without_losing_previous_route(
    monkeypatch, failure
):
    layout = _create_layout()
    request = LoRAUpdateRequest.from_layout(layout, 1)
    apps, models_by_url, calls = {}, {}, []
    failures_enabled = False

    for index in range(2):
        url = f"http://server-{index}.example.com"
        app = FastAPI()

        async def collective_rpc(method, kwargs, server=url):
            calls.append((server, method, kwargs))
            if (
                failures_enabled
                and server == "http://server-1.example.com"
                and failure == "partial_stage_cleanup"
            ):
                if method == "stage_lora_nccl_adapter" or (
                    method == "discard_lora_transport_adapter"
                    and sum(item[0] == server and item[1] == method for item in calls)
                    == 1
                ):
                    raise RuntimeError("injected staging cleanup failure")

        _add_custom_endpoints(app, SimpleNamespace(collective_rpc=collective_rpc))
        ids = iter(range(10, 30))
        models = SimpleNamespace(
            lora_requests={},
            lora_resolver_lock=defaultdict(asyncio.Lock),
            lora_id_counter=SimpleNamespace(
                inc=lambda amount, counter=ids: next(counter)
            ),
        )
        app.state.openai_serving_models = models
        apps[url], models_by_url[url] = app, models

    async def call_http(url, endpoint, payload):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(apps[url]), base_url=url
        ) as http:
            response = await http.post(endpoint, json=payload)
            response.raise_for_status()
            return url, {"status": response.status_code, "body": response.json()}

    common = {"lora_name": "adapter", "transport": "nccl"}
    initial = {
        **common,
        "request": LoRAUpdateRequest.from_layout(layout, 0).to_json_dict(),
    }
    for url in apps:
        _, response = await call_http(url, "/skyrl/v1/stage_lora_nccl_adapter", initial)
        payload = {**initial, "adapter_id": response["body"]["lora_int_id"]}
        for phase in ("activate", "commit"):
            await call_http(url, f"/skyrl/v1/{phase}_lora_transport_adapter", payload)
    previous = {
        url: models.lora_requests["adapter"] for url, models in models_by_url.items()
    }
    failures_enabled = True

    async def call_server(url, endpoint, payload):
        result = await call_http(url, endpoint, payload)
        if (
            failure == "lost_response"
            and url == "http://server-1.example.com"
            and endpoint.endswith("/stage_lora_nccl_adapter")
        ):
            raise TimeoutError("stage response lost")
        return result

    client = RemoteInferenceClient(
        proxy_url="http://router.example.com",
        server_urls=list(apps),
        data_parallel_size=2,
    )
    monkeypatch.setattr(client, "_call_server", call_server)
    try:
        with pytest.raises((TimeoutError, httpx.HTTPStatusError)):
            await client.load_lora_nccl_adapter("adapter", request.to_json_dict())
    finally:
        await client.teardown()

    for url, models in models_by_url.items():
        assert models.lora_requests["adapter"] is previous[url]
        lifecycle = models._skyrl_lora_transport_lifecycle
        assert lifecycle.get_active_adapter_id("adapter") == previous[url].lora_int_id
        assert not lifecycle._staged
        aborted = {**common, "request": request.to_json_dict()}
        before = list(calls)
        await call_http(url, "/skyrl/v1/rollback_lora_transport_adapter", aborted)
        assert calls == before
        assert models.lora_requests["adapter"] is previous[url]

    failures_enabled = False
    for url in apps:
        following = {
            **common,
            "request": LoRAUpdateRequest.from_layout(layout, 2).to_json_dict(),
        }
        await call_http(url, "/skyrl/v1/stage_lora_nccl_adapter", following)
