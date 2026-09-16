import asyncio
import importlib
from argparse import Namespace
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

pytest.importorskip(
    "vllm", reason="LoRA transport request gating is installed on the vLLM server actor"
)
pytestmark = pytest.mark.vllm

import skyrl.backends.skyrl_train.inference_servers.vllm_server_actor as server_actor  # noqa: E402


def _create_app():
    cli_args = Namespace(middleware=[])
    gate = server_actor.VLLMServerActor._prepare_lora_transport_admission(cli_args)
    app = FastAPI()
    module_path, object_name = cli_args.middleware[0].rsplit(".", 1)
    app.add_middleware(getattr(importlib.import_module(module_path), object_name))
    paused = asyncio.Event()
    resumed = asyncio.Event()

    async def pause_generation(**kwargs):
        paused.set()

    async def resume_generation():
        resumed.set()

    engine = SimpleNamespace(
        pause_generation=pause_generation, resume_generation=resume_generation
    )
    server_actor.VLLMServerActor._bind_lora_transport_admission(app, gate)
    server_actor.VLLMServerActor._add_custom_endpoints(
        app,
        engine,
        Namespace(),
        lora_transport_admission_gate=gate,
    )
    return app, engine, paused, resumed


def test_custom_endpoints_can_be_added_after_middleware_stack_is_built():
    cli_args = Namespace(middleware=[])
    gate = server_actor.VLLMServerActor._prepare_lora_transport_admission(cli_args)
    app = FastAPI()
    module_path, object_name = cli_args.middleware[0].rsplit(".", 1)
    app.add_middleware(getattr(importlib.import_module(module_path), object_name))
    app.middleware_stack = app.build_middleware_stack()
    server_actor.VLLMServerActor._bind_lora_transport_admission(app, gate)
    engine = SimpleNamespace()

    server_actor.VLLMServerActor._add_custom_endpoints(
        app,
        engine,
        Namespace(),
        lora_transport_admission_gate=gate,
    )

    paths = {route.path for route in app.routes}
    assert "/skyrl/v1/pause_lora_transport" in paths
    assert app.state.lora_transport_admission_gate is gate
    with pytest.raises(
        RuntimeError, match="Cannot add middleware after an application has started"
    ):
        app.add_middleware(
            getattr(importlib.import_module(module_path), object_name),
            gate=server_actor.LoRATransportAdmissionGate(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample_path",
    ["/v1/completions", "/inference/v1/generate", "/skyrl/v1/generate", "/score"],
)
async def test_queued_request_drains_before_pause_and_new_request_resolves_replacement(
    sample_path,
):
    app, _, paused, _ = _create_app()
    current_adapter = {"id": 1}
    observed = []
    old_entered = asyncio.Event()
    finish_old = asyncio.Event()

    async def sample(request: Request):
        adapter_id = current_adapter["id"]
        observed.append(adapter_id)
        if (await request.json())["wait"]:
            old_entered.set()
            await finish_old.wait()
        return {"adapter_id": adapter_id}

    # The SkyRL generate route already exists; replace only its test endpoint.
    if sample_path == "/skyrl/v1/generate":
        app.router.routes = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) != sample_path
        ]
    app.add_api_route(sample_path, sample, methods=["POST"])
    gate = app.state.lora_transport_admission_gate
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://example.com"
    ) as client:
        async with asyncio.timeout(5):
            async with asyncio.TaskGroup() as tasks:
                old_request = tasks.create_task(
                    client.post(sample_path, json={"wait": True})
                )
                await old_entered.wait()
                gate.close()
                pause = tasks.create_task(client.post("/skyrl/v1/pause_lora_transport"))
                new_request = tasks.create_task(
                    client.post(sample_path, json={"wait": False})
                )
                await asyncio.sleep(0)

                assert observed == [1]
                assert not paused.is_set()
                finish_old.set()
                assert (await old_request).json()["adapter_id"] == 1
                assert (await pause).status_code == 200
                assert paused.is_set()
                assert observed == [1]

                current_adapter["id"] = 2
                assert (
                    await client.post("/skyrl/v1/resume_lora_transport")
                ).status_code == 200
                assert (await new_request).json()["adapter_id"] == 2
                assert observed == [1, 2]


@pytest.mark.asyncio
async def test_streaming_response_keeps_admission_counted_until_cancelled():
    app, _, _, _ = _create_app()
    streaming = asyncio.Event()
    never_finish = asyncio.Event()

    @app.post("/v1/completions")
    async def sample():
        async def chunks():
            streaming.set()
            yield b"first token"
            await never_finish.wait()

        return StreamingResponse(chunks())

    gate = app.state.lora_transport_admission_gate
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://example.com"
    ) as client:
        async with asyncio.timeout(5):
            request = asyncio.create_task(client.post("/v1/completions"))
            await streaming.wait()
            gate.close()
            drain = asyncio.create_task(gate.wait_until_idle())
            await asyncio.sleep(0)
            assert not drain.done()

            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await drain


@pytest.mark.asyncio
async def test_drain_timeout_reopens_admission_without_pausing_scheduler(monkeypatch):
    app, _, paused, _ = _create_app()
    gate = app.state.lora_transport_admission_gate
    monkeypatch.setattr(server_actor, "SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", 0.01)
    entered = asyncio.Event()
    finish = asyncio.Event()

    @app.post("/v1/completions")
    async def sample():
        entered.set()
        await finish.wait()
        return {"adapter_id": 1}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://example.com"
    ) as client:
        request = asyncio.create_task(client.post("/v1/completions"))
        try:
            await entered.wait()
            response = await client.post("/skyrl/v1/pause_lora_transport")
            assert response.status_code == 504
            assert not paused.is_set()
            async with asyncio.timeout(1):
                async with gate.admit():
                    pass
        finally:
            finish.set()
            await request


@pytest.mark.asyncio
async def test_scheduler_pause_failure_keeps_admission_closed():
    app, engine, _, _ = _create_app()

    async def fail_pause(**kwargs):
        raise RuntimeError("scheduler state unknown")

    engine.pause_generation = fail_pause
    gate = app.state.lora_transport_admission_gate
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://example.com"
    ) as client:
        with pytest.raises(RuntimeError, match="scheduler state unknown"):
            await client.post("/skyrl/v1/pause_lora_transport")
        entered = asyncio.Event()

        async def try_enter():
            async with gate.admit():
                entered.set()

        request = asyncio.create_task(try_enter())
        await asyncio.sleep(0)
        assert not entered.is_set()
        request.cancel()
        with suppress(asyncio.CancelledError):
            await request
