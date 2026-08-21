"""CPU tests for the generator-owned TITO proxy."""

import asyncio

import httpx
import pytest

from skyrl.train.generators.tito.proxy import TITOProxy, TITOProxyConfig
from skyrl.train.generators.tito.renderer import RenderedPrompt
from skyrl.train.generators.tito.trace import Trace


class FakeRenderer:
    def render(self, messages, *, tools=None):
        token_ids = []
        message_indices = []
        for index, message in enumerate(messages):
            token_ids.append(10 + index)
            message_indices.append(index)
        token_ids.append(99)
        message_indices.append(-1)
        return RenderedPrompt(tuple(token_ids), tuple(message_indices))

    def bridge(self, anchor, new_messages, *, tools=None):
        token_ids = list(anchor.previous_prompt_ids + anchor.previous_completion_ids)
        message_indices = [-1] * len(token_ids)
        for index, _ in enumerate(new_messages):
            token_ids.append(30 + index)
            message_indices.append(index)
        token_ids.append(99)
        message_indices.append(-1)
        return RenderedPrompt(
            tuple(token_ids),
            tuple(message_indices),
            reused_prefix_length=len(anchor.previous_prompt_ids) + len(anchor.previous_completion_ids),
        )

    def parse_response(self, token_ids, *, tools=None):
        return {"role": "assistant", "content": ",".join(str(token_id) for token_id in token_ids)}

    def get_stop_token_ids(self):
        return [2]

    def decode_token(self, token_id):
        return str(token_id)


class FakeInferenceEngine:
    def __init__(self):
        self.generate_calls = []
        self.finished_sessions = []
        self.delay = 0.0

    async def generate(self, input_batch, model=None):
        self.generate_calls.append((input_batch, model))
        if self.delay:
            await asyncio.sleep(self.delay)
        return {
            "responses": ["response"],
            "response_ids": [[50, 51]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.2, -0.3]],
            "prompt_logprobs": None,
            "rollout_expert_indices": None,
        }

    async def finish_session(self, session_id):
        self.finished_sessions.append(session_id)


@pytest.mark.asyncio
async def test_proxy_runs_token_inference_and_commits_trace():
    engine = FakeInferenceEngine()
    proxy = TITOProxy(engine, FakeRenderer())
    trace = Trace()

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="session-1", cache_salt="salt", model="model") as handle:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{handle.base_url}/v1/chat/completions",
                    json={
                        "model": "model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "temperature": 0.7,
                        "max_tokens": 8,
                    },
                )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "50,51"
    assert len(trace.committed_turns()) == 1
    assert trace.transition(0).model == "model"
    assert trace.transition(0).sampling_params["temperature"] == 0.7
    assert trace.transition(0).sampling_params["max_tokens"] == 8
    assert engine.generate_calls[0][0]["prompt_token_ids"] == [[10, 99]]
    assert engine.generate_calls[0][0]["cache_salt"] == "salt"
    assert engine.finished_sessions == ["session-1"]
    assert trace.is_sealed


@pytest.mark.asyncio
async def test_identical_concurrent_requests_are_coalesced():
    engine = FakeInferenceEngine()
    engine.delay = 0.05
    proxy = TITOProxy(engine, FakeRenderer())
    trace = Trace()
    body = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="session-1", cache_salt=None, model="model") as handle:
            async with httpx.AsyncClient() as client:
                first, second = await asyncio.gather(
                    client.post(f"{handle.base_url}/v1/chat/completions", json=body),
                    client.post(f"{handle.base_url}/v1/chat/completions", json=body),
                )

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(engine.generate_calls) == 1
    assert len(trace.committed_turns()) == 1


@pytest.mark.asyncio
async def test_proxy_rejects_unsupported_streaming():
    engine = FakeInferenceEngine()
    proxy = TITOProxy(engine, FakeRenderer())
    trace = Trace()

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="session-1", cache_salt=None, model="model") as handle:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{handle.base_url}/v1/chat/completions",
                    json={
                        "model": "model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )

    assert response.status_code == 400
    assert "Streaming" in response.json()["error"]["message"]
    assert not engine.generate_calls


@pytest.mark.asyncio
async def test_proxy_enforces_configured_model_length():
    engine = FakeInferenceEngine()
    proxy = TITOProxy(
        engine,
        FakeRenderer(),
        config=TITOProxyConfig(max_model_len=1),
    )
    trace = Trace()

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="session-1", cache_salt=None, model="model") as handle:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{handle.base_url}/v1/chat/completions",
                    json={
                        "model": "model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

    assert response.status_code == 400
    assert "model" in response.json()["error"]["message"]
    assert not engine.generate_calls


@pytest.mark.asyncio
async def test_registration_cleanup_survives_drain_timeout():
    engine = FakeInferenceEngine()
    engine.delay = 0.1
    proxy = TITOProxy(
        engine,
        FakeRenderer(),
        config=TITOProxyConfig(drain_timeout_seconds=0.01),
    )
    trace = Trace()
    body = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="session-1", cache_salt=None, model="model") as handle:
            async with httpx.AsyncClient() as client:
                request_task = asyncio.create_task(client.post(f"{handle.base_url}/v1/chat/completions", json=body))
                while not engine.generate_calls:
                    await asyncio.sleep(0.001)
        try:
            response = await request_task
        except httpx.TransportError:
            response = None

    if response is not None:
        assert response.status_code >= 500
    assert trace.is_sealed
    assert engine.finished_sessions == ["session-1"]
