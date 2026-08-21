"""Generator-owned OpenAI-compatible TITO proxy."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Set
from uuid import uuid4

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
)

from .renderer import TITORenderer, convert_routed_experts
from .trace import Trace
from .types import ModelTurnResult
from .vllm_openai import (
    OpenAIProtocolError,
    build_chat_response,
    build_sampling_params,
    parse_chat_request,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TITOProxyConfig:
    host: str = "127.0.0.1"
    port: int = 0
    max_request_bytes: int = 64 * 1024 * 1024
    drain_timeout_seconds: float = 30.0
    max_model_len: int = 32768
    default_max_tokens: Optional[int] = None


@dataclass(frozen=True)
class ProxyHandle:
    base_url: str


@dataclass
class _Registration:
    trace: Trace
    router_session_id: str
    cache_salt: Optional[str]
    model: str
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_tasks: Set[asyncio.Task] = field(default_factory=set)
    active_requests: int = 0
    closing: bool = False
    drained: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.drained.set()


class TITOProxy:
    """Serve exact token-in/token-out inference to external chat clients."""

    def __init__(
        self,
        inference_engine_client: InferenceEngineInterface,
        renderer: TITORenderer,
        *,
        config: Optional[TITOProxyConfig] = None,
    ) -> None:
        self._inference_engine_client = inference_engine_client
        self._renderer = renderer
        self._config = config or TITOProxyConfig()
        self._registrations: Dict[str, _Registration] = {}
        self._registrations_lock = asyncio.Lock()
        self._app = FastAPI()
        self._app.post("/sessions/{registration_token}/v1/chat/completions")(self._chat_completions)
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        self._socket: Optional[socket.socket] = None
        self._base_url: Optional[str] = None

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        if self._server_task is not None:
            raise RuntimeError("TITO proxy is already serving")

        # Reserve the ephemeral port while uvicorn starts.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._config.host, self._config.port))
        sock.listen(2048)
        sock.setblocking(False)
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                self._app,
                host=self._config.host,
                port=port,
                log_level="warning",
                lifespan="off",
            )
        )
        self._socket = sock
        self._server = server
        self._base_url = f"http://{self._config.host}:{port}"
        self._server_task = asyncio.create_task(server.serve(sockets=[sock]))

        while not server.started:
            if self._server_task.done():
                await self._server_task
                raise RuntimeError("TITO proxy failed to start")
            await asyncio.sleep(0.01)

        try:
            yield
        finally:
            async with self._registrations_lock:
                if self._registrations:
                    raise RuntimeError("Cannot stop TITO proxy while registrations are active")
            server.should_exit = True
            await self._server_task
            sock.close()
            self._server_task = None
            self._server = None
            self._socket = None
            self._base_url = None

    @asynccontextmanager
    async def register(
        self,
        trace: Trace,
        *,
        router_session_id: str,
        cache_salt: Optional[str],
        model: str,
    ) -> AsyncIterator[ProxyHandle]:
        if self._base_url is None:
            raise RuntimeError("TITO proxy must be serving before registering a trace")
        if not router_session_id:
            raise ValueError("router_session_id must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")

        token = uuid4().hex
        # Isolate each registration with a random path token.
        registration = _Registration(
            trace=trace,
            router_session_id=router_session_id,
            cache_salt=cache_salt,
            model=model,
        )
        async with self._registrations_lock:
            self._registrations[token] = registration

        try:
            yield ProxyHandle(base_url=f"{self._base_url}/sessions/{token}")
        finally:
            # Complete cleanup even if the caller is cancelled.
            cleanup_task = asyncio.create_task(
                self._close_registration(
                    token,
                    registration,
                    router_session_id=router_session_id,
                )
            )
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
                raise

    async def _close_registration(
        self,
        token: str,
        registration: _Registration,
        *,
        router_session_id: str,
    ) -> None:
        registration.closing = True
        if registration.active_requests:
            try:
                await asyncio.wait_for(
                    registration.drained.wait(),
                    timeout=self._config.drain_timeout_seconds,
                )
            except TimeoutError:
                logger.warning("Timed out draining TITO proxy session %s", router_session_id)
                active_tasks = list(registration.active_tasks)
                for task in active_tasks:
                    task.cancel()
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
        registration.trace.seal()
        async with self._registrations_lock:
            self._registrations.pop(token, None)
        await self._inference_engine_client.finish_session(router_session_id)

    async def _chat_completions(self, registration_token: str, request: Request):
        registration = await self._begin_request(registration_token)
        try:
            content_length = request.headers.get("content-length")
            if content_length is not None and int(content_length) > self._config.max_request_bytes:
                raise HTTPException(status_code=413, detail="Request body is too large")
            raw_body = await request.body()
            if len(raw_body) > self._config.max_request_bytes:
                raise HTTPException(status_code=413, detail="Request body is too large")
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Request body must be a JSON object")

            try:
                parsed = parse_chat_request(
                    body,
                    registered_model=registration.model,
                )
            except OpenAIProtocolError as exc:
                return JSONResponse(content=exc.body, status_code=exc.status_code)
            try:
                async with registration.turn_lock:
                    return await self._execute_turn(registration, parsed)
            except OpenAIProtocolError as exc:
                return JSONResponse(content=exc.body, status_code=exc.status_code)
            except aiohttp.ClientResponseError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.message) from exc
        finally:
            self._end_request(registration)

    async def _begin_request(self, token: str) -> _Registration:
        async with self._registrations_lock:
            registration = self._registrations.get(token)
            if registration is None:
                raise HTTPException(status_code=404, detail="Unknown TITO proxy registration")
            if registration.closing:
                raise HTTPException(status_code=410, detail="TITO proxy registration is closing")
            registration.active_requests += 1
            current_task = asyncio.current_task()
            if current_task is not None:
                registration.active_tasks.add(current_task)
            registration.drained.clear()
            return registration

    @staticmethod
    def _end_request(registration: _Registration) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            registration.active_tasks.discard(current_task)
        registration.active_requests -= 1
        if registration.active_requests == 0:
            registration.drained.set()

    async def _execute_turn(self, registration: _Registration, parsed) -> Dict[str, Any]:
        pending = registration.trace.prepare_turn(
            parsed.messages,
            tools=parsed.tools,
        )
        bridge_transition = (
            registration.trace.transition(pending.bridge_transition_id)
            if pending.bridge_transition_id is not None
            else None
        )
        rendered = None
        if bridge_transition is not None:
            matched_message_count = len(bridge_transition.node_ids)
            new_messages = pending.messages[matched_message_count:]
        else:
            matched_message_count = 0
            new_messages = pending.messages
        if bridge_transition is not None and new_messages:
            # Preserve prior sampled IDs when the renderer can bridge.
            rendered = await asyncio.to_thread(
                self._renderer.bridge,
                bridge_transition.prompt_token_ids,
                bridge_transition.completion_ids,
                new_messages,
                tools=pending.tools,
            )
        if rendered is None:
            # Fall back to a full render when bridging is unsafe.
            rendered = await asyncio.to_thread(
                self._renderer.render,
                pending.messages,
                tools=pending.tools,
            )

        prompt_message_indices = list(rendered.message_indices)
        if rendered.reused_prefix_length:
            if bridge_transition is None:
                raise RuntimeError("Renderer returned a reused prefix without a bridge Transition")
            # Convert tail-relative attribution to full-message indices.
            for index in range(rendered.reused_prefix_length, len(prompt_message_indices)):
                if prompt_message_indices[index] >= 0:
                    prompt_message_indices[index] += matched_message_count

        sampling_params = build_sampling_params(
            parsed,
            prompt_token_count=len(rendered.token_ids),
            max_model_len=self._config.max_model_len,
            renderer_stop_token_ids=self._renderer.get_stop_token_ids(),
            default_max_tokens=self._config.default_max_tokens,
        )
        engine_input = InferenceEngineInput(
            prompts=None,
            prompt_token_ids=[list(rendered.token_ids)],
            sampling_params=sampling_params,
            session_ids=[registration.router_session_id],
            mm_features=None,
            cache_salt=registration.cache_salt,
        )
        engine_output = await self._inference_engine_client.generate(engine_input, model=registration.model)
        if len(engine_output["response_ids"]) != 1 or len(engine_output["stop_reasons"]) != 1:
            raise RuntimeError("TITO proxy expected exactly one inference result")
        response_ids = engine_output["response_ids"][0]
        response_logprobs_batch = engine_output.get("response_logprobs")
        if response_logprobs_batch is None or len(response_logprobs_batch) != 1:
            raise RuntimeError("Inference engine did not return selected-token logprobs")
        response_logprobs = response_logprobs_batch[0]
        if len(response_ids) != len(response_logprobs):
            raise RuntimeError("Inference completion IDs and logprobs have different lengths")

        assistant_message = await asyncio.to_thread(
            self._renderer.parse_response,
            response_ids,
            tools=pending.tools,
        )
        routed_experts_batch = engine_output.get("rollout_expert_indices")
        routed_experts = routed_experts_batch[0] if routed_experts_batch is not None else None
        expected_expert_length = len(rendered.token_ids) + len(response_ids)
        if routed_experts is not None and len(routed_experts) != expected_expert_length:
            raise RuntimeError("Inference routed-expert data must cover the full prompt and completion sequence")
        result = ModelTurnResult(
            prompt_token_ids=tuple(rendered.token_ids),
            prompt_message_indices=tuple(prompt_message_indices),
            reused_prefix_length=rendered.reused_prefix_length,
            completion_ids=tuple(response_ids),
            completion_logprobs=tuple(response_logprobs),
            assistant_message=assistant_message,
            stop_reason=engine_output["stop_reasons"][0],
            routed_experts=convert_routed_experts(routed_experts),
            model=parsed.model,
            sampling_params_json=json.dumps(sampling_params, sort_keys=True, separators=(",", ":")),
        )
        # Commit before returning the successful response.
        registration.trace.commit(pending, result)

        return build_chat_response(
            parsed=parsed,
            assistant_message=assistant_message,
            prompt_token_ids=rendered.token_ids,
            completion_ids=response_ids,
            completion_logprobs=response_logprobs,
            finish_reason=engine_output["stop_reasons"][0],
            decode_token=self._renderer.decode_token,
        )
