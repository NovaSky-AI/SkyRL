"""Drain HTTP requests before replacing adapters referenced by queued inference."""

import asyncio
from contextlib import asynccontextmanager

from starlette.types import ASGIApp, Receive, Scope, Send


class LoRATransportAdmissionGate:
    """Keep admitted requests alive through response streaming and cancellation."""

    def __init__(self) -> None:
        self._open = asyncio.Event()
        self._open.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active = 0

    @asynccontextmanager
    async def admit(self):
        while not self._open.is_set():
            await self._open.wait()
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    def close(self) -> None:
        self._open.clear()

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    def open(self) -> None:
        self._open.set()


class LoRATransportAdmissionMiddleware:
    def __init__(self, app: ASGIApp, gate: LoRATransportAdmissionGate | None = None) -> None:
        self.app = app
        self.gate = gate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        model_request = path.startswith(("/v1/", "/inference/v1/")) or path in {
            "/skyrl/v1/generate",
            "/score",
            "/rerank",
            "/pooling",
            "/classify",
        }
        if not model_request or path.endswith("/cancel"):
            await self.app(scope, receive, send)
            return
        gate = self.gate or scope["app"].state.lora_transport_admission_gate
        async with gate.admit():
            await self.app(scope, receive, send)
