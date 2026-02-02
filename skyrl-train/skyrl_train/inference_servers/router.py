"""
Inference Router - HTTP proxy with session-aware routing (data plane only).

This router handles data plane operations only, similar to vllm-router:
- Routes requests to ONE server (session-aware or round-robin)
- Does NOT handle control plane operations (pause, resume, weight sync, etc.)

Control plane operations should be handled by RemoteInferenceClient, which
fans out directly to all backend servers.
"""

import asyncio
import hashlib
import itertools
import logging
import threading
import time
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

from skyrl_train.inference_servers.common import get_node_ip
from skyrl_train.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

logger = logging.getLogger(__name__)


class InferenceRouter:
    """
    HTTP proxy router for multiple vLLM servers (data plane only).

    This is a simple load-balancing router similar to vllm-router. It handles
    data plane operations only (generation, tokenization, etc.).

    Routing behavior:
    - If X-Session-ID header present: consistent hash to same backend
    - Otherwise: round-robin

    Control plane operations (pause, resume, sleep, weight sync) should be
    handled by RemoteInferenceClient, which fans out directly to all backends.

    Usage:
        router = InferenceRouter(server_urls, host="0.0.0.0", port=8080)
        router_url = router.start()
        # ... use router_url for inference ...
        router.shutdown()
    """

    def __init__(
        self,
        server_urls: List[str],
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        """
        Initialize the router.

        Args:
            server_urls: List of backend vLLM server URLs
            host: Host to bind router to
            port: Port to bind router to
        """
        self._server_urls = server_urls
        self._host = host
        self._port = port
        self._server_cycle = itertools.cycle(server_urls)
        self._client: Optional[httpx.AsyncClient] = None
        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None

        logger.info(f"InferenceRouter: {len(server_urls)} servers, port={port}")

    def _hash_session_id(self, session_id: str) -> int:
        """Hash session ID to get consistent server index."""
        hash_bytes = hashlib.sha256(session_id.encode()).digest()
        return int.from_bytes(hash_bytes[:8], "big")

    def _get_server_for_session(self, session_id: str) -> str:
        """Get consistent server URL for a session ID."""
        idx = self._hash_session_id(session_id) % len(self._server_urls)
        return self._server_urls[idx]

    def _get_server_round_robin(self) -> str:
        """Get next server URL in round-robin order."""
        return next(self._server_cycle)

    def _get_server_for_request(self, request: Request) -> str:
        """
        Determine server for a request.

        If X-Session-ID header is present, use consistent hashing.
        Otherwise, use round-robin.
        """
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return self._get_server_for_session(session_id)
        return self._get_server_round_robin()

    def _build_app(self) -> FastAPI:
        """Build the FastAPI app with proxy routes."""
        app = FastAPI(
            title="SkyRL Inference Router",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

        @app.get("/health")
        async def health():
            """Router health check (doesn't proxy to backends)."""
            # TODO: What should be the health check for router?
            # https://github.com/NovaSky-AI/SkyRL/issues/958
            return {"status": "healthy"}

        @app.get("/servers")
        async def list_servers():
            """Return list of server URLs."""
            return {"servers": self._server_urls}

        # Catch-all: proxy everything else to backends
        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def proxy(request: Request, path: str):
            return await self._proxy_request(request, f"/{path}")

        return app

    async def _proxy_request(self, request: Request, path: str) -> Response:
        """
        Proxy a request to one backend (session-aware or round-robin).
        """
        return await self._proxy_to_one(request, path)

    def _forward_headers(self, request: Request) -> dict:
        """Forward headers (filter out hop-by-hop headers)."""
        return {
            k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

    async def _proxy_to_one(self, request: Request, path: str) -> Response:
        """Proxy request to one server (data plane)."""
        server_url = self._get_server_for_request(request)
        url = f"{server_url}{path}"

        # Forward headers (filter out hop-by-hop headers)
        headers = self._forward_headers(request)

        response = await self._client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=await request.body(),
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    def start(self) -> str:
        """
        Start the router server in background.

        Returns:
            Router URL (e.g., "http://192.168.1.1:8080")
        """
        if not self._server_urls:
            raise ValueError("No servers available")

        # Create HTTP client for proxying
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        # Build FastAPI app and uvicorn server
        self._app = self._build_app()
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        # Start server in background thread
        self._server_thread = threading.Thread(target=asyncio.run, args=(self._server.serve(),), daemon=True)
        self._server_thread.start()

        ip = get_node_ip()
        router_url = f"http://{ip}:{self._port}"
        self._wait_until_healthy(router_url)

        logger.info(f"Router started at {router_url}")
        logger.info("  GET /servers - list backend server URLs")
        return router_url

    def _wait_until_healthy(
        self, router_url: str, timeout: float = SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S
    ) -> None:
        """Poll health endpoint until server is ready."""
        health_url = f"{router_url}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with httpx.Client() as client:
                    if client.get(health_url, timeout=1).status_code == 200:
                        return
            except httpx.RequestError:
                time.sleep(0.1)
        raise RuntimeError(f"Router failed to start within {timeout}s")

    def shutdown(self) -> None:
        """Shutdown the router gracefully."""
        logger.info("Shutting down router...")
        if self._server:
            self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=5)
