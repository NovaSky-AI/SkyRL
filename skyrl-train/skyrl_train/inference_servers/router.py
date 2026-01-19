"""
Inference Router - HTTP proxy with session-aware routing and control plane fan-out.
"""

import asyncio
import hashlib
import itertools
import logging
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from skyrl_train.inference_servers.common import ServerInfo, get_node_ip

logger = logging.getLogger(__name__)


# Routes that go to ONE backend (data plane)
DATA_PLANE_ROUTES = [
    "/v1/completions",
    "/v1/chat/completions",
    "/tokenize",
    "/detokenize",
    "/health",
    "/models",
    "/version",
]

# Routes that go to ALL backends (control plane)
CONTROL_PLANE_ROUTES = [
    "/pause",
    "/resume",
    "/sleep",
    "/wake_up",
    "/wakeup",
    "/reset_prefix_cache",
    "/collective_rpc",
    "/init_weight_transfer",
    "/update_weights",
    "/finalize_weight_update",
]


class InferenceRouter:
    """
    HTTP proxy router for multiple vLLM backends.

    Routing behavior:
    - Data plane (generation requests): Routes to ONE backend
      - If X-Session-ID header present: consistent hash to same backend
      - Otherwise: round-robin
    - Control plane (sleep, pause, weight sync): Fans out to ALL backends

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
        self._server_task: Optional[asyncio.Task] = None
        self._shutdown_event: Optional[asyncio.Event] = None

        logger.info(f"InferenceRouter: {len(server_urls)} backends, port={port}")

    def _hash_session_id(self, session_id: str) -> int:
        """Hash session ID to get consistent backend index."""
        hash_bytes = hashlib.sha256(session_id.encode()).digest()
        return int.from_bytes(hash_bytes[:8], "big")

    def _get_backend_for_session(self, session_id: str) -> str:
        """Get consistent backend URL for a session ID."""
        idx = self._hash_session_id(session_id) % len(self._server_urls)
        return self._server_urls[idx]

    def _get_backend_round_robin(self) -> str:
        """Get next backend URL in round-robin order."""
        return next(self._server_cycle)

    def _get_backend_for_request(self, request: Request) -> str:
        """
        Determine backend for a request.

        If X-Session-ID header is present, use consistent hashing.
        Otherwise, use round-robin.
        """
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return self._get_backend_for_session(session_id)
        return self._get_backend_round_robin()

    def _is_control_plane_route(self, path: str) -> bool:
        """Check if path is a control plane route (fan-out to all)."""
        return any(path.startswith(route) for route in CONTROL_PLANE_ROUTES)

    def _build_app(self) -> FastAPI:
        """Build the FastAPI app with proxy routes."""
        app = FastAPI(
            title="SkyRL Inference Router",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

        @app.get("/servers")
        async def list_servers():
            """Return list of backend URLs."""
            return {"servers": self._server_urls}

        @app.get("/get_server_info")
        async def get_server_info():
            """Fetch server info from any backend (all should return same)."""
            backend = self._server_urls[0]
            try:
                resp = await self._client.get(f"{backend}/get_server_info", timeout=10.0)
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

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
        Proxy a request to backend(s).

        Control plane routes go to ALL backends.
        Data plane routes go to ONE backend (session-aware or round-robin).
        """
        if self._is_control_plane_route(path):
            return await self._proxy_to_all(request, path)
        else:
            return await self._proxy_to_one(request, path)

    async def _proxy_to_one(self, request: Request, path: str) -> Response:
        """Proxy request to one backend (data plane)."""
        backend_url = self._get_backend_for_request(request)
        url = f"{backend_url}{path}"
        method = request.method

        body = await request.body()

        # Check if streaming is requested
        is_streaming = False
        if body:
            try:
                import json

                data = json.loads(body)
                is_streaming = data.get("stream", False)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Forward headers (filter out hop-by-hop headers)
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        if is_streaming:
            return await self._proxy_streaming(url, method, headers, body)
        else:
            return await self._proxy_regular(url, method, headers, body)

    async def _proxy_regular(
        self, url: str, method: str, headers: dict, body: bytes
    ) -> Response:
        """Proxy a regular (non-streaming) request."""
        response = await self._client.request(
            method=method,
            url=url,
            headers=headers,
            content=body,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    async def _proxy_streaming(
        self, url: str, method: str, headers: dict, body: bytes
    ) -> StreamingResponse:
        """Proxy a streaming request."""

        async def stream_generator():
            async with self._client.stream(
                method=method,
                url=url,
                headers=headers,
                content=body,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )

    async def _proxy_to_all(self, request: Request, path: str) -> Response:
        """Proxy request to all backends (control plane), aggregate responses."""
        method = request.method
        body = await request.body()

        # Forward headers
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        # Send to all backends concurrently
        async def call_backend(backend_url: str):
            url = f"{backend_url}{path}"
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                    timeout=300.0,  # Long timeout for weight sync
                )
                return {
                    "url": backend_url,
                    "status": response.status_code,
                    "body": response.json() if response.content else None,
                }
            except Exception as e:
                return {
                    "url": backend_url,
                    "status": 500,
                    "error": str(e),
                }

        results = await asyncio.gather(
            *[call_backend(url) for url in self._server_urls]
        )

        # Check if all succeeded
        all_ok = all(r.get("status") == 200 for r in results)

        if all_ok:
            return Response(
                content='{"status": "ok"}',
                status_code=200,
                media_type="application/json",
            )
        else:
            import json

            return Response(
                content=json.dumps({"status": "partial_failure", "results": results}),
                status_code=207,  # Multi-Status
                media_type="application/json",
            )

    def start(self) -> str:
        """
        Start the router server in background.

        Returns:
            Router URL (e.g., "http://192.168.1.1:8080")
        """
        if not self._server_urls:
            raise ValueError("No backend servers available")

        # Create HTTP client for proxying
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        # Build FastAPI app
        self._app = self._build_app()

        # Create shutdown event
        self._shutdown_event = asyncio.Event()

        # Start server in background thread (since we're not in async context)
        import threading

        def run_server():
            asyncio.run(self._run_server())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait a bit for server to start
        import time

        time.sleep(1)

        ip = get_node_ip()
        router_url = f"http://{ip}:{self._port}"
        logger.info(f"Router started at {router_url}")
        logger.info(f"  GET /servers - list backend servers")
        logger.info(f"  GET /get_server_info - get parallelism info")
        return router_url

    async def _run_server(self) -> None:
        """Run the uvicorn server."""
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()

    def shutdown(self) -> None:
        """Shutdown the router."""
        logger.info("Shutting down router...")
        if self._shutdown_event:
            self._shutdown_event.set()
        # Note: Thread will exit when uvicorn server stops
