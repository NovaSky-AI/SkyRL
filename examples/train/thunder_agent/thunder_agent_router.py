"""
ThunderAgent Router - SkyRL integration layer.

Composes ThunderAgent's standard routes (via register_routes) with
SkyRL-specific endpoints:
  - POST /inference/v1/generate
  - POST /tokenize
  - POST /detokenize

Same interface as InferenceRouter: start() -> url, shutdown().
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from skyrl.backends.skyrl_train.inference_servers.common import get_node_ip, get_open_port
from skyrl.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

try:
    from ThunderAgent import Config, MultiBackendRouter, get_program_id, register_routes, set_config

    THUNDERAGENT_AVAILABLE = True
except ImportError:
    THUNDERAGENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThunderAgentRouter:
    """SkyRL integration layer for ThunderAgent."""

    def __init__(
        self,
        server_urls: List[str],
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        log_file: Optional[str] = None,
        router_mode: str = "tr",
        backend_type: str = "vllm",
        acting_token_weight: float = 1.0,
        scheduler_interval: float = 5.0,
        use_acting_token_decay: bool = False,
        profile_enabled: bool = False,
        metrics_enabled: bool = False,
        metrics_interval: float = 5.0,
    ):
        if not THUNDERAGENT_AVAILABLE:
            raise ImportError("ThunderAgent is not installed. Install with: cd ThunderAgent && pip install -e .")

        self._server_urls = server_urls
        self._host = host
        self._port = port
        self._log_file = log_file
        self._access_logger = logging.getLogger("thunderagent.access")

        self._ta_config = Config(
            backends=list(server_urls),
            router_mode=router_mode,
            backend_type=backend_type,
            profile_enabled=profile_enabled,
            metrics_enabled=metrics_enabled,
            metrics_interval=metrics_interval,
            scheduler_interval=scheduler_interval,
            acting_token_weight=acting_token_weight,
            use_acting_token_decay=use_acting_token_decay,
        )

        self._ta_router: Optional[MultiBackendRouter] = None
        self._proxy_client: Optional[httpx.AsyncClient] = None
        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None

    def _forward_headers(self, request: Request) -> dict:
        return {
            k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

    def _get_first_healthy_backend_url(self) -> str:
        if self._ta_router is not None:
            for url, backend in self._ta_router.backends.items():
                if backend.healthy:
                    return url
        if not self._server_urls:
            raise HTTPException(status_code=503, detail="No backend servers configured")
        return self._server_urls[0]

    async def _proxy_request_to_backend(self, request: Request, *, backend_url: str, path: str) -> Response:
        if self._proxy_client is None:
            raise HTTPException(status_code=503, detail="Router proxy client is not initialized")

        response = await self._proxy_client.request(
            method=request.method,
            url=f"{backend_url}{path}",
            headers=self._forward_headers(request),
            content=await request.body(),
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    def _build_app(self) -> FastAPI:
        ta_router = self._ta_router

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await ta_router.start()
            yield
            await ta_router.stop()

        app = FastAPI(
            title="SkyRL ThunderAgent Router",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
            lifespan=lifespan,
        )

        register_routes(app, ta_router, config=self._ta_config)

        @app.post("/inference/v1/generate")
        async def inference_generate(request: Request):
            start_time = time.perf_counter()
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON") from exc

            program_id = get_program_id(payload)
            program_state = ta_router.get_or_create_program(program_id)

            if program_state.profile:
                program_state.profile.on_request_arrive()

            await ta_router.update_program_before_request(program_id, program_state, payload)

            if program_state.profile:
                program_state.profile.on_request_start()

            backend = ta_router.get_backend_for_program(program_id)
            forward_payload = {k: v for k, v in payload.items() if k != "program_id"}
            headers = self._forward_headers(request)

            if self._proxy_client is None:
                raise HTTPException(status_code=503, detail="Router proxy client is not initialized")

            try:
                response = await self._proxy_client.request(
                    method="POST",
                    url=f"{backend.url}/inference/v1/generate",
                    headers=headers,
                    json=forward_payload,
                )
            except Exception:
                ta_router.update_program_after_request(program_id, program_state, 0, 0)
                self._access_logger.exception(
                    "/inference/v1/generate program_id=%s backend=%s failed after %.1fms",
                    program_id,
                    backend.url,
                    (time.perf_counter() - start_time) * 1000.0,
                )
                raise

            total_tokens = 0
            prompt_tokens = 0
            try:
                resp_data = response.json()
                token_ids = payload.get("token_ids", [])
                prompt_tokens = len(token_ids)
                generated_tokens = len(resp_data.get("choices", [{}])[0].get("token_ids", []))
                total_tokens = prompt_tokens + generated_tokens
            except Exception:
                pass

            ta_router.update_program_after_request(program_id, program_state, total_tokens, prompt_tokens)

            if program_state.profile:
                try:
                    program_state.profile.on_request_end(prompt_tokens, 0)
                except Exception:
                    logger.exception("ThunderAgent profiling failed at generation request end for program_id=%s", program_id)

            self._access_logger.info(
                "/inference/v1/generate program_id=%s backend=%s status=%s latency_ms=%.1f",
                program_id,
                backend.url,
                response.status_code,
                (time.perf_counter() - start_time) * 1000.0,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        @app.post("/tokenize")
        async def tokenize(request: Request):
            return await self._proxy_request_to_backend(
                request,
                backend_url=self._get_first_healthy_backend_url(),
                path="/tokenize",
            )

        @app.post("/detokenize")
        async def detokenize(request: Request):
            return await self._proxy_request_to_backend(
                request,
                backend_url=self._get_first_healthy_backend_url(),
                path="/detokenize",
            )

        return app

    def start(self) -> str:
        if not self._server_urls:
            raise ValueError("No servers available")

        set_config(self._ta_config)
        self._ta_router = MultiBackendRouter(
            self._ta_config.backends,
            profile_enabled=self._ta_config.profile_enabled,
            scheduling_enabled=(self._ta_config.router_mode == "tr"),
            scheduler_interval=self._ta_config.scheduler_interval,
            backend_type=self._ta_config.backend_type,
            acting_token_weight=self._ta_config.acting_token_weight,
            use_acting_token_decay=self._ta_config.use_acting_token_decay,
        )
        self._proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        self._app = self._build_app()
        self._port = get_open_port(self._port)
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=asyncio.run, args=(self._server.serve(),), daemon=True)
        self._server_thread.start()

        router_url = f"http://{get_node_ip()}:{self._port}"
        self._wait_until_healthy(router_url)
        logger.info("ThunderAgentRouter started at %s", router_url)
        return router_url

    def _wait_until_healthy(
        self, router_url: str, timeout: float = SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S
    ) -> None:
        health_url = f"{router_url}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with httpx.Client() as client:
                    if client.get(health_url, timeout=1).status_code == 200:
                        return
            except httpx.RequestError:
                time.sleep(0.1)
        raise RuntimeError(f"ThunderAgentRouter failed to start within {timeout}s")

    def shutdown(self) -> None:
        logger.info("Shutting down ThunderAgentRouter...")
        if self._server:
            self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=5)
