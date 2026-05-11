"""
ThunderAgent router for the SkyRL integration layer.

Composes ThunderAgent's standard routes (via register_routes) with
SkyRL-specific endpoints:
  - POST /inference/v1/generate  (token-based generation with scheduling)
  - GET  /servers                (list backend URLs)
  - /{path:path}                 (catch-all proxy, round-robin / session-aware)

Same interface as InferenceRouter: start() -> url, shutdown().
"""

import asyncio
import hashlib
import itertools
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpcore
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from ThunderAgent import (
    Config,
    MultiBackendRouter,
    get_program_id,
    register_routes,
    set_config,
)
from ThunderAgent.scheduler.vllm_request_processor import remove_program_id

from skyrl.backends.skyrl_train.inference_servers.common import get_node_ip
from skyrl.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

logger = logging.getLogger(__name__)


def _attach_thunderagent_file_handler(
    log_file: Optional[str],
) -> tuple[Optional[logging.Logger], Optional[logging.Handler]]:
    if not log_file:
        return None, None

    log_path = Path(log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path.resolve(), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"))

    file_logger = logging.getLogger("examples.train.thunder_agent.skyrl_integration")
    file_logger.setLevel(logging.INFO)
    file_logger.addHandler(file_handler)

    logger.info("ThunderAgent file logging enabled at %s", log_path.resolve())
    return file_logger, file_handler


def _detach_file_handler(file_logger: Optional[logging.Logger], file_handler: Optional[logging.Handler]) -> None:
    if file_logger and file_handler:
        file_logger.removeHandler(file_handler)
        file_handler.close()


class ThunderAgentRouter:
    """SkyRL integration layer for ThunderAgent.

    Standard ThunderAgent endpoints (registered via register_routes):
    - POST /v1/chat/completions, GET /health, GET /programs, etc.

    SkyRL-specific endpoints:
    - POST /inference/v1/generate  -> ThunderAgent-scheduled token generation
    - GET  /servers                -> list of backend URLs
    - /{path:path}                 -> catch-all proxy (round-robin / session-aware)

    Same interface as InferenceRouter: start() -> url, shutdown().
    """

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
        self._server_urls = server_urls
        self._host = host
        self._port = port
        self._log_file = log_file
        self._file_logger, self._file_handler = _attach_thunderagent_file_handler(log_file)

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

        # Round-robin state for catch-all proxy
        self._server_cycle = itertools.cycle(server_urls)

        # Created lazily in start()
        self._ta_router: Optional[MultiBackendRouter] = None
        self._proxy_client: Optional[httpx.AsyncClient] = None
        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None

        logger.info(
            "ThunderAgentRouter configured with "
            f"{len(server_urls)} servers, port={self._port}, mode={router_mode}, "
            f"backend_type={backend_type}, log_file={self._log_file}"
        )

    # ------------------------------------------------------------------
    # Session-aware routing for catch-all proxy
    # ------------------------------------------------------------------

    def _hash_session_id(self, session_id: str) -> int:
        hash_bytes = hashlib.sha256(session_id.encode()).digest()
        return int.from_bytes(hash_bytes[:8], "big")

    def _get_server_for_request(self, request: Request) -> str:
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            idx = self._hash_session_id(session_id) % len(self._server_urls)
            return self._server_urls[idx]
        return next(self._server_cycle)

    def _forward_headers(self, request: Request) -> dict:
        return {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")
        }

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        ta_router = self._ta_router
        proxy_client = self._proxy_client
        if ta_router is None or proxy_client is None:
            raise RuntimeError("ThunderAgentRouter app built before start() initialized router state")

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await ta_router.start()
            try:
                yield
            finally:
                await ta_router.stop()
                await proxy_client.aclose()

        app = FastAPI(
            title="SkyRL ThunderAgent Router",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
            lifespan=lifespan,
        )

        # Register all standard ThunderAgent routes
        register_routes(app, ta_router, config=self._ta_config)

        # -- SkyRL-specific: /inference/v1/generate -> ThunderAgent-scheduled generation --
        @app.post("/inference/v1/generate")
        async def inference_generate(request: Request):
            """Route SkyRL's primary generation endpoint through ThunderAgent.

            SkyRL's RemoteInferenceClient.generate() sends requests here with
            {token_ids, sampling_params, model, program_id}. This is the main
            rollout traffic during training.
            """
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON") from exc

            program_id = get_program_id(payload, request.headers)
            program_state = ta_router.get_or_create_program(program_id)

            if program_state.profile:
                program_state.profile.on_request_arrive()

            await ta_router.update_program_before_request(program_id, program_state, payload)

            if program_state.profile:
                program_state.profile.on_request_start()

            backend = ta_router.get_backend_for_program(program_id)

            # Strip ThunderAgent-only request metadata before forwarding to vLLM.
            forward_payload = remove_program_id(payload)

            url = f"{backend.url}/inference/v1/generate"
            headers = self._forward_headers(request)
            try:
                # Retry on ReadError (stale keepalive connections closed by vLLM/uvicorn)
                for _attempt in range(2):
                    try:
                        response = await proxy_client.request(
                            method="POST",
                            url=url,
                            headers=headers,
                            json=forward_payload,
                        )
                        break
                    except (httpcore.ReadError, httpx.ReadError) as exc:
                        if _attempt == 0:
                            logger.warning(
                                "/inference/v1/generate ReadError on %s (retry 1/1): %s",
                                backend.url,
                                exc,
                            )
                            continue
                        raise
            except Exception:
                ta_router.update_program_after_request(program_id, program_state, 0, 0)
                logger.exception("/inference/v1/generate failed for program_id=%s backend=%s", program_id, backend.url)
                raise

            # Estimate token count from the response for program state tracking
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
                    logger.exception(
                        "ThunderAgent profiling failed at generation request end for program_id=%s", program_id
                    )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        # -- SkyRL-specific: /servers --
        @app.get("/servers")
        async def list_servers():
            return {"servers": self._server_urls}

        # -- Catch-all proxy for all other endpoints --
        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def proxy(request: Request, path: str):
            server_url = self._get_server_for_request(request)
            url = f"{server_url}/{path}"
            headers = self._forward_headers(request)
            body = await request.body()
            for _attempt in range(2):
                try:
                    response = await proxy_client.request(
                        method=request.method,
                        url=url,
                        headers=headers,
                        content=body,
                    )
                    break
                except (httpcore.ReadError, httpx.ReadError) as exc:
                    if _attempt == 0:
                        logger.warning("Catch-all proxy ReadError on %s (retry 1/1): %s", url, exc)
                        continue
                    raise
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        return app

    # ------------------------------------------------------------------
    # Lifecycle (same interface as InferenceRouter)
    # ------------------------------------------------------------------

    def start(self) -> str:
        if not self._server_urls:
            raise ValueError("No servers available")

        # Set ThunderAgent global config before creating the router
        set_config(self._ta_config)

        # Create ThunderAgent router
        self._ta_router = MultiBackendRouter(
            self._ta_config.backends,
            profile_enabled=self._ta_config.profile_enabled,
            scheduling_enabled=(self._ta_config.router_mode == "tr"),
            scheduler_interval=self._ta_config.scheduler_interval,
            backend_type=self._ta_config.backend_type,
            acting_token_weight=self._ta_config.acting_token_weight,
            use_acting_token_decay=self._ta_config.use_acting_token_decay,
        )

        # Create HTTP client for catch-all proxy
        proxy_limits = httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
            keepalive_expiry=3.0,  # Must be < uvicorn's timeout-keep-alive (5s default)
        )
        self._proxy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            limits=proxy_limits,
            transport=httpx.AsyncHTTPTransport(
                limits=proxy_limits,
                retries=1,  # Retry on ConnectError (stale pool connections)
            ),
        )

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

        logger.info(f"ThunderAgentRouter started at {router_url}")
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
        _detach_file_handler(self._file_logger, self._file_handler)
