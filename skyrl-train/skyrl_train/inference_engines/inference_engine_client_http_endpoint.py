"""
OpenAI-compatible HTTP endpoint using InferenceEngineClient as backend.

This module provides a FastAPI-based HTTP endpoint that exposes OpenAI's chat completion API
while routing requests to our internal InferenceEngineClient system.

Main functions:
- serve(): Start the HTTP endpoint.
- wait_for_server_ready(): Wait for server to be ready.
- shutdown_server(): Shutdown the server.
"""

import asyncio
import json
import logging
import time
import requests
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional, TYPE_CHECKING, Dict, Any

import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

logger = logging.getLogger(__name__)

# Global state to hold the inference engine client and backend
_global_inference_engine_client: Optional["InferenceEngineClient"] = None
_global_uvicorn_server: Optional[uvicorn.Server] = None


# Adapted from vllm.entrypoints.openai.protocol.ErrorResponse
class ErrorInfo(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


def set_global_state(inference_engine_client: "InferenceEngineClient", uvicorn_server: uvicorn.Server):
    """Set the global inference engine client."""
    global _global_inference_engine_client
    global _global_uvicorn_server
    _global_inference_engine_client = inference_engine_client
    _global_uvicorn_server = uvicorn_server


def _validate_chat_completion(request_json: Dict[str, Any]) -> Optional[ErrorResponse]:
    """
    The only validation that SkyRL does to the request. Rest of the validations are done
    by the underlying inference engines (vLLM / SGLang).
    """
    if _global_inference_engine_client is None:
        return ErrorResponse(
            error=ErrorInfo(
                message="Inference engine client not initialized",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
    if "model" not in request_json:
        return ErrorResponse(
            error=ErrorInfo(
                message="The field `model` is required in your `/chat/completion` request.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if _global_inference_engine_client.model_name != request_json["model"]:
        # TODO(Charlie): add a config similar to vllm's `served_model_name`.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        return ErrorResponse(
            error=ErrorInfo(
                message=f"Model name mismatch: loaded model name {_global_inference_engine_client.model_name} != model name in request {request_json['model']}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if request_json.get("stream", False):
        return ErrorResponse(
            error=ErrorInfo(
                message="Streaming is not supported in SkyRL yet, please set stream to False.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    return None


async def handle_chat_completion(raw_request: Request) -> JSONResponse:
    """Handle chat completion request."""
    try:
        request_json = await raw_request.json()

        # SkyRL-side validation
        error_response = _validate_chat_completion(request_json)
        if error_response is not None:
            return JSONResponse(content=error_response.model_dump(), status_code=error_response.error.code)

        # Serialize fastapi.Request because it is not pickable, which causes ray methods to fail.
        payload = {
            "json": request_json,
            "headers": dict(raw_request.headers) if hasattr(raw_request, "headers") else {},
        }
        response = await _global_inference_engine_client.chat_completion(payload)

        if "error" in response and "message" in response["error"]:
            return JSONResponse(content=response, status_code=response["error"]["code"])
        else:
            return JSONResponse(content=response)

    except json.JSONDecodeError as e:
        # To catch possible raw_request.json() errors
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Invalid JSON error: {str(e)}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
        return JSONResponse(content=error_response.model_dump(), status_code=HTTPStatus.BAD_REQUEST.value)
    except Exception as e:
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Error in chat completion: {str(e)}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return JSONResponse(content=error_response.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value)


def shutdown_server(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """Shutdown the server.

    Args:
        host: Server host.
        port: Server port.
        max_wait_seconds: How long to wait until the server stops listening.

    Raises:
        Exception: If the server is still responding after *max_wait_seconds*.
    """
    if _global_uvicorn_server is not None:
        _global_uvicorn_server.should_exit = True

    health_url = f"http://{host}:{port}/health"

    for i in range(max_wait_seconds):
        try:
            # If this succeeds, server is still alive
            requests.get(health_url, timeout=1)
        except requests.exceptions.RequestException:
            # A network error / connection refused means server is down.
            logger.info(f"Server shut down after {i+1} seconds")
            return
        time.sleep(1)

    raise Exception(f"Server failed to shut down within {max_wait_seconds} seconds")


def wait_for_server_ready(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """
    Wait for the HTTP endpoint to be ready by polling the health endpoint.

    Args:
        host: Host where the server is running
        port: Port where the server is running
        max_wait_seconds: Maximum time to wait in seconds

    Raises:
        Exception: If server doesn't become ready within max_wait_seconds
    """
    max_retries = max_wait_seconds
    health_url = f"http://{host}:{port}/health"

    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                logger.info(f"Server ready after {i+1} attempts ({i+1} seconds)")
                return
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            if i == max_retries - 1:
                raise Exception(f"Server failed to start within {max_wait_seconds} seconds")
            time.sleep(1)  # Wait 1 second between retries


def create_app() -> fastapi.FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        logger.info("Starting inference HTTP endpoint...")
        yield

    app = fastapi.FastAPI(
        title="InferenceEngine OpenAI-Compatible API",
        description="OpenAI-compatible chat completion API using InferenceEngineClient",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Chat completion endpoint
    @app.post("/v1/chat/completions")
    async def chat_completion(raw_request: Request):
        """
        Takes in OpenAI's `ChatCompletionRequest` and returns OpenAI's `ChatCompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        Same for SGLang. SkyRL does not perform field validation beyond `model` and `trajectory_id`,
        and otherwise depends on the underlying engines' validation.

        Make sure you add in `trajectory_id` to ensure load balancing and sticky routing. The same
        agentic rollout / session should share the same `trajectory_id` so they get routed to the
        same engine for better prefix caching. If unprovided, we will route to a random engine which
        is not performant.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_chat_completion(raw_request)

    # Health check endpoint
    # All inference engine replicas are initialized before creating `InferenceEngineClient`, and thus
    # we can start receiving requests as soon as the FastAPI server starts
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # This handler only catches unexpected server-side exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Unhandled exception: {str(exc)}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return JSONResponse(content=error_response.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value)

    return app


def serve(
    inference_engine_client: "InferenceEngineClient",
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
):
    """
    Start the HTTP endpoint.

    Args:
        inference_engine_client: The InferenceEngineClient to use as backend
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 8000)
        log_level: Logging level (default: "info")
    """
    # Create app
    app = create_app()

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    logger.info(f"Starting server on {host}:{port}")

    # Run server
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=True)
    server = uvicorn.Server(config)

    # Expose server for external shutdown control (tests)
    set_global_state(inference_engine_client, server)

    try:
        # Run until shutdown
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
