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
import logging
import time
import requests
import traceback
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional, TYPE_CHECKING

import fastapi
import uvicorn
from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput, ConversationType

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    check_unsupported_fields,
    build_sampling_params,
)

logger = logging.getLogger(__name__)

# Global state to hold the inference engine client and backend
_global_inference_engine_client: Optional["InferenceEngineClient"] = None
_global_uvicorn_server: Optional[uvicorn.Server] = None


def set_global_state(inference_engine_client: "InferenceEngineClient", uvicorn_server: uvicorn.Server):
    """Set the global inference engine client."""
    global _global_inference_engine_client
    global _global_uvicorn_server
    _global_inference_engine_client = inference_engine_client
    _global_uvicorn_server = uvicorn_server


def convert_openai_to_inference_input(request: ChatCompletionRequest, backend: str) -> InferenceEngineInput:
    """Convert OpenAI request to InferenceEngineInput format."""
    # Convert messages to our ConversationType format
    conversation: ConversationType = []
    for msg in request.messages:
        conversation.append({"role": msg.role, "content": msg.content})

    sampling_params = build_sampling_params(request, backend)

    engine_input: InferenceEngineInput = {
        "prompts": [conversation],
        "prompt_token_ids": None,
        "sampling_params": sampling_params if sampling_params else None,
        "trajectory_ids": [request.trajectory_id] if request.trajectory_id else None,
    }
    return engine_input


def convert_inference_output_to_openai(engine_output: InferenceEngineOutput, model_name: str) -> ChatCompletionResponse:
    """Convert InferenceEngineOutput to OpenAI response format."""
    response_text = engine_output["responses"][0]
    stop_reason = engine_output["stop_reasons"][0]

    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response_text),
        finish_reason=stop_reason,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        model=model_name,
        choices=[choice],
    )


async def handle_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse:
    """Handle chat completion request."""
    if _global_inference_engine_client is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Inference engine client not initialized"
        )
    if _global_inference_engine_client.model_name != request.model:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Model name mismatch: loaded model name {_global_inference_engine_client.model_name} != model name in request {request.model}",
        )

    try:
        check_unsupported_fields(request)
        # Convert to our internal format
        engine_input = convert_openai_to_inference_input(request, _global_inference_engine_client.backend)

        # Call the inference engine
        engine_output = await _global_inference_engine_client.generate(engine_input)

        # Convert back to OpenAI format
        response = convert_inference_output_to_openai(engine_output, _global_inference_engine_client.model_name)

        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}\n{traceback.format_exc()}")
        raise e


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
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completion(request: ChatCompletionRequest, raw_request: Request):
        return await handle_chat_completion(request, raw_request)

    # Health check endpoint
    # All inference engine replicas are initialized before creating `InferenceEngineClient`, and thus
    # we can start receiving requests as soon as the FastAPI server starts
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Exception handler for unhandled server errors
    # Note: Pydantic validation errors (400-level) are handled automatically by FastAPI
    # This handler only catches unexpected server-side exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message=f"Internal server error: {str(exc)}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR,
            ).model_dump(),
        )

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
