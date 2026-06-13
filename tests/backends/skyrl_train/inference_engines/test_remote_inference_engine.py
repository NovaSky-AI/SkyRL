"""
Tests for `skyrl/backends/skyrl_train/inference_engines/remote_inference_engine.py`.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/inference_engines/test_remote_inference_engine.py
"""

import asyncio
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import (
    RemoteInferenceEngine,
)
from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.train.config import (
    GeneratorConfig,
    InferenceEngineConfig,
    ModelConfig,
    PolicyConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)

MODEL_PATH = "org/test-model"
SERVED_MODEL_NAME = "test-model"


class FakeTokenizer:
    """Minimal tokenizer stub; `generate()` only calls `encode` on response texts."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def create_mock_vllm_server() -> FastAPI:
    """Mock vLLM OpenAI-compatible server that only knows `SERVED_MODEL_NAME`."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        model = body.get("model")
        if model != SERVED_MODEL_NAME:
            # Mirrors vLLM's 404 response body for an unknown model name.
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"The model `{model}` does not exist.",
                        "type": "NotFoundError",
                        "param": "model",
                        "code": 404,
                    }
                },
            )
        prompts = body.get("prompt", [])
        return {
            "choices": [{"index": i, "text": f"response {i}", "finish_reason": "stop"} for i in range(len(prompts))],
            "model": model,
        }

    return app


def wait_ready(url: str, timeout: float = 5.0) -> bool:
    """Wait for server to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock vLLM server, return its host:port (no scheme)."""
    port = get_open_port()
    config = uvicorn.Config(create_mock_vllm_server(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True).start()
    assert wait_ready(f"http://127.0.0.1:{port}"), "Mock server failed to start"

    yield f"127.0.0.1:{port}"

    server.should_exit = True
    time.sleep(0.3)


def _make_engine(mock_server: str, model_name: str) -> RemoteInferenceEngine:
    return RemoteInferenceEngine(
        url=mock_server,
        model_name=model_name,
        engine_backend="vllm",
        tokenizer=FakeTokenizer(),
    )


@pytest.mark.asyncio
async def test_generate_parses_choices(mock_server):
    """Happy path: a 200 response with choices is parsed into InferenceEngineOutput."""
    engine = _make_engine(mock_server, SERVED_MODEL_NAME)
    output = await engine.generate({"prompt_token_ids": [[1, 2, 3], [4, 5]], "sampling_params": {"max_tokens": 4}})

    assert output["responses"] == ["response 0", "response 1"]
    assert output["stop_reasons"] == ["stop", "stop"]
    assert len(output["response_ids"]) == 2


@pytest.mark.asyncio
async def test_generate_raises_on_http_error(mock_server):
    """A non-200 response (e.g. vLLM's 404 for an unknown model name) must raise with the
    status and error body instead of being silently parsed into empty outputs.

    See https://github.com/NovaSky-AI/SkyRL/issues/1672.
    """
    engine = _make_engine(mock_server, MODEL_PATH)
    with pytest.raises(RuntimeError, match=r"404.*does not exist") as exc_info:
        await engine.generate({"prompt_token_ids": [[1, 2, 3]], "sampling_params": {"max_tokens": 4}})

    # The served model name should appear in the error to make the mismatch debuggable.
    assert MODEL_PATH in str(exc_info.value)


# -------------------------------------------
# tests for create_remote_inference_engines_from_config
# --------------------------------------------


def _make_remote_cfg(served_model_name=None) -> SkyRLTrainConfig:
    return SkyRLTrainConfig(
        trainer=TrainerConfig(
            policy=PolicyConfig(model=ModelConfig(path=MODEL_PATH)),
        ),
        generator=GeneratorConfig(
            inference_engine=InferenceEngineConfig(
                backend="vllm",
                run_engines_locally=False,
                remote_urls=["127.0.0.1:8000"],
                served_model_name=served_model_name,
            ),
        ),
    )


def test_create_remote_engines_uses_served_model_name():
    """`generator.inference_engine.served_model_name` is used as the model name when set."""
    from skyrl.train.entrypoints.main_base import (
        create_remote_inference_engines_from_config,
    )

    cfg = _make_remote_cfg(served_model_name=SERVED_MODEL_NAME)
    engines = create_remote_inference_engines_from_config(cfg, tokenizer=FakeTokenizer())
    assert all(engine.model_name == SERVED_MODEL_NAME for engine in engines)


def test_create_remote_engines_falls_back_to_model_path():
    """Without `served_model_name`, the policy model path is used as the model name."""
    from skyrl.train.entrypoints.main_base import (
        create_remote_inference_engines_from_config,
    )

    cfg = _make_remote_cfg(served_model_name=None)
    engines = create_remote_inference_engines_from_config(cfg, tokenizer=FakeTokenizer())
    assert all(engine.model_name == MODEL_PATH for engine in engines)
