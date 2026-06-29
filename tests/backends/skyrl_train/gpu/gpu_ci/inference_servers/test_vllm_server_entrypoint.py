"""
GPU CI test for the standalone vLLM server entrypoint:
``python -m skyrl.backends.skyrl_train.inference_servers.vllm_server_actor``.

Launches the entrypoint as a subprocess (the way external rollout-server
deployments like the Thunder agent use it), waits for the server to become
healthy, and exercises a basic health check + a simple chat completion for
Qwen2.5-1.5B-Instruct.

Run:
    uv run --isolated --extra dev --extra fsdp pytest \
        tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vllm_server_entrypoint.py -v -s
"""

import subprocess
import sys
import time

import httpx
import pytest

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS,
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    """Poll /health until OK, failing fast if the server process exits early."""
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server process exited early with code {proc.returncode}")
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server at {base_url} did not become healthy within {timeout}s")


@pytest.fixture(scope="module")
def standalone_server():
    """Launch the standalone vLLM server entrypoint as a subprocess (TP=1)."""
    port = get_open_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "skyrl.backends.skyrl_train.inference_servers.vllm_server_actor",
        "--model",
        MODEL,
        "--tensor-parallel-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.5",
        "--max-model-len",
        "2048",
        # The realistic deployment also wires up the SkyRL weight-sync worker extension.
        "--worker-extension-cls",
        VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS,
    ]
    proc = subprocess.Popen(cmd)
    try:
        _wait_for_health(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.mark.vllm
def test_entrypoint_health(standalone_server):
    resp = httpx.get(f"{standalone_server}/health", timeout=10.0)
    assert resp.status_code == 200


@pytest.mark.vllm
def test_entrypoint_chat_completion(standalone_server):
    resp = httpx.post(
        f"{standalone_server}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
        timeout=60.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    assert content and content.strip(), f"empty completion: {data}"


@pytest.mark.vllm
def test_entrypoint_custom_reset_prefix_cache(standalone_server):
    """The SkyRL custom endpoints (absent from a vanilla vLLM OpenAI server) are present."""
    resp = httpx.post(f"{standalone_server}/reset_prefix_cache", timeout=30.0)
    assert resp.status_code == 200, resp.text
    assert resp.json().get("status") == "ok"
