import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from sqlmodel import Session, create_engine, select

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from skyrl.tinker.db_models import FutureDB, RequestStatus  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


class _FakeInferenceState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


def _start_fake_inference_server():
    state = _FakeInferenceState()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            state.enter()
            try:
                time.sleep(0.05)
                token_count = int(payload["max_tokens"])
                choices = [
                    {
                        "token_ids": [1000 + index] * token_count,
                        "logprobs": {"token_logprobs": [-0.1] * token_count},
                        "finish_reason": "length",
                    }
                    for index in range(int(payload["n"]))
                ]
                body = json.dumps({"choices": choices}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                state.exit()

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="fake-vllm", daemon=True)
    thread.start()
    return server, thread, state


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _api_is_up(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/healthz", timeout=1).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


@contextmanager
def _tinker_api(tmp_path, *, external_url: str, limit: int):
    port = _free_port()
    db_path = tmp_path / "tinker-e2e.db"
    log_path = tmp_path / "tinker-e2e.log"
    cmd = [
        "uv",
        "run",
        "--isolated",
        "--extra",
        "tinker",
        "--extra",
        "jax",
        "-m",
        "skyrl.tinker.api",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-model",
        BASE_MODEL,
        "--backend",
        "jax",
        "--database-url",
        f"sqlite:///{db_path}",
        "--external-inference-url",
        external_url,
        "--sampling-concurrency",
        json.dumps({"enabled": True, "initial_limit": limit}),
    ]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and not _api_is_up(port):
                if process.poll() is not None:
                    break
                time.sleep(0.25)
            if not _api_is_up(port):
                pytest.fail(f"Tinker API failed to start:\n{log_path.read_text()}")
            yield port, db_path, log_path
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def test_sdk_to_tinker_engine_fixed_window_e2e(tmp_path):
    inference_server, inference_thread, inference_state = _start_fake_inference_server()
    external_url = f"http://127.0.0.1:{inference_server.server_port}"
    try:
        with _tinker_api(tmp_path, external_url=external_url, limit=3) as (port, db_path, log_path):
            service = tinker.ServiceClient(base_url=f"http://127.0.0.1:{port}/", api_key="tml-dummy")
            sampler = service.create_sampling_client(base_model=BASE_MODEL)
            prompt = tinker_types.ModelInput.from_ints([1, 2, 3])
            futures = [
                sampler.sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker_types.SamplingParams(
                        max_tokens=4,
                        temperature=0.0,
                        top_k=1,
                        seed=index,
                    ),
                )
                for index in range(24)
            ]
            outputs = [future.result(timeout=120) for future in futures]

            assert inference_state.calls == 24
            assert inference_state.max_active == 3
            assert all(len(output.sequences) == 1 for output in outputs)
            assert all(len(output.sequences[0].tokens) == 4 for output in outputs)

            db_engine = create_engine(f"sqlite:///{db_path}")
            with Session(db_engine) as session:
                rows = session.exec(select(FutureDB).order_by(FutureDB.request_id)).all()
            status_counts = Counter(row.status for row in rows)
            assert status_counts == {RequestStatus.COMPLETED: 24}, log_path.read_text()
            assert all(row.request_type.value == "external" for row in rows)
    finally:
        inference_server.shutdown()
        inference_server.server_close()
        inference_thread.join(timeout=5)
