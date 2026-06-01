"""SDK-driven tests for session-aware routing of Tinker sample requests.

Drives the real Tinker SDK against a CPU API server (jax backend, tiny model)
and checks that the SDK-issued sampling_session_id + seq_id become the routing
key (X-Session-ID) on both sample dispatch paths:

- default: ``asample`` persists the ids on ``SampleInput`` and
  ``prepare_sample_batch`` derives the key that ``_sample_with_remote_client``
  forwards to the inference engine.
- external: the forwarded ``/v1/completions`` request carries the
  ``X-Session-ID`` header.

Run:
  uv run --extra dev --extra jax --extra tinker pytest tests/tinker/test_sample_session_routing.py
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_sample_batch  # noqa: E402
from tests.tinker.test_api import start_api_server  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"


def _start_jax_api_server(port: int, db_path: str, extra_overrides: dict[str, str] | None = None):
    """CPU Tinker API server (jax backend) on the given port and DB."""
    overrides = {"port": str(port), "backend": "jax"}
    if extra_overrides:
        overrides.update(extra_overrides)
    return start_api_server(
        overrides=overrides,
        extras=("tinker", "jax"),
        db_path=db_path,
        wait_for_up=True,
        teardown_timeout=15.0,
    )


@contextmanager
def _fake_completions_server():
    """A minimal /v1/completions server that records the X-Session-ID header."""
    captured_session_ids: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            captured_session_ids.append(self.headers.get("X-Session-ID"))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "token_ids": [1, 2, 3, 4],
                            "logprobs": {"token_logprobs": [0.0, 0.0, 0.0, 0.0]},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], captured_session_ids
    finally:
        server.shutdown()
        server.server_close()


def _make_prompt():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return tinker_types.ModelInput.from_ints(tokenizer.encode("Hello", add_special_tokens=True))


def _sample_once(port: int):
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{port}/", api_key=TINKER_API_KEY)
    sampler = sc.create_sampling_client(base_model=BASE_MODEL)
    sampler.sample(
        prompt=_make_prompt(),
        sampling_params=tinker_types.SamplingParams(temperature=0.0, max_tokens=4, seed=0),
        num_samples=1,
    ).result()


def _latest_sample_input(db_path: str) -> types.SampleInput:
    """Return the most recently submitted SAMPLE future's persisted SampleInput."""
    from sqlmodel import Session, create_engine, select

    from skyrl.tinker.db_models import FutureDB

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    try:
        with Session(engine) as session:
            stmt = (
                select(FutureDB)
                .where(FutureDB.request_type == types.RequestType.SAMPLE)
                .order_by(FutureDB.request_id.desc())
            )
            row = session.exec(stmt).first()
            assert row is not None, "no SAMPLE future was persisted"
            return types.SampleInput.model_validate(row.request_data)
    finally:
        engine.dispose()


def test_in_process_threads_session_ids_to_sample_input():
    """Default path: asample persists the SDK's sampling_session_id + seq_id,
    and prepare_sample_batch derives the routing key handed to the engine."""
    port = 8021
    with tempfile.TemporaryDirectory() as db_dir:
        db_path = os.path.join(db_dir, "server.db")
        with _start_jax_api_server(port, db_path):
            _sample_once(port)
        sample_input = _latest_sample_input(db_path)

    # First .sample() on a fresh sampling client → seq_id 0, with the
    # server-assigned sampling session id.
    assert sample_input.seq_id == 0
    assert sample_input.sampling_session_id is not None
    assert sample_input.sampling_session_id.startswith("sampling_")

    batch = prepare_sample_batch({"req": ("", sample_input)})
    assert batch.all_session_ids == [f"{sample_input.sampling_session_id}:0"]


def test_external_forwards_session_header():
    """External path: the forwarded /v1/completions request carries
    X-Session-ID derived from the SDK's sampling_session_id + seq_id."""
    port = 8022
    with _fake_completions_server() as (fake_port, captured_session_ids):
        with tempfile.TemporaryDirectory() as db_dir:
            db_path = os.path.join(db_dir, "server.db")
            overrides = {"external-inference-url": f"http://127.0.0.1:{fake_port}"}
            with _start_jax_api_server(port, db_path, overrides):
                _sample_once(port)

    assert len(captured_session_ids) >= 1, "fake inference server received no request"
    assert any(
        s is not None and s.startswith("sampling_") and s.endswith(":0") for s in captured_session_ids
    ), captured_session_ids
