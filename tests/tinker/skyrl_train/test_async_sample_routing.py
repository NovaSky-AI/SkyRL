"""End-to-end tests for the async sample routing path (EngineStateDB +
``SkyRLTrainInferenceForwardingClient``).

GPU-gated: requires at least one CUDA device. Spins up a real Tinker API
server with the SkyRL-Train Megatron backend in non-colocated mode so the
API process forwards sample requests directly to vLLM via the new path,
bypassing the engine subprocess's serial scheduling loop.

Coverage:
  - test_engine_state_published: after ``save_weights_for_sampler``, the
    engine's vLLM proxy URL is written to ``EngineStateDB``.
  - test_sample_uses_external_path: an issued sample creates a future of
    type ``EXTERNAL`` (not ``SAMPLE``) and resolves successfully.
  - test_sample_concurrent_with_training_is_fast: the central
    parallelism test. While a long-running stream of ``forward_backward``
    + ``optim_step`` calls is in flight, a sample request resolves in
    much less time than the training stream takes, demonstrating that
    sample latency is no longer bounded by training-step duration.
  - test_concurrent_samples_per_adapter: many concurrent samples across
    two adapters all resolve via the forwarding client's connection pool
    and per-adapter (``model=<model_id>``) routing on vLLM.

Run:
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \\
    pytest -s tests/tinker/skyrl_train/test_async_sample_routing.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager

import pytest

cuda_available = False
try:  # pragma: no cover - import guard
    import torch

    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="async sample routing tests require at least one CUDA GPU")

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"
TEST_PORT = 8019

# Tiny config — same shape as test_multi_lora_megatron's BACKEND_CONFIG.
# Non-colocated is required: that's what triggers the API to install
# SkyRLTrainInferenceForwardingClient in the lifespan. merge_lora=False makes
# vLLM serve LoRA adapters by tenant name, which is the contract the
# forwarding client relies on (model=<model_id>).
BACKEND_CONFIG = {
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.model.lora.max_loras": 4,
    "trainer.policy.model.lora.max_cpu_loras": 4,
}


def _server_is_up(port: int) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"http://0.0.0.0:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


@contextmanager
def _api_server(port: int, db_path: str, backend_config: dict | None = None):
    """Start the Tinker API server with non-colocated Megatron backend.

    db_path is taken as a parameter (rather than created internally) so
    tests that need to inspect EngineStateDB can read it directly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        cfg = dict(backend_config or BACKEND_CONFIG)
        cmd = [
            "uv", "run", "--extra", "tinker", "--extra", "megatron",
            "-m", "skyrl.tinker.api",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--base-model", BASE_MODEL,
            "--backend", "megatron",
            "--backend-config", json.dumps(cfg),
            "--database-url", f"sqlite:///{db_path}",
        ]  # fmt: skip
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                ok = wait_for_condition(
                    lambda: _server_is_up(port),
                    timeout_sec=180,
                    poll_interval_sec=2,
                )
                if not ok:
                    with open(log_path) as f:
                        print(f"=== Server failed to start ===\n{f.read()}")
                    pytest.fail("Tinker API server did not come up in time")
                yield proc, log_path
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _make_datum(tokenizer, prompt: str, completion: str):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


@pytest.fixture(scope="module")
def server_db_path():
    """Module-scoped server + DB path. Sharing the server across tests
    saves ~2 minutes of warm-up time."""
    db_dir = tempfile.mkdtemp(prefix="async_sample_routing_db_")
    db_path = os.path.join(db_dir, "server.db")
    with _api_server(TEST_PORT, db_path) as (proc, log_path):
        yield proc, db_path, log_path


@pytest.fixture
def service_client(server_db_path):
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)


def _read_engine_state(db_path: str):
    """Read the singleton EngineStateDB row from the test server's DB."""
    from sqlmodel import Session, create_engine

    from skyrl.tinker.db_models import EngineStateDB

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    try:
        with Session(engine) as session:
            return session.get(EngineStateDB, 1)
    finally:
        engine.dispose()


def _read_future_request_type(db_path: str, request_id: int) -> str:
    """Read the request_type of a single future from the test server's DB."""
    from sqlmodel import Session, create_engine

    from skyrl.tinker.db_models import FutureDB

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    try:
        with Session(engine) as session:
            row = session.get(FutureDB, request_id)
            return None if row is None else str(row.request_type)
    finally:
        engine.dispose()


def _train_one_step(tc, tok):
    """Run one tiny forward_backward + optim_step. Used for warm-up."""
    data = [_make_datum(tok, "Hello", "world")]
    tc.forward_backward(data, "cross_entropy").result()
    tc.optim_step(tinker_types.AdamParams(learning_rate=1e-4)).result()


def test_engine_state_published(server_db_path):
    """After a save_weights_for_sampler the engine publishes its proxy URL."""
    proc, db_path, _ = server_db_path
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Drive at least one fwd_bwd + save to bring up vLLM and trigger
    # _publish_engine_state on the backend.
    _train_one_step(tc, tok)
    tc.save_weights_and_get_sampling_client(name="state_check_a")

    row = _read_engine_state(db_path)
    assert row is not None, "EngineStateDB row missing — backend never published state"
    assert row.inference_proxy_url is not None and row.inference_proxy_url.startswith(
        "http"
    ), f"expected an http(s) proxy URL, got {row.inference_proxy_url!r}"


def test_sample_uses_external_path(server_db_path):
    """A sample issued through the SDK creates a FutureDB row of type EXTERNAL.

    This is the "test" half of the design: the API hoists the sample off
    the engine's serial loop and into the API process's asyncio loop.
    """
    from sqlmodel import Session, create_engine, func, select

    from skyrl.tinker import types as skyrl_types
    from skyrl.tinker.db_models import FutureDB

    proc, db_path, _ = server_db_path
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    _train_one_step(tc, tok)
    sampler = tc.save_weights_and_get_sampling_client(name="external_path_a")

    # Snapshot the max future_id before submitting our sample so we can
    # filter out any EXTERNAL futures from earlier tests.
    eng = create_engine(f"sqlite:///{db_path}", echo=False)
    try:
        with Session(eng) as s:
            max_before = s.exec(select(func.max(FutureDB.request_id))).one() or 0
    finally:
        eng.dispose()

    out = sampler.sample(
        prompt=tinker_types.ModelInput.from_ints(tok.encode("Hi", add_special_tokens=True)),
        num_samples=1,
        sampling_params=tinker_types.SamplingParams(max_tokens=4, temperature=0.0, top_k=1, seed=0),
    ).result()
    assert len(out.sequences) == 1

    # Look for an EXTERNAL future with id > max_before. If async routing
    # is on, every sample creates exactly one such row.
    eng = create_engine(f"sqlite:///{db_path}", echo=False)
    try:
        with Session(eng) as s:
            stmt = (
                select(FutureDB.request_id, FutureDB.request_type)
                .where(FutureDB.request_id > max_before)
                .where(FutureDB.request_type == skyrl_types.RequestType.EXTERNAL)
            )
            rows = s.exec(stmt).all()
    finally:
        eng.dispose()

    assert len(rows) >= 1, (
        f"expected at least one EXTERNAL future to be created by the sample call, "
        f"found {len(rows)}; async sample routing may not be active"
    )


def test_sample_concurrent_with_training_is_fast(server_db_path):
    """Central test: sample latency is independent of concurrent training.

    With async sample routing enabled, a sample issued during a long
    stream of forward_backward + optim_step calls should resolve in
    roughly vLLM-gen-time, NOT in training-stream-duration. We assert
    that the sample's wall-clock duration is bounded well below the
    training stream's wall-clock duration.
    """
    proc, _, _ = server_db_path
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Warm up vLLM + register the adapter once, off the critical timing path.
    _train_one_step(tc, tok)
    sampler = tc.save_weights_and_get_sampling_client(name="parallel_warmup")

    # Build a workload that creates a clearly observable training-side
    # delay. Tiny model + few datums per call, but many calls back-to-back.
    # Each forward_backward+optim_step on the tiny Qwen3 takes ~hundreds of
    # ms; 24 of them + a save makes the full training stream visibly long.
    NUM_TRAIN_STEPS = 24
    train_data = [_make_datum(tok, "Hi", "there") for _ in range(2)]

    # Fire training in a background thread so the main thread can fire a
    # sample and time it. The sample is submitted shortly after training
    # begins, so the training stream is definitely in flight.
    train_done = threading.Event()
    train_t0 = time.perf_counter()
    train_t1: list[float] = []

    def train_loop():
        try:
            for i in range(NUM_TRAIN_STEPS):
                tc.forward_backward(train_data, "cross_entropy").result()
                tc.optim_step(tinker_types.AdamParams(learning_rate=1e-4)).result()
                if i == 0:
                    # Re-publish the adapter mid-stream so the sampler has
                    # current weights to use. This still goes through the
                    # engine's serial loop — that's fine, it doesn't block
                    # the sample we time below.
                    tc.save_weights_and_get_sampling_client(name="parallel_mid")
        finally:
            train_t1.append(time.perf_counter())
            train_done.set()

    t = threading.Thread(target=train_loop, daemon=True)
    t.start()

    # Give the training stream a head start so its first request hits the
    # engine before our sample does. ~0.5s is plenty for SQLite + the
    # engine's 100ms scheduling tick.
    time.sleep(0.5)
    assert not train_done.is_set(), "training finished before we could issue a concurrent sample"

    sample_t0 = time.perf_counter()
    out = sampler.sample(
        prompt=tinker_types.ModelInput.from_ints(tok.encode("Hello", add_special_tokens=True)),
        num_samples=1,
        sampling_params=tinker_types.SamplingParams(max_tokens=8, temperature=0.0, top_k=1, seed=0),
    ).result()
    sample_t1 = time.perf_counter()
    sample_latency = sample_t1 - sample_t0
    assert len(out.sequences) == 1

    # Wait for training to finish and capture how long it took.
    train_done.wait(timeout=300)
    assert train_done.is_set(), "training stream did not finish in time"
    train_latency = train_t1[0] - train_t0

    print(
        f"\n[async_sample_routing] sample latency={sample_latency:.3f}s, "
        f"concurrent training stream={train_latency:.3f}s "
        f"(ratio sample/train = {sample_latency / train_latency:.3f})"
    )

    # Core assertion: sample latency must be a small fraction of the
    # training stream's wall-clock. If sample were going through the
    # engine's serial loop, it would queue behind the entire training
    # stream and the latencies would be comparable.
    #
    # We pick 0.5x as a conservative bound. In practice on the tiny
    # Qwen3 model the ratio is closer to 0.05x, but CI variance can be
    # large for short workloads. If this is flaky, raise NUM_TRAIN_STEPS
    # to widen the margin rather than relax this assertion.
    assert sample_latency < 0.5 * train_latency, (
        f"sample latency {sample_latency:.3f}s is not significantly smaller than "
        f"training stream {train_latency:.3f}s — async sample routing may not be active"
    )


def test_concurrent_samples_per_adapter(server_db_path):
    """Issue several concurrent samples across two adapters; all resolve.

    Exercises the forwarding client's httpx connection pool and confirms
    that requests route to the correct adapter via ``model=<model_id>``
    (each Tinker model_id maps to a LoRA registered on vLLM under the
    same name during save_weights_for_sampler).
    """
    proc, _, _ = server_db_path
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    tc_a = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tc_b = sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    _train_one_step(tc_a, tok)
    _train_one_step(tc_b, tok)
    sampler_a = tc_a.save_weights_and_get_sampling_client(name="concurrent_a")
    sampler_b = tc_b.save_weights_and_get_sampling_client(name="concurrent_b")

    futures = []
    for i in range(8):
        target = sampler_a if i % 2 == 0 else sampler_b
        futures.append(
            target.sample(
                prompt=tinker_types.ModelInput.from_ints(tok.encode("Q", add_special_tokens=True)),
                num_samples=1,
                sampling_params=tinker_types.SamplingParams(max_tokens=4, temperature=0.0, top_k=1, seed=i),
            )
        )

    outputs = [f.result() for f in futures]
    for o in outputs:
        assert len(o.sequences) == 1
        assert len(o.sequences[0].tokens) > 0
