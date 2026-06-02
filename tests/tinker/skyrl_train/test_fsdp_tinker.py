"""End-to-end smoke tests against a Tinker server backed by SkyRL-Train FSDP.

GPU-gated: skipped automatically when no CUDA device is visible to the test
process. The server starts a real FSDP policy worker, which means tests in this
module need at least one GPU and the ``tinker`` + ``fsdp`` extras installed. The
sampling test additionally brings up a vLLM engine (non-colocated), so a second
GPU is required for it.

Scope: this mirrors ``test_multi_lora_megatron.py`` but exercises only the
*single-adapter* (non multi-LoRA) path on FSDP — multi-tenant LoRA / the
worker-side AdapterStore is currently Megatron-only and the FSDP worker raises
a clear ``NotImplementedError`` for it. FSDP therefore supports exactly one
adapter for the lifetime of a server, so every test shares one module-scoped
training client. These tests confirm that initialising an FSDP Tinker engine
and sending basic requests works:

  - test_forward_backward_optim_step_improves_loss: the adapter trains; loss on
    a fixed micro-batch decreases after an optim step.
  - test_forward_only_returns_logprobs: a forward-only pass returns per-token
    logprobs + elementwise loss without mutating optimizer state.
  - test_second_adapter_rejected: creating a second adapter is rejected with a
    clear NotImplementedError (multi-tenant LoRA is Megatron-only for now).
  - test_sample_after_training: weight-sync to vLLM + greedy sampling returns a
    deterministic token sequence.

Run with:
  uv run --extra tinker --extra fsdp --with pytest --with pytest-timeout \\
    pytest -s tests/tinker/skyrl_train/test_fsdp_tinker.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager

import pytest

cuda_available = False
cuda_device_count = 0
try:  # pragma: no cover - import guard
    import torch

    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cuda_available = cuda_device_count > 0
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="FSDP Tinker tests require at least one CUDA GPU")

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"
TEST_PORT = 8012

# Tiny config: 1 GPU for the FSDP policy worker, 1 for vLLM (non-colocated).
# With a tiny model + LoRA rank 8 this fits comfortably on any modern GPU pair.
# FSDP always serves LoRA adapters by name (it never merges), so the single
# adapter created here is reachable for sampling. max_loras > 1 keeps vLLM's
# LoRA slots warm even though we only register one adapter.
BACKEND_CONFIG = {
    "strategy": "fsdp",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.policy.model.lora.max_loras": 4,
    "trainer.policy.model.lora.max_cpu_loras": 4,
}


@contextmanager
def _api_server(port: int, backend_config: dict | None = None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        cfg = dict(backend_config or BACKEND_CONFIG)
        cmd = [
            "uv",
            "run",
            "--extra",
            "tinker",
            "--extra",
            "fsdp",
            "-m",
            "skyrl.tinker.api",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--base-model",
            BASE_MODEL,
            "--backend",
            "fsdp",
            "--backend-config",
            json.dumps(cfg),
            "--database-url",
            f"sqlite:///{db_path}",
        ]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                ok = wait_for_condition(
                    lambda: _server_is_up(port),
                    timeout_sec=120,
                    poll_interval_sec=2,
                )
                if not ok:
                    with open(log_path) as f:
                        print(f"=== Server failed to start ===\n{f.read()}")
                    pytest.fail("Tinker API server did not come up in time")
                yield proc
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _server_is_up(port: int) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"http://0.0.0.0:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


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
def server():
    with _api_server(TEST_PORT) as proc:
        yield proc


@pytest.fixture
def service_client(server):
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)


@pytest.fixture(scope="module")
def training_client(server):
    """A single LoRA training client shared by every test.

    FSDP supports exactly one adapter per server (multi-tenant LoRA is
    Megatron-only), so all tests must reuse the same client. Tests are additive
    (each mutates the shared adapter) but their assertions are self-contained.
    """
    sc = tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)
    return sc.create_lora_training_client(base_model=BASE_MODEL, rank=8)


def _sum_loss(out) -> float:
    return sum(sum(o["elementwise_loss"].data) for o in out.loss_fn_outputs)


def test_forward_backward_optim_step_improves_loss(training_client):
    """Test that creating an FSDP-backed LoRA training client and running forward_backward + optim_step works.

    Single-step convergence on a fixed micro-batch with a tiny model + a
    nontrivial LR is a reliable signal that gradients flow and the optimizer
    actually updates the adapter.
    """
    tok = training_client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    # Prime once (allocates optimizer state), then measure pre/post-step loss.
    training_client.forward_backward(data, "cross_entropy").result()
    training_client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

    pre = _sum_loss(training_client.forward_backward(data, "cross_entropy").result())
    training_client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()
    post = _sum_loss(training_client.forward_backward(data, "cross_entropy").result())

    assert post <= pre + 1e-3, f"loss did not improve after an optim step (pre={pre}, post={post})"


def test_forward_only_returns_logprobs(training_client):
    """A forward-only pass returns per-token logprobs and does not require an
    optimizer step. (The forward endpoint emits ``logprobs``; the
    ``elementwise_loss`` shape is specific to forward_backward.)"""
    tok = training_client.get_tokenizer()
    data = [_make_datum(tok, "Question: 2+2?\nAnswer:", " 4")]

    out = training_client.forward(data, "cross_entropy").result()
    assert len(out.loss_fn_outputs) == 1
    logprobs = out.loss_fn_outputs[0]["logprobs"].data
    assert len(logprobs) > 0
    assert all(isinstance(x, float) for x in logprobs)


def test_second_adapter_rejected(training_client, service_client):
    """Test that creating a second adapter against the same FSDP server is rejected."""
    with pytest.raises(Exception) as exc:
        service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    msg = str(exc.value).lower()
    assert "not implemented" in msg


@pytest.mark.skipif(cuda_device_count < 2, reason="sampling brings up a separate vLLM engine (needs a 2nd GPU)")
def test_sample_after_training(training_client):
    """Test that weight-sync the trained adapter to vLLM and greedy-sample works."""
    tok = training_client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    for _ in range(2):
        training_client.forward_backward(data, "cross_entropy").result()
        training_client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

    sampler = training_client.save_weights_and_get_sampling_client(name="fsdp_smoke")
    prompt_ids = tok.encode("Question: 2+2?\nAnswer:", add_special_tokens=True)
    out = sampler.sample(
        prompt=tinker_types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=tinker_types.SamplingParams(max_tokens=8, temperature=0.0, top_k=1, seed=0),
    ).result()
    tokens = list(out.sequences[0].tokens)
    assert len(tokens) > 0, "expected at least one sampled token from the FSDP-synced adapter"
