"""End-to-end torch profiling against a Tinker API server backed by SkyRL-Train FSDP.

The rest of the profiler suite runs on CPU against synthetic loops. This test is
the only one that shows the profiler attached to real policy-worker GPU work: it
brackets a few optim steps with /start_profiling and /stop_profiling and asserts
the resulting trace contains CUDA kernel events.

Single GPU, tiny model. GPU-gated and skipped without the ``tinker`` SDK, so it
skips on the CPU job that also runs this directory and runs in the Tinker
SkyRL-Train GPU workflow (``ci/gpu_ci_run_tinker_skyrl_train_backend.sh``).

Run with:
  uv run --isolated --extra tinker --extra fsdp --with pytest \\
    pytest -s tests/tinker/skyrl_train/test_tinker_torch_profiler.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import urllib.request
from contextlib import contextmanager

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Profiler E2E test requires a GPU",
)

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"
TEST_PORT = 8013
GLOBAL_STEP = 7

BACKEND_CONFIG = {
    "strategy": "fsdp",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
}


@contextmanager
def _api_server(port: int, export_dir: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        cmd = [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "tinker",
            "--extra",
            "fsdp",
            "--with",
            "ray==2.56.0",
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
            json.dumps(BACKEND_CONFIG),
            "--database-url",
            f"sqlite:///{os.path.join(tmp_dir, 'server.db')}",
            "--torch-profiler",
            json.dumps({"export_dir": export_dir, "ranks": [0]}),
        ]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                if not wait_for_condition(lambda: _server_is_up(port), timeout_sec=180, poll_interval_sec=2):
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

    try:
        urllib.request.urlopen(f"http://0.0.0.0:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


def _post_json(port: int, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://0.0.0.0:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


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


def test_torch_profiler_end_to_end(tmp_path):
    export_dir = str(tmp_path / "traces")

    with _api_server(TEST_PORT, export_dir):
        service_client = tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)
        client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
        tok = client.get_tokenizer()
        data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

        started = _post_json(
            TEST_PORT,
            "/start_profiling",
            {
                "model_id": client.model_id,
                "global_step": GLOBAL_STEP,
                # Capture the very next optim step. The window closes on the
                # second prof.step(), so two optim steps are enough.
                "schedule_options": {"skip_first": 0, "wait": 0, "warmup": 0, "active": 1, "repeat": 1},
                "profile_options": {"activities": ["cpu", "cuda"], "use_gzip": False},
            },
        )
        assert started["active"] is True
        assert started["export_path"] == f"{export_dir}/{GLOBAL_STEP}"

        for _ in range(2):
            client.forward_backward(data, "cross_entropy").result()
            client.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

        stopped = _post_json(TEST_PORT, "/stop_profiling", {"model_id": client.model_id})
        assert stopped["active"] is False

        traces = glob.glob(os.path.join(export_dir, str(GLOBAL_STEP), "*.pt.trace.json"))
        assert traces, f"no trace written under {export_dir}/{GLOBAL_STEP}"

        with open(traces[0]) as fh:
            events = json.load(fh)["traceEvents"]
        assert events, "trace has no events"
        categories = {e.get("cat") for e in events}
        assert {"kernel", "gpu_memcpy", "gpu_memset"} & categories, (
            f"no CUDA activity in the trace, so the profiler was not attached to GPU work; "
            f"categories={sorted(c for c in categories if c)}"
        )
