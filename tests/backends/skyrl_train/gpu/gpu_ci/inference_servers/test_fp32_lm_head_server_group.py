"""
GPU CI integration test: fp32 LM-head patch is picked up by a SkyRL ServerGroup.

Launches a vLLM server via SkyRL's ``ServerGroup`` (Ray actors, exactly like
production) with ``additional_config={"skyrl_enable_fp32_lm_head": True}`` and the
``NewInferenceWorkerWrap`` worker extension. Then it verifies, via the vLLM
dev-mode ``/collective_rpc`` endpoint, that every TP worker:

    1. imported + installed the SkyRL fp32 lm-head patch,
    2. received the ``skyrl_enable_fp32_lm_head`` flag through ``additional_config``,
    3. has the flag set on its model's ``LogitsProcessor`` instance(s).

A control server (flag off) confirms the patch is a no-op when disabled, and a
completion request confirms generation still works with fp32 logits enabled.

Run:
    uv run --isolated --extra dev --extra fsdp pytest \
        tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_fp32_lm_head_server_group.py -m vllm -v -s
"""

import argparse
import time

import httpx
import pytest

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS,
)
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_vllm_cli_args(
    model: str,
    enable_fp32_lm_head: bool,
    tp_size: int = 1,
) -> argparse.Namespace:
    """CLI args for a vLLM server, wired like SkyRL's build_vllm_cli_args.

    Sets the SkyRL worker extension (so the patch module is imported in each
    worker) and, when requested, the fp32 lm-head flag via additional_config.
    """
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vLLM server")
    parser = make_arg_parser(parser)
    args = parser.parse_args(
        [
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.5",
            "--max-model-len",
            "2048",
        ]
    )
    args.worker_extension_cls = VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS
    if enable_fp32_lm_head:
        args.additional_config = {"skyrl_enable_fp32_lm_head": True}
    return args


def wait_for_url(url: str, timeout: float = 240.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(2.0)
    return False


def _collective_rpc(server_url: str, method: str) -> list:
    """Call a worker-extension method on all TP workers via /collective_rpc."""
    resp = httpx.post(f"{server_url}/collective_rpc", json={"method": method}, timeout=60.0)
    assert resp.status_code == 200, f"/collective_rpc failed: {resp.status_code} {resp.text}"
    return resp.json()["results"]


def _start_group(enable_fp32_lm_head: bool) -> ServerGroup:
    cli_args = make_vllm_cli_args(MODEL, enable_fp32_lm_head=enable_fp32_lm_head)
    group = ServerGroup(cli_args=cli_args, num_servers=1, start_port=get_open_port())
    server_infos = group.start()
    for info in server_infos:
        assert wait_for_url(info.url), f"Server {info.url} failed to start"
    return group


@pytest.fixture(scope="class")
def fp32_enabled_group(class_scoped_ray_init_fixture):
    group = _start_group(enable_fp32_lm_head=True)
    yield group
    group.shutdown()


@pytest.mark.vllm
class TestFP32LMHeadServerGroup:
    def test_patch_picked_up_by_workers(self, fp32_enabled_group):
        """Each Ray-actor worker installed the patch and has the flag set."""
        server_url = fp32_enabled_group.server_infos[0].url
        results = _collective_rpc(server_url, "get_fp32_lm_head_status")

        assert len(results) >= 1
        for worker_status in results:
            assert worker_status["patch_installed"] is True
            assert worker_status["additional_config_flag"] is True
            flags = worker_status["logits_processor_flags"]
            assert len(flags) >= 1, "no LogitsProcessor found on the worker model"
            assert all(flags), f"LogitsProcessor fp32 flag not set on all heads: {flags}"

    def test_generation_works_with_fp32(self, fp32_enabled_group):
        """Generation still succeeds with fp32 logits enabled."""
        server_url = fp32_enabled_group.server_infos[0].url
        payload = {
            "model": MODEL,
            "prompt": "What is 2 + 2? Answer:",
            "max_tokens": 16,
            "temperature": 0.0,
        }
        resp = httpx.post(f"{server_url}/v1/completions", json=payload, timeout=60.0)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["choices"][0]["text"] is not None


@pytest.fixture(scope="class")
def fp32_disabled_group(class_scoped_ray_init_fixture):
    group = _start_group(enable_fp32_lm_head=False)
    yield group
    group.shutdown()


@pytest.mark.vllm
class TestFP32LMHeadDisabled:
    def test_patch_noop_when_disabled(self, fp32_disabled_group):
        """Patch is installed but a strict no-op when the flag is not set."""
        server_url = fp32_disabled_group.server_infos[0].url
        results = _collective_rpc(server_url, "get_fp32_lm_head_status")

        assert len(results) >= 1
        for worker_status in results:
            # The patch module is always imported via the worker extension...
            assert worker_status["patch_installed"] is True
            # ...but the flag must be off, so no LogitsProcessor upcasts.
            assert worker_status["additional_config_flag"] is False
            flags = worker_status["logits_processor_flags"]
            assert len(flags) >= 1
            assert not any(flags), f"fp32 flag unexpectedly set when disabled: {flags}"
