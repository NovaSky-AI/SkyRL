"""
Integration test for PD (Prefill-Decode) routing verification.

Verifies that the vllm-router correctly routes prefill requests to prefill
servers and decode requests to decode servers by inspecting router debug logs.

Run:
    uv run --isolated --extra dev --extra fsdp pytest \
        tests/backends/skyrl_train/gpu/gpu_ci/integrations/test_pd_routing.py -v -s
"""

import re
import time

import httpx
import pytest

from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ANSI escape code pattern for stripping colored terminal output
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# The Rust vllm-router emits lines like:
#   vLLM Stage 1 - Prefill: http://<ip>:<port>/v1/completions with request_id: ...
#   vLLM Stage 2 - Decode:  http://<ip>:<port>/v1/completions with request_id: ...
_STAGE1_URL_RE = re.compile(r"Stage 1 - Prefill:\s+(https?://[^\s/]+)")
_STAGE2_URL_RE = re.compile(r"Stage 2 - Decode:\s+(https?://[^\s/]+)")


def get_test_actor_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 64
    cfg.generator.sampling_params.min_p = 0.0
    # Enable debug logging on the router so we can parse Stage 1/Stage 2 lines
    cfg.generator.inference_engine.router_init_kwargs = {"log_level": "debug"}
    return cfg


def test_pd_routing_verification(ray_init_fixture):
    """Verify that the router sends prefill traffic to prefill servers and decode traffic to decode servers.

    Setup: 1P1D (2 engines, TP=1) with NixlConnector.
    Sends a single prompt via httpx.post to the router's /v1/completions endpoint,
    then parses the router log file to verify correct Stage 1 (prefill) and Stage 2 (decode) routing.
    """
    cfg = get_test_actor_config()

    with InferenceEngineState.create(
        cfg,
        tp_size=1,
        num_inference_engines=2,
        enable_pd=True,
        num_prefill=1,
        use_new_inference_servers=True,
        engine_init_kwargs={
            "kv_transfer_config": {
                "kv_connector": "NixlConnector",
            },
        },
    ) as engines:
        # -- Extract ground-truth URLs from router args --
        router_args = engines.router._router_args
        # prefill_urls is List[Tuple[str, Optional[int]]]
        prefill_urls = [url for url, _ in router_args.prefill_urls]
        # decode_urls is List[str]
        decode_urls = list(router_args.decode_urls)

        assert len(prefill_urls) > 0, "Expected at least one prefill URL"
        assert len(decode_urls) > 0, "Expected at least one decode URL"

        # -- Check log file availability --
        log_file = engines.router._log_file
        if log_file is None:
            pytest.skip("Router log file not available")

        # -- Send one prompt through the router --
        router_url = engines.client.proxy_url
        payload = {
            "model": MODEL,
            "prompt": "What is 2 + 2?",
            "max_tokens": 32,
            "temperature": 0.0,
        }

        with httpx.Client(timeout=httpx.Timeout(120.0)) as http_client:
            resp = http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200, f"Router returned status {resp.status_code}: {resp.text}"

        # Give the router a moment to flush logs
        time.sleep(2)

        # -- Parse router log (strip ANSI codes first) --
        with open(log_file, "r") as f:
            raw_log = f.read()
        clean_log = _ANSI_ESCAPE.sub("", raw_log)

        stage1_lines = []
        stage1_target_urls = []
        stage2_lines = []
        stage2_target_urls = []

        for line in clean_log.splitlines():
            m1 = _STAGE1_URL_RE.search(line)
            if m1:
                stage1_lines.append(line)
                stage1_target_urls.append(m1.group(1))

            m2 = _STAGE2_URL_RE.search(line)
            if m2:
                stage2_lines.append(line)
                stage2_target_urls.append(m2.group(1))

        # -- Assertions --

        # At least one Stage 1 line exists and its target URL is a prefill URL
        assert len(stage1_lines) > 0, (
            f"Expected at least one Stage 1 log line but found none.\n"
            f"Log file: {log_file}\n"
            f"Log contents (last 2000 chars):\n{clean_log[-2000:]}"
        )
        for target_url, line in zip(stage1_target_urls, stage1_lines):
            assert any(target_url == purl or target_url in purl or purl in target_url for purl in prefill_urls), (
                f"Stage 1 target URL does not match any prefill URL.\n"
                f"Target URL: {target_url}\n"
                f"Prefill URLs: {prefill_urls}\n"
                f"Line: {line}"
            )

        # At least one Stage 2 line exists and its target URL is a decode URL
        assert len(stage2_lines) > 0, (
            f"Expected at least one Stage 2 log line but found none.\n"
            f"Log file: {log_file}\n"
            f"Log contents (last 2000 chars):\n{clean_log[-2000:]}"
        )
        for target_url, line in zip(stage2_target_urls, stage2_lines):
            assert any(target_url == durl or target_url in durl or durl in target_url for durl in decode_urls), (
                f"Stage 2 target URL does not match any decode URL.\n"
                f"Target URL: {target_url}\n"
                f"Decode URLs: {decode_urls}\n"
                f"Line: {line}"
            )

        # No Stage 1 target URL references a decode URL
        for target_url, line in zip(stage1_target_urls, stage1_lines):
            for decode_url in decode_urls:
                assert target_url != decode_url and decode_url not in target_url and target_url not in decode_url, (
                    f"Stage 1 (prefill) target URL unexpectedly matches a decode URL.\n"
                    f"Target URL: {target_url}\n"
                    f"Decode URL: {decode_url}\n"
                    f"Line: {line}"
                )

        # No Stage 2 target URL references a prefill URL
        for target_url, line in zip(stage2_target_urls, stage2_lines):
            for prefill_url in prefill_urls:
                assert target_url != prefill_url and prefill_url not in target_url and target_url not in prefill_url, (
                    f"Stage 2 (decode) target URL unexpectedly matches a prefill URL.\n"
                    f"Target URL: {target_url}\n"
                    f"Prefill URL: {prefill_url}\n"
                    f"Line: {line}"
                )
