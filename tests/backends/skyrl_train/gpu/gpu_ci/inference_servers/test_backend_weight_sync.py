"""
GPU CI test for weight sync through the SkyRLTrainBackend API (new inference path).

Uses the non-colocated setting (colocate_all=False) with 2 GPUs (TP=1, 2 engines, 2 FSDP2 workers):
    - Backend creates FSDP2 workers with real weights from HF
    - Inference servers start with dummy (random) weights via engine_init_kwargs
    - save_sampler_checkpoint() broadcasts real training weights via NCCL
    - Verified by querying the server before and after sync

Run:
    uv run --isolated --extra dev --extra fsdp pytest \
        tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_backend_weight_sync.py -v -s
"""

import os
import time
from unittest import mock

import httpx
import pytest
import ray
from functools import lru_cache
from loguru import logger

from skyrl.train.utils.utils import peer_access_supported
from skyrl.env_vars import SKYRL_PYTHONPATH_EXPORT


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


def wait_for_url(url: str, timeout: float = 180.0) -> bool:
    """Wait for a URL to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(2.0)
    return False


@pytest.fixture(scope="class")
def ray_env_with_new_inference():
    """Ray init fixture with _SKYRL_USE_NEW_INFERENCE=1 in runtime env."""
    if ray.is_initialized():
        ray.shutdown()

    env_vars = {
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "_SKYRL_USE_NEW_INFERENCE": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars.update(
            {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
            }
        )

    if SKYRL_PYTHONPATH_EXPORT:
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath is None:
            raise RuntimeError("SKYRL_PYTHONPATH_EXPORT is set but PYTHONPATH is not defined in environment")
        env_vars["PYTHONPATH"] = pythonpath

    logger.info(f"Initializing Ray with environment variables: {env_vars}")
    ray.init(runtime_env={"env_vars": env_vars})

    yield

    ray.shutdown()


@pytest.mark.asyncio(loop_scope="class")
class TestBackendWeightSync:
    """Test weight sync through SkyRLTrainBackend with new inference path (non-colocated)."""

    async def test_backend_weight_sync_non_colocated(self, ray_env_with_new_inference):
        """
        End-to-end non-colocated weight sync test via SkyRLTrainBackend:

        1. Create backend with 2 FSDP2 workers (real weights from HF)
        2. Start 2 inference servers with dummy (random) weights
        3. Verify dummy weights produce gibberish
        4. Run save_sampler_checkpoint() to broadcast real weights via NCCL
        5. Verify real weights produce correct output
        """
        from skyrl.backends.skyrl_train_backend import (
            SkyRLTrainBackend,
            FSDPBackendOverrides,
        )
        from skyrl.tinker.types import LoraConfig

        # ===== Step 1: Create backend =====
        overrides = {
            "trainer.placement.colocate_all": False,
            "trainer.placement.policy_num_gpus_per_node": 2,
            "trainer.placement.policy_num_nodes": 1,
            "trainer.logger": "console",
            "generator.inference_engine.tensor_parallel_size": 1,
            "generator.inference_engine.num_engines": 2,
            "generator.inference_engine.gpu_memory_utilization": 0.5,
            "generator.inference_engine.async_engine": True,
        }
        backend = SkyRLTrainBackend(MODEL, FSDPBackendOverrides(**overrides))

        # ===== Step 2: Create model (real weights, FSDP2 sharded across 2 GPUs) =====
        model_id = "test-model"
        backend.create_model(model_id, LoraConfig(rank=0, alpha=0, seed=42))

        # ===== Step 3: Inject dummy weight config before inference engine creation =====
        backend._cfg.generator.inference_engine.engine_init_kwargs = {"load_format": "dummy"}

        # ===== Step 4: Create inference engines with dummy weights =====
        with mock.patch("skyrl.backends.skyrl_train_backend._SKYRL_USE_NEW_INFERENCE", True):
            backend._ensure_inference_engines()

        # Wait for servers to be healthy
        server_urls = backend._server_group.get_server_urls()
        assert len(server_urls) == 2, f"Expected 2 server URLs, got {len(server_urls)}"
        for url in server_urls:
            assert wait_for_url(url), f"Server {url} failed to start"

        try:
            # ===== Step 5: Verify dummy weights produce gibberish =====
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
                resp = await http_client.post(f"{server_urls[0]}/v1/completions", json=payload)
                assert resp.status_code == 200, f"Completions request failed: {resp.text}"

                text_before = resp.json()["choices"][0]["text"]
                assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 6: Sleep inference engines (required before weight sync) =====
            await backend._inference_engine_client.sleep()

            # ===== Step 7: Sync weights via save_sampler_checkpoint =====
            with mock.patch("skyrl.backends.skyrl_train_backend._SKYRL_USE_NEW_INFERENCE", True):
                backend._validate_model_state(model_id)
                backend._ensure_inference_engines()
                await backend._dispatch.save_weights_for_sampler()

            # ===== Step 8: Verify real weights produce correct output =====
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
                resp = await http_client.post(f"{server_urls[0]}/v1/completions", json=payload)
                assert resp.status_code == 200, f"Completions request failed: {resp.text}"

                text_after = resp.json()["choices"][0]["text"]
                assert "Paris" in text_after, f"Weight sync failed - expected 'Paris' but got: {text_after!r}"

        finally:
            # Cleanup: teardown inference client session
            if backend._inference_engine_client is not None:
                await backend._inference_engine_client.teardown()
