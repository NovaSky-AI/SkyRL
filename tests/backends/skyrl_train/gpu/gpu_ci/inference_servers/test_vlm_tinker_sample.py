"""
GPU end-to-end test: VLM sampling via tinker API with RemoteInferenceClient.

Full stack exercised:
  tinker API (--backend fsdp, _SKYRL_USE_NEW_INFERENCE=1)
    → TinkerEngine.process_sample()
    → prepare_sample_batch()  (no crash on multimodal after to_token_list() fix)
    → SkyRLTrainBackend.sample()
    → RemoteInferenceClient.sample()
    → HTTP POST /sample (router → VLLMServerActor)
    → _assemble_tokens_from_chunks (renders image via /v1/chat/completions/render)
    → /inference/v1/generate
    → decoded output contains "red"

GPU Requirements: 2 GPUs (1 for FSDP policy worker, 1 for vLLM server)

Run with:
uv run --isolated --extra dev --extra fsdp pytest \
  tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vlm_tinker_sample.py \
  -m vllm -v
"""

import io
import json
import os
import subprocess
import tempfile
import time

import pytest
import requests
from transformers import AutoTokenizer

import tinker
import tinker.types as ttypes

from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
TINKER_API_KEY = "tml-dummy"
TINKER_PORT = 9100
TINKER_BASE_URL = f"http://0.0.0.0:{TINKER_PORT}"
SERVER_STARTUP_TIMEOUT = 300  # seconds


def _get_test_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.run_engines_locally = True
    cfg.trainer.placement.colocate_all = False
    return cfg


def _wait_for_tinker_api(base_url: str, timeout: int = SERVER_STARTUP_TIMEOUT) -> None:
    """Poll /health until the tinker API is up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.ok:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Tinker API at {base_url} did not come up within {timeout}s")


@pytest.fixture(scope="module")
def vllm_server(module_scoped_ray_init_fixture):
    """Start vLLM server + router, then tinker API subprocess wired to the same servers."""
    cfg = _get_test_config()

    engines = InferenceEngineState.create(
        cfg=cfg,
        model=MODEL,
        use_local=True,
        backend="vllm",
        use_new_inference_servers=True,
        sleep_level=1,
    )

    proxy_url = engines.client.proxy_url
    server_urls = engines.client.server_urls

    backend_config = json.dumps(
        {
            "generator.inference_engine.external_proxy_url": proxy_url,
            "generator.inference_engine.external_server_urls": server_urls,
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "tinker.log")
        db_path = os.path.join(tmp_dir, "tinker.db")

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
            str(TINKER_PORT),
            "--base-model",
            MODEL,
            "--backend",
            "fsdp",
            "--backend-config",
            backend_config,
            "--database-url",
            f"sqlite:///{db_path}",
        ]

        env = os.environ.copy()
        env["_SKYRL_USE_NEW_INFERENCE"] = "1"

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
            try:
                _wait_for_tinker_api(TINKER_BASE_URL)
                yield engines, proxy_url, server_urls
            except Exception:
                with open(log_path) as f:
                    print(f"=== Test failed. Tinker log ===\n{f.read()}")
                raise
            finally:
                process.terminate()
                process.wait(timeout=10)

    engines.close()


@pytest.mark.vllm
def test_vlm_sample_red_square(vllm_server):
    """
    VLM e2e: tinker API + RemoteInferenceClient samples from a red image prompt.

    Builds a ModelInput with an ImageChunk (red 100x100 PNG) plus an
    EncodedTextChunk for the question text.  After weight sync via
    save_weights_and_get_sampling_client(), samples 1 completion and
    asserts "red" appears in the decoded output.
    """
    import numpy as np
    from PIL import Image

    # --- 1. Create a red 100x100 PNG in memory ---
    arr = np.full((100, 100, 3), fill_value=[255, 0, 0], dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # --- 2. Tokenize the text question ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    question = "What color is this image? Answer with one word."
    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    # --- 3. Build ModelInput with ImageChunk + EncodedTextChunk ---
    model_input = ttypes.ModelInput(
        chunks=[
            ttypes.ImageChunk(data=img_bytes, format="png"),
            ttypes.EncodedTextChunk(tokens=question_tokens),
        ]
    )

    # --- 4. Create tinker client + sync weights ---
    service_client = tinker.ServiceClient(base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY)
    training_client = service_client.create_lora_training_client(base_model=MODEL)

    # Sync policy weights to vLLM server and get a sampling client
    sampling_client = training_client.save_weights_and_get_sampling_client()

    # --- 5. Sample from the VLM ---
    sampling_params = ttypes.SamplingParams(
        max_tokens=32,
        temperature=0.0,
        seed=42,
    )
    result = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    # --- 6. Decode and assert "red" in output ---
    assert len(result.sequences) == 1, f"Expected 1 sequence, got {len(result.sequences)}"
    decoded = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    assert "red" in decoded.lower(), f"Expected 'red' in decoded output, got: {decoded!r}"
