"""
GPU CI test for the reserved eval group's load-from-path route.

One vLLM engine (TP=1) launched the way the trainer launches inference servers (so SkyRL's worker
extension is loaded), then ``RemoteInferenceClient.load_weights_from_path`` against a perturbed
copy of the checkpoint and against the original:
    - every worker attests the version it was asked to stamp
    - a fixed prompt's greedy output changes after the perturbed load and is restored by the
      original load (the prompt is long enough to be prefix-cached, so a stale cache would show)
    - ``paths_exist`` answers per worker, for a present and an absent path

Run:
    uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_load_weights_from_path.py -v -s
"""

import asyncio
import shutil
from pathlib import Path

import httpx
import pytest
from safetensors.torch import load_file, save_file

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.setup import create_inference_servers
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
    resolve_checkpoint_path,
)
from skyrl.train.config.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer

MODEL = "Qwen/Qwen3-0.6B"
# Long enough to occupy several prefix-cache blocks: a stale cache after a reload would surface here.
PROMPT = "The quick brown fox jumps over the lazy dog. " * 12 + "Question: what is 2 + 2? Answer:"


def _perturbed_copy(src: Path, dst: Path) -> Path:
    """A copy of the checkpoint whose token embeddings are negated -- unmistakably different weights."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.safetensors"))
    for shard in sorted(src.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for name in list(tensors):
            if name.endswith("embed_tokens.weight"):
                tensors[name] = -tensors[name]
        save_file(tensors, str(dst / shard.name), metadata={"format": "pt"})
    return dst


def _greedy(proxy_url: str, model: str) -> str:
    resp = httpx.post(
        f"{proxy_url}/v1/completions",
        json={"model": model, "prompt": PROMPT, "max_tokens": 16, "temperature": 0.0},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


@pytest.fixture(scope="class")
def reserved_group(class_scoped_ray_init_fixture, tmp_path_factory):
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            f"trainer.policy.model.path={MODEL}",
            "trainer.placement.colocate_all=false",
            "trainer.logger=console",
            "generator.inference_engine.num_engines=1",
            "generator.inference_engine.tensor_parallel_size=1",
            "generator.inference_engine.gpu_memory_utilization=0.5",
            "generator.inference_engine.enforce_eager=true",
            "generator.inference_engine.max_num_seqs=8",
        ]
    )
    ie_cfg = cfg.generator.inference_engine
    setup = create_inference_servers(
        ie_cfg, build_vllm_cli_args(cfg), log_path=str(tmp_path_factory.mktemp("logs")), placement_group=None
    )
    client = RemoteInferenceClient(
        proxy_url=setup.proxy_url,
        server_urls=setup.server_urls,
        model_name=MODEL,
        data_parallel_size=1,
        tokenizer=get_tokenizer(MODEL, trust_remote_code=True),
    )
    original = resolve_checkpoint_path(MODEL)
    perturbed = _perturbed_copy(original, tmp_path_factory.mktemp("ckpt") / "perturbed")
    cache_root = tmp_path_factory.mktemp("cache")
    try:
        yield setup, client, str(original), str(perturbed), str(cache_root)
    finally:
        asyncio.run(client.teardown())
        for group in setup.server_groups:
            group.shutdown()
        if setup.router is not None:
            setup.router.shutdown()


class TestLoadWeightsFromPath:
    def test_reload_changes_the_output_and_the_original_restores_it(self, reserved_group):
        setup, client, original, perturbed, cache_root = reserved_group
        n_workers = len(setup.server_urls)  # TP=1, DP=1: one worker per server
        baseline = _greedy(setup.proxy_url, MODEL)

        versions = asyncio.run(
            client.load_weights_from_path(perturbed, weight_version="7", cache_dir=f"{cache_root}/global_step_7")
        )
        asyncio.run(client.reset_prefix_cache())
        assert versions == ["7"] * n_workers
        assert _greedy(setup.proxy_url, MODEL) != baseline

        versions = asyncio.run(
            client.load_weights_from_path(original, weight_version="8", cache_dir=f"{cache_root}/global_step_8")
        )
        asyncio.run(client.reset_prefix_cache())
        assert versions == ["8"] * n_workers
        assert _greedy(setup.proxy_url, MODEL) == baseline

    def test_paths_exist_answers_per_worker(self, reserved_group):
        setup, client, _, perturbed, _ = reserved_group
        n_workers = len(setup.server_urls)

        assert asyncio.run(client.paths_exist(perturbed)) == [True] * n_workers
        assert asyncio.run(client.paths_exist("/definitely/not/a/path")) == [False] * n_workers
