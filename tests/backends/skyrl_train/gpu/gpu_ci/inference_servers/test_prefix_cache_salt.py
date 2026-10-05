"""
Prefix-cache isolation across weight versions on a real vLLM server.

With ``use_cache_salt``, ``RemoteInferenceClient`` gives a request that carries no ``cache_salt`` the salt
of its current weight version, so after ``increment_weight_version`` (one per weight sync) the request
cannot reuse prefix-cache blocks computed under the previous version. Without the salt it can.
Cached tokens are read from ``usage.prompt_tokens_details.cached_tokens``.

GPU Requirements: 1 GPU.

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_prefix_cache_salt.py -m vllm -v
"""

import pytest

from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = "Qwen/Qwen3-0.6B"
# Spans many 16-token KV blocks, so a prefix hit shows up as a nonzero cached-token count.
SYSTEM_PROMPT = "You are a careful assistant who answers in one word. " * 20


def _cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    ie = cfg.generator.inference_engine
    ie.num_engines = 1
    ie.tensor_parallel_size = 1
    ie.run_engines_locally = True
    ie.enable_prefix_caching = True
    return cfg


async def _cached_tokens(client, route: str) -> int:
    if route == "chat":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Say hi."}]
        response = await client.chat_completion({"json": {"messages": messages, "max_tokens": 1}, "headers": {}})
    else:
        response = await client.completion(
            {"json": {"prompt": SYSTEM_PROMPT + "Say hi.", "max_tokens": 1}, "headers": {}}
        )
    return response["usage"]["prompt_tokens_details"]["cached_tokens"]


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["chat", "completion"])
async def test_unsalted_request_cannot_reuse_blocks_of_the_previous_weight_version(ray_init_fixture, route):
    async with InferenceEngineState.create(
        cfg=_cfg(),
        sleep_level=1,
        gpu_memory_utilization=0.5,
        engine_init_kwargs={"enable_prompt_tokens_details": True},
    ) as engines:
        client = engines.client

        # No salt: a request after a weight sync reuses the blocks computed before it.
        client.use_cache_salt = False
        await _cached_tokens(client, route)
        client.increment_weight_version()
        assert await _cached_tokens(client, route) > 0

        client.use_cache_salt = True
        assert await _cached_tokens(client, route) == 0
        assert await _cached_tokens(client, route) > 0
        client.increment_weight_version()
        assert await _cached_tokens(client, route) == 0
        assert await _cached_tokens(client, route) > 0
