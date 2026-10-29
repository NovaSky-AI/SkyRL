"""GPU test for ``patch_routed_experts_pp_relay``: routed-experts capture (R3) with inference PP > 1.

The same prompts are generated greedily, one request at a time, by an engine at PP=1 and one at
PP=2, both with ``enable_return_routed_experts``. Each PP stage captures only its own MoE layers,
so without the relay the first stage's layers come back as expert 0.

Both engines use the Triton MoE backend: FlashInfer's MoE autotunes kernel tactics per engine, so
two engines can route near-tie tokens differently.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/patches/vllm/test_routed_experts_pp_relay.py -v -s
"""

import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInput
from skyrl.train.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState, get_test_prompts

MODEL = "allenai/OLMoE-1B-7B-0924"
NUM_LAYERS = 16
TOP_K = 8
NUM_PROMPTS = 4
MAX_TOKENS = 32


def _cfg(pipeline_parallel_size: int, enforce_eager: bool) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = pipeline_parallel_size
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.num_engines = 1
    ie_cfg.run_engines_locally = True
    ie_cfg.tensor_parallel_size = 1
    ie_cfg.pipeline_parallel_size = pipeline_parallel_size
    ie_cfg.distributed_executor_backend = "mp"
    ie_cfg.enable_return_routed_experts = True
    ie_cfg.enforce_eager = enforce_eager
    return cfg


@pytest.fixture(scope="module")
def prompt_token_ids():
    # The OLMoE base model has no chat template: tokenize the user turns as plain text.
    tokenizer = get_tokenizer(MODEL)
    return [
        tokenizer.encode(conversation[-1]["content"])
        for conversation in get_test_prompts(MODEL, num_samples=NUM_PROMPTS)
    ]


async def _generate(cfg: SkyRLTrainConfig, prompt_token_ids) -> list:
    """Greedy response ids and routed experts per prompt."""
    results = []
    # The PP=1 engine holds all ~12.9 GiB of OLMoE weights on one GPU: on a 24 GB L4, 0.6 leaves no
    # room for the KV cache ("No available memory for the cache blocks").
    async with InferenceEngineState.create(
        cfg=cfg, sleep_level=1, gpu_memory_utilization=0.85, engine_init_kwargs={"moe_backend": "triton"}
    ) as engines:
        for ids in prompt_token_ids:
            output = await engines.client.generate(
                InferenceEngineInput(
                    prompt_token_ids=[ids], sampling_params={"temperature": 0.0, "max_tokens": MAX_TOKENS}
                )
            )
            results.append((output["response_ids"][0], np.asarray(output["rollout_expert_indices"][0])))
    return results


def _matching_rows(pp1, pp2) -> list:
    """Top-k sets of the rows both runs computed from the same tokens: the prompt, then the response
    up to the first greedy divergence."""
    pairs = []
    for (pp1_ids, pp1_experts), (pp2_ids, pp2_experts) in zip(pp1, pp2):
        same = next((i for i, (a, b) in enumerate(zip(pp1_ids, pp2_ids)) if a != b), min(len(pp1_ids), len(pp2_ids)))
        rows = len(pp1_experts) - len(pp1_ids) + same
        pairs.append((np.sort(pp1_experts[:rows], axis=-1), np.sort(pp2_experts[:rows], axis=-1)))
    return pairs


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "cuda_graphs"])
async def test_pp2_routes_match_pp1(ray_init_fixture, prompt_token_ids, enforce_eager):
    pp1 = await _generate(_cfg(1, enforce_eager), prompt_token_ids)
    pp2 = await _generate(_cfg(2, enforce_eager), prompt_token_ids)

    for _, experts in pp2:
        assert experts.shape[1:] == (NUM_LAYERS, TOP_K)
        # A layer whose rows are all expert 0 was dropped.
        assert all(experts[:, layer].any() for layer in range(NUM_LAYERS))

    pairs = _matching_rows(pp1, pp2)
    same = np.concatenate([(a == b).all(axis=-1) for a, b in pairs])  # [rows, layers]
    print(f"eager={enforce_eager}: per-layer match rate {np.round(same.mean(axis=0), 3).tolist()}")
    if enforce_eager:
        assert same.all()
    else:
        # With CUDA graphs, PP=1 and PP=2 compile different graphs (torch.compile fuses PP=1's
        # layer 7 -> 8 boundary, which PP=2 splits across stages), so near-tie tokens can route
        # differently on any layer: on an L4 even the relayed layers differ on a few rows. A dropped
        # or misplaced relay matches on almost no rows, so a per-layer rate still catches it.
        assert (same.mean(axis=0) > 0.9).all()
