"""Megatron MTP weight-sync round trip through vLLM's native draft session.

The engine sleeps at level 2 before the sync, which discards every weight --
the drafter's included. Draft acceptance after the sync is therefore only
healthy if the draft session reloaded the drafter from the policy's MTP head.

Run with::
    uv run --isolated --extra dev --extra megatron pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_weight_sync.py
"""

import re
from typing import List, Tuple

import httpx
import pytest
import ray

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import DeltaWeightSyncConfig, SkyRLTrainConfig
from skyrl.train.config.config import SamplingParams
from skyrl.train.utils.utils import validate_cfg
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    are_responses_similar,
    init_worker_with_type,
    run_inference,
)

MODEL_NAME = "Qwen/Qwen3.5-2B"
NUM_SPECULATIVE_TOKENS = 2
MAX_GENERATE_LENGTH = 96
RESPONSE_TOLERANCE = 0.05
# Greedy decoding flips near-ties across a sleep/wake (2-3 of 8 on main as well);
# corrupted main-model weights diverge on every prompt.
MIN_MATCHING_RESPONSE_FRACTION = 0.5
MIN_ACCEPTANCE_AFTER_SYNC = 0.5
MAX_ACCEPTANCE_DROP = 0.15

PROMPTS = [
    "What is 12 times 13?",
    "Explain why the sky is blue.",
    "Write a Python function that reverses a string.",
    "What is the capital of France?",
    "Name three planets in the solar system.",
    "How many minutes are in two hours?",
    "Explain photosynthesis in one sentence.",
    "Continue the sequence: 2, 4, 8, 16.",
]

_SPEC_COUNTER_RE = re.compile(r"^vllm:spec_decode_num_(draft_tokens|accepted_tokens)(?:_total)?(?:\{[^}]*\})? (\S+)$")


async def _spec_decode_counters(client: RemoteInferenceClient) -> Tuple[float, float]:
    """(drafted tokens, accepted tokens) summed over every server's /metrics."""
    drafted = accepted = 0.0
    seen: List[str] = []
    async with httpx.AsyncClient(timeout=30.0) as http:
        for url in client.server_urls:
            response = await http.get(f"{url}/metrics")
            response.raise_for_status()
            for line in response.text.splitlines():
                if "spec_decode" in line and not line.startswith("#"):
                    seen.append(line)
                match = _SPEC_COUNTER_RE.match(line)
                if match is None:
                    continue
                if match.group(1) == "draft_tokens":
                    drafted += float(match.group(2))
                else:
                    accepted += float(match.group(2))
    assert drafted > 0 or not seen, f"spec-decode metric lines present but none parsed: {seen[:8]}"
    return drafted, accepted


def _make_cfg(weight_sync_backend: str, colocate_all: bool, inference_tp: int, megatron_tp: int, tmp_path):
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.trainer.placement.policy_num_gpus_per_node = megatron_tp
    cfg.trainer.placement.ref_num_gpus_per_node = megatron_tp
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = megatron_tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = NUM_SPECULATIVE_TOKENS

    ie_cfg = cfg.generator.inference_engine
    ie_cfg.backend = "vllm"
    ie_cfg.weight_sync_backend = weight_sync_backend
    ie_cfg.tensor_parallel_size = inference_tp
    ie_cfg.gpu_memory_utilization = 0.6
    ie_cfg.enforce_eager = True
    ie_cfg.max_num_seqs = 16
    ie_cfg.engine_init_kwargs = {"gdn_prefill_backend": "triton", "max_model_len": 1024}
    ie_cfg.enable_ray_prometheus_stats = False
    if weight_sync_backend == "delta":
        ie_cfg.delta_weight_sync = DeltaWeightSyncConfig(
            sync_dir=str(tmp_path / "sync"),
            local_checkpoint_dir=str(tmp_path / "receiver"),
            checkpoint_load_format="vllm_multi_thread_safetensors",
            multi_thread_safetensors_max_workers=2,
            publish_num_workers=2,
        )
    validate_cfg(cfg)
    assert ie_cfg.speculative_config == {"method": "mtp", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS}
    return cfg


@pytest.mark.parametrize(
    ("weight_sync_backend", "colocate_all", "inference_tp", "megatron_tp"),
    [
        pytest.param("nccl", True, 2, 2, id="cuda_ipc_colocated"),
        pytest.param("nccl", False, 1, 2, id="nccl_non_colocated"),
        pytest.param("delta", False, 1, 1, id="delta_non_colocated"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.megatron
async def test_megatron_mtp_weight_sync_roundtrip(
    ray_init_fixture, tmp_path, weight_sync_backend, colocate_all, inference_tp, megatron_tp
):
    cfg = _make_cfg(weight_sync_backend, colocate_all, inference_tp, megatron_tp, tmp_path)
    ie_cfg = cfg.generator.inference_engine

    tokenizer = get_tokenizer(MODEL_NAME)
    prompts = [[{"role": "user", "content": prompt}] for prompt in PROMPTS]
    sampling_params = get_sampling_params_for_backend(
        "vllm", SamplingParams(temperature=0.0, max_generate_length=MAX_GENERATE_LENGTH)
    )

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_NAME,
        use_local=True,
        backend="vllm",
        tp_size=inference_tp,
        colocate_all=colocate_all,
        num_inference_engines=1,
        sleep_level=2,
    ) as engines:
        client, pg = engines.client, engines.pg

        before = await run_inference(client, prompts, sampling_params, tokenizer=tokenizer)
        drafted_before, accepted_before = await _spec_decode_counters(client)
        assert drafted_before > 0, "speculative decoding is not active on the engine"
        acceptance_before = accepted_before / drafted_before
        print(f"[mtp weight sync] acceptance before sync: {acceptance_before:.3f} ({drafted_before:.0f} drafted)")

        # Level 2 discards every weight, the drafter's included: only the sync restores them.
        await client.sleep()
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=colocate_all,
            num_gpus_per_node=megatron_tp,
            cfg=cfg,
        )
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client, ie_cfg))
        await client.wake_up(tags=["weights"])
        with Timer("sync_weights"):
            ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client, ie_cfg))
        policy.offload_to_cpu()
        await client.wake_up(tags=["kv_cache"])

        after = await run_inference(client, prompts, sampling_params, tokenizer=tokenizer)
        drafted_after, accepted_after = await _spec_decode_counters(client)
        drafted = drafted_after - drafted_before
        accepted = accepted_after - accepted_before
        assert drafted > 0, "no drafts after the sync: speculative decoding stopped"
        acceptance_after = accepted / drafted
        print(f"[mtp weight sync] acceptance after sync: {acceptance_after:.3f} ({drafted:.0f} drafted)")

        assert all(before["responses"]) and all(after["responses"]), "empty generations: nothing to compare"
        diverged = [
            (i, resp_before, resp_after)
            for i, (resp_before, resp_after) in enumerate(zip(before["responses"], after["responses"]))
            if not are_responses_similar([resp_before], [resp_after], tolerance=RESPONSE_TOLERANCE)
        ]
        for i, resp_before, resp_after in diverged:
            print(f"[mtp weight sync] generation {i} diverged:\n  before: {resp_before!r}\n  after:  {resp_after!r}")
        matching = len(prompts) - len(diverged)
        assert matching >= MIN_MATCHING_RESPONSE_FRACTION * len(prompts), (
            f"only {matching}/{len(prompts)} generations survived the weight sync "
            f"(diverged: {[i for i, _, _ in diverged]})"
        )
        assert acceptance_after >= MIN_ACCEPTANCE_AFTER_SYNC, (
            f"draft acceptance collapsed to {acceptance_after:.3f} after the sync "
            f"(before: {acceptance_before:.3f}): the drafter was not reloaded"
        )
        assert (
            acceptance_after >= acceptance_before - MAX_ACCEPTANCE_DROP
        ), f"draft acceptance dropped from {acceptance_before:.3f} to {acceptance_after:.3f} across the sync"


@pytest.mark.megatron
def test_megatron_draft_source_selects_mtp_block_and_shared_weights():
    from types import SimpleNamespace

    import torch

    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronPolicyWorkerBase,
    )

    trunk = [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "decoder.final_layernorm.weight",
        "output_layer.weight",
    ]
    head = ["mtp.layers.0.enorm.weight", "mtp.layers.0.eh_proj.weight"]

    class _Bridge:
        def __init__(self, names):
            self.names = names

        def get_conversion_tasks(self, module):
            return [SimpleNamespace(global_param_name=name) for name in self.names]

        def export_hf_weights(self, module, show_progress, conversion_tasks):
            for task in conversion_tasks:
                yield task.global_param_name, torch.zeros(2)

    worker = SimpleNamespace(bridge=_Bridge(trunk + head), actor_module=[object()])
    source = MegatronPolicyWorkerBase._build_draft_weight_source(worker, torch.bfloat16)
    assert [name for name, _ in source] == ["embedding.word_embeddings.weight", "output_layer.weight", *head]

    worker = SimpleNamespace(bridge=_Bridge(trunk), actor_module=[object()])
    with pytest.raises(ValueError, match="no `mtp.\\*` parameters"):
        MegatronPolicyWorkerBase._build_draft_weight_source(worker, torch.bfloat16)
