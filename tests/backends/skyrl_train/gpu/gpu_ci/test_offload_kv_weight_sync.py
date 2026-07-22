"""GPU test for the ``offload_kv_for_weight_sync`` weight-sync path.

With ``generator.inference_engine.offload_kv_for_weight_sync=True`` (fully-async),
``WorkerDispatch.save_weights_for_sampler`` freezes in-flight requests (KEEP pause),
offloads the KV cache to CPU, re-syncs weights into the freed space, then restores
the KV cache. This asserts an in-flight request survives that offload/restore and
finishes cleanly.

GPU Requirements: 2 GPUs (1 inference + 1 policy).

Run with:
uv run --isolated --extra dev --extra fsdp \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/test_offload_kv_weight_sync.py -v -s
"""

import asyncio
from typing import List

import pytest
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers import remote_inference_client as _ric
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_inference_engine_cfg
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    init_worker_with_type,
    make_dummy_training_batch,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Long prompt + ignore_eos + large max_tokens keeps a request mid-generation while
# the training step and weight sync land underneath it.
LONG_PROMPT = "Tell me a very long, detailed story about a dragon who learns to code."


def _offload_kv_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.logger = "console"
    # Required by offload_kv_for_weight_sync so the broadcast doesn't reset the
    # prefix cache and preempt the request we are preserving.
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.clear_kv_cache_on_weight_sync = False
    ie = cfg.generator.inference_engine
    ie.num_engines = 1
    ie.tensor_parallel_size = 1
    ie.run_engines_locally = True
    ie.max_num_seqs = 16
    # Modest KV pool so the CPU offload/restore is fast in CI.
    ie.gpu_memory_utilization = 0.6
    ie.offload_kv_for_weight_sync = True
    validate_inference_engine_cfg(cfg)
    return cfg


def _prompt_token_ids(tokenizer) -> List[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": LONG_PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


def _sample_payload(prompt_token_ids: List[int], model: str, max_tokens: int) -> dict:
    return {
        "json": {
            "model": model,
            "prompt": {"chunks": [{"tokens": prompt_token_ids}]},
            "num_samples": 1,
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "seed": 1234,
                "ignore_eos": True,  # keep emitting past EOS so the request stays in-flight
            },
        }
    }


@pytest.mark.asyncio
async def test_offload_kv_weight_sync_preserves_inflight_request(ray_init_fixture, monkeypatch):
    """An in-flight request must survive the KV offload/restore weight sync.

    While a long sample is mid-generation we run a real training step and
    ``save_weights_for_sampler()`` (which takes the ``offload_kv_for_weight_sync``
    branch). The in-flight request must finish with a non-abort stop reason and a
    non-empty token stream, and the offload path (``sleep_preserving_inflight`` +
    ``wake_up_preserved(["kv_cache"])``) must actually have been exercised.
    """
    # ignore_eos isn't in the default Tinker->vLLM forwarding map; widen it.
    monkeypatch.setitem(_ric._TINKER_SAMPLE_TO_VLLM_PARAM_MAP, "ignore_eos", "ignore_eos")

    cfg = _offload_kv_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_token_ids = _prompt_token_ids(tokenizer)

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL,
        use_local=True,
        tp_size=1,
        colocate_all=False,
        # enable_sleep_mode is turned on by offload_kv_for_weight_sync; sleep_level just needs >=1.
        sleep_level=1,
    ) as engines:
        client, pg = engines.client, engines.pg
        assert isinstance(
            client, RemoteInferenceClient
        ), f"This test targets the HTTP path (RemoteInferenceClient), got {type(client).__name__}"

        # Policy worker + dispatch for a real NCCL-broadcast weight sync.
        policy_group = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        dispatch = WorkerDispatch(cfg=cfg, policy_actor_group=policy_group, inference_engine_client=client)
        dispatch.init_weight_sync_state(client)

        # Spy on the offload primitives to prove the path actually ran.
        sleep_calls: List[tuple] = []
        wake_tags: List[list] = []
        orig_sleep = client.sleep_preserving_inflight
        orig_wake = client.wake_up_preserved

        async def spy_sleep(*args, **kwargs):
            sleep_calls.append((args, kwargs))
            return await orig_sleep(*args, **kwargs)

        async def spy_wake(*args, **kwargs):
            wake_tags.append(kwargs.get("tags", args[0] if args else None))
            return await orig_wake(*args, **kwargs)

        monkeypatch.setattr(client, "sleep_preserving_inflight", spy_sleep)
        monkeypatch.setattr(client, "wake_up_preserved", spy_wake)

        # Launch a long in-flight sample and let it start emitting tokens.
        task = asyncio.create_task(_sample_via(client, _sample_payload(prompt_token_ids, MODEL, max_tokens=1024)))
        await asyncio.sleep(1.5)
        assert not task.done(), (
            "sample completed before the weight sync landed — bump max_tokens or "
            "shorten the pre-sync sleep so the sync truly lands mid-flight"
        )

        # Real training step, then weight sync (takes the offload_kv path).
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)
        dispatch.forward_backward("policy", dummy_batch)
        dispatch.optim_step("policy")
        await dispatch.save_weights_for_sampler()

        # The frozen in-flight request must resume and finish cleanly.
        result = await asyncio.wait_for(task, timeout=180.0)
        seq = result["sequences"][0]
        assert seq["stop_reason"] in ("stop", "length"), (
            f"in-flight sample was aborted across the offload_kv weight sync — "
            f"stop_reason={seq['stop_reason']!r}, expected 'stop' or 'length'"
        )
        assert len(seq["tokens"]) > 0, "in-flight sample produced no tokens after the sync"

        # Confirm the KV-offloading path was actually exercised.
        assert sleep_calls, "sleep_preserving_inflight was never called — offload_kv path not taken"
        assert any(tags and "weights" in tags for tags in wake_tags), "weights were never woken for the broadcast"
        assert any(tags and "kv_cache" in tags for tags in wake_tags), "KV cache was never restored"

        # Engine still healthy after restore: a fresh sample works.
        await client.reset_prefix_cache()
        fresh = await _sample_via(client, _sample_payload(prompt_token_ids, MODEL, max_tokens=16))
        assert len(fresh["sequences"][0]["tokens"]) > 0, "engine produced no tokens after KV restore"


async def _sample_via(client: RemoteInferenceClient, payload: dict):
    return await client.sample(payload)
