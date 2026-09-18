"""Megatron MTP weight-sync round trip through vLLM, using CUDA IPC and NCCL."""

import json
import re
from dataclasses import dataclass, fields
from typing import Any, List, Optional, Set, Tuple

import httpx
import pytest
import ray
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.weight_sync import (
    WEIGHT_UPDATE_TARGET_DRAFT,
    WEIGHT_UPDATE_TARGET_MODEL,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import SamplingParams
from skyrl.train.utils.utils import validate_cfg
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
MIN_MATCHING_RESPONSE_FRACTION = 0.75
MIN_ACCEPTANCE_AFTER_SYNC = 0.5
MAX_ACCEPTANCE_DROP = 0.15

_SPEC_COUNTER_RE = re.compile(r"^vllm:spec_decode_num_(draft_tokens|accepted_tokens)(?:_total)?(?:\{[^}]*\})? (\S+)$")


@ray.remote
class _SyncRecorder:
    """Collects the control-plane calls the policy's rank 0 makes during a sync."""

    def __init__(self):
        self.events: List[Tuple[str, Any]] = []

    def record(self, event: Tuple[str, Any]) -> None:
        self.events.append(event)

    def get(self) -> List[Tuple[str, Any]]:
        return list(self.events)


@dataclass
class _RecordingClient(RemoteInferenceClient):
    """Record weight-update sessions across policy workers."""

    recorder: Any = None

    async def start_weight_update(self, is_checkpoint_format: bool = True, target: str = "model"):
        await self.recorder.record.remote(("start", target))
        return await super().start_weight_update(is_checkpoint_format=is_checkpoint_format, target=target)

    async def update_weights_ipc(self, update_info):
        await self.recorder.record.remote(("names", list(update_info["names"])))
        return await super().update_weights_ipc(update_info)

    async def update_weights_nccl(self, update_info):
        await self.recorder.record.remote(("names", list(update_info["names"])))
        return await super().update_weights_nccl(update_info)

    async def _call_all_servers(self, endpoint, payload=None, **kwargs):
        await self.recorder.record.remote(("endpoint", endpoint))
        return await super()._call_all_servers(endpoint, payload, **kwargs)

    async def finish_weight_update(self, target: str = "model"):
        await self.recorder.record.remote(("finish", target))
        return await super().finish_weight_update(target=target)


def _recording_client(client: RemoteInferenceClient, recorder) -> _RecordingClient:
    init_kwargs = {f.name: getattr(client, f.name) for f in fields(RemoteInferenceClient) if f.init}
    return _RecordingClient(**init_kwargs, recorder=recorder)


def _sessions(events: List[Tuple[str, Any]]) -> List[Tuple[str, List[str]]]:
    """Fold the recorded events into ``[(target, names sent in that session), ...]``."""
    sessions: List[Tuple[str, List[str]]] = []
    open_session: Optional[Tuple[str, List[str]]] = None
    for kind, payload in events:
        if kind == "start":
            assert open_session is None, "start_weight_update while a session is open"
            open_session = (payload, [])
        elif kind == "names":
            assert open_session is not None, "weights sent outside a session"
            open_session[1].extend(payload)
        elif kind == "finish":
            assert open_session is not None, "finish_weight_update without a session"
            sessions.append(open_session)
            open_session = None
    assert open_session is None, "a session was never finished"
    return sessions


def _hf_mtp_weight_names(model_name: str) -> Set[str]:
    """Read the checkpoint's MTP tensor names independently of the exporter."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model_name, "model.safetensors.index.json")) as f:
        keys = json.load(f)["weight_map"]
    names = {name for name in keys if name.startswith("mtp.") or ".mtp." in name}
    assert names, f"No MTP weights in {model_name}"
    return names


async def _spec_decode_counters(client: RemoteInferenceClient) -> Tuple[float, float]:
    """(drafted tokens, accepted tokens) summed over every backend server's /metrics."""
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


def _make_cfg(colocate_all: bool, inference_tp: int, megatron_tp: int) -> SkyRLTrainConfig:
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
    ie_cfg.weight_sync_backend = "nccl"
    ie_cfg.tensor_parallel_size = inference_tp
    ie_cfg.gpu_memory_utilization = 0.6
    ie_cfg.enforce_eager = True
    ie_cfg.max_num_seqs = 16
    ie_cfg.engine_init_kwargs = {"gdn_prefill_backend": "triton", "max_model_len": 1024}
    ie_cfg.enable_ray_prometheus_stats = False
    validate_cfg(cfg)
    assert ie_cfg.speculative_config == {"method": "mtp", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS}
    return cfg


@pytest.mark.parametrize(
    ("colocate_all", "inference_tp", "megatron_tp"),
    [
        pytest.param(True, 2, 2, id="cuda_ipc_colocated"),
        pytest.param(False, 1, 2, id="nccl_non_colocated"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.megatron
async def test_megatron_mtp_weight_sync_roundtrip(ray_init_fixture, colocate_all, inference_tp, megatron_tp):
    cfg = _make_cfg(colocate_all=colocate_all, inference_tp=inference_tp, megatron_tp=megatron_tp)
    ie_cfg = cfg.generator.inference_engine
    hf_mtp_names = _hf_mtp_weight_names(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompts = [
        [{"role": "user", "content": prompt}]
        for prompt in [
            "What is 12 times 13?",
            "Explain why the sky is blue.",
            "Write a Python function that reverses a string.",
            "What is the capital of France?",
            "Name three planets in the solar system.",
            "How many minutes are in two hours?",
            "Explain photosynthesis in one sentence.",
            "Continue the sequence: 2, 4, 8, 16.",
        ]
    ]
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
        sleep_level=2,  # the sync below re-syncs every weight, drafter included
    ) as engines:
        client, pg = engines.client, engines.pg

        before = await run_inference(client, prompts, sampling_params, tokenizer=tokenizer)
        drafted_before, accepted_before = await _spec_decode_counters(client)
        assert drafted_before > 0, "speculative decoding is not active on the engine"
        acceptance_before = accepted_before / drafted_before
        print(f"[mtp weight sync] acceptance before sync: {acceptance_before:.3f} ({drafted_before:.0f} drafted)")

        await client.sleep()
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=colocate_all,
            num_gpus_per_node=megatron_tp,
            cfg=cfg,
        )
        recorder = _SyncRecorder.remote()
        sync_client = _recording_client(client, recorder)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", sync_client, ie_cfg))
        await client.wake_up(tags=["weights"])
        with Timer("sync_weights"):
            ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", sync_client, ie_cfg))
        policy.offload_to_cpu()
        await sync_client.aclose()
        await client.wake_up(tags=["kv_cache"])

        events = ray.get(recorder.get.remote())
        endpoints = [payload for kind, payload in events if kind == "endpoint"]
        assert "/start_draft_weight_update" in endpoints
        assert "/finish_weight_update" in endpoints
        sessions = _sessions(events)
        assert [target for target, _ in sessions] == [
            WEIGHT_UPDATE_TARGET_MODEL,
            WEIGHT_UPDATE_TARGET_DRAFT,
        ], f"expected a main-model session followed by a draft session, got {[t for t, _ in sessions]}"
        draft_names = set(sessions[1][1])
        missing = sorted(hf_mtp_names - draft_names)
        assert not missing, f"MTP head tensors missing from the draft session: {missing}"
        print(
            f"[mtp weight sync] draft session carried {len(draft_names)} tensors "
            f"({len(hf_mtp_names)} mtp.* + embedding/lm_head): {sorted(draft_names)[:6]} ..."
        )

        after = await run_inference(client, prompts, sampling_params, tokenizer=tokenizer)
        drafted_after, accepted_after = await _spec_decode_counters(client)
        drafted = drafted_after - drafted_before
        accepted = accepted_after - accepted_before
        assert drafted > 0, "no drafts after the sync: speculative decoding stopped"
        acceptance_after = accepted / drafted
        print(f"[mtp weight sync] acceptance after sync: {acceptance_after:.3f} ({drafted:.0f} drafted)")

        assert len(before["responses"]) == len(after["responses"]) == len(prompts)
        assert all(before["responses"]) and all(after["responses"]), "empty generations: nothing to compare"
        diverged = [
            (i, resp_before, resp_after)
            for i, (resp_before, resp_after) in enumerate(zip(before["responses"], after["responses"]))
            if not are_responses_similar([resp_before], [resp_after], tolerance=RESPONSE_TOLERANCE)
        ]
        for i, resp_before, resp_after in diverged:
            print(f"[mtp weight sync] generation {i} diverged:\n  before: {resp_before!r}\n  after:  {resp_after!r}")
        matching = len(before["responses"]) - len(diverged)
        assert matching >= MIN_MATCHING_RESPONSE_FRACTION * len(before["responses"]), (
            f"only {matching}/{len(before['responses'])} generations survived the weight sync "
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
def test_megatron_draft_extractor_selects_mtp_block_and_embeddings(monkeypatch):
    from types import SimpleNamespace

    import torch

    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker as mw

    monkeypatch.setattr(mw, "broadcast_object_across_pp_ranks", lambda obj, allow_missing=False: obj)

    class _Mapping:
        tp_size = 1
        ep_size = 1
        is_expert = False
        is_grouped_export = False

    def _task(name):
        return SimpleNamespace(
            global_param_name=name,
            param_name=name,
            param_weight=torch.zeros(4, dtype=torch.bfloat16),
            mapping=_Mapping(),
        )

    trunk = [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "decoder.layers.0.mlp.linear_fc1.weight",
        "decoder.final_layernorm.weight",
        "output_layer.weight",
    ]
    head = [
        "mtp.layers.0.enorm.weight",
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.transformer_layer.mlp.linear_fc1.weight",
    ]

    class _Bridge:
        def __init__(self, names):
            self.tasks = [_task(n) for n in names]

        def get_conversion_tasks(self, module):
            return self.tasks

        def export_hf_weights(self, module, show_progress, conversion_tasks):
            for task in conversion_tasks:
                yield task.global_param_name, task.param_weight

    extractor = mw.MegatronWeightExtractor(bridge=_Bridge(trunk + head), actor_module=[object()], enable_bucketing=True)
    draft = extractor.draft_extractor()

    assert extractor.get_weight_metadata(torch.bfloat16)["names"] == trunk + head
    assert draft.get_weight_metadata(torch.bfloat16)["names"] == [
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        *head,
    ]

    with pytest.raises(ValueError, match="no MTP block"):
        mw.MegatronWeightExtractor(
            bridge=_Bridge(trunk), actor_module=[object()], enable_bucketing=True
        ).draft_extractor()
