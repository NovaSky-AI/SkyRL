"""Colocated Megatron -> vLLM weight sync: byte parity + memory/time measurement.

The harness behind docs/design/colocated_weight_sync.md. It boots colocated vLLM
engines from the real checkpoint, records a checksum of every vLLM parameter and
buffer, sleeps them at level 2 (weights discarded), then runs
``WorkerDispatch.save_weights_for_sampler`` -- the production colocated path --
and asserts every vLLM tensor is **byte-identical** to the fresh load. A sync that
skips, truncates or misroutes any tensor fails here; a logprob-tolerance test
would not see it. It repeats the sync ``SKYRL_CWS_NUM_SYNCS`` times (each one is a
level-2 sleep + full reload) and dumps one JSON line per sync with the trainer's
and every vLLM worker's memory and timing.

A training step between syncs is checked separately
(``test_colocated_sync_after_train_step_matches_baseline``): the feature path
must produce the same vLLM bytes as today's path for the same trained weights.

Defaults run Qwen3-30B-A3B on one 8-GPU node, Megatron TP2/EP8, vLLM TP8.
Override with env vars (see ``_Case``). Run with:

SKYRL_CWS_MODEL=Qwen/Qwen3-30B-A3B uv run --isolated --extra dev --extra megatron -- \
    pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_colocated_weight_sync.py
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import pytest
import ray

from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    init_worker_with_type,
    make_dummy_training_batch,
    run_inference,
)

# Opt-in: boots a 30B MoE on 8 GPUs, so it never runs unless the model is named.
if not os.environ.get("SKYRL_CWS_MODEL"):
    pytest.skip("set SKYRL_CWS_MODEL to run the colocated weight-sync harness", allow_module_level=True)

_PROMPTS = [
    [{"role": "user", "content": q}]
    for q in (
        "What is 17 * 23? Show your steps.",
        "Write a haiku about GPUs.",
        "Name three prime numbers greater than 100.",
        "Explain expert parallelism in one paragraph.",
    )
]


async def _greedy(client) -> list:
    """Greedy completions (token ids): checks the synced weights *serve* (incl. CUDA graphs)."""
    from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
        get_sampling_params_for_backend,
    )
    from skyrl.train.config import SamplingParams

    params = get_sampling_params_for_backend("vllm", SamplingParams(temperature=0.0, max_generate_length=48))
    await client.reset_prefix_cache()
    out = await run_inference(client, _PROMPTS, params)
    return [list(r) for r in out["response_ids"]]


def _prefix_lens(a: list, b: list) -> list:
    out = []
    for x, y in zip(a, b):
        n = 0
        for u, v in zip(x, y):
            if u != v:
                break
            n += 1
        out.append(n)
    return out


def _env_int(name, default):
    return int(os.environ.get(name, default))


@dataclass
class _Case:
    model: str = field(default_factory=lambda: os.environ.get("SKYRL_CWS_MODEL", "Qwen/Qwen3-30B-A3B"))
    num_gpus: int = field(default_factory=lambda: _env_int("SKYRL_CWS_GPUS", 8))
    num_nodes: int = field(default_factory=lambda: _env_int("SKYRL_CWS_NODES", 1))
    tp: int = field(default_factory=lambda: _env_int("SKYRL_CWS_TP", 2))
    ep: int = field(default_factory=lambda: _env_int("SKYRL_CWS_EP", 8))
    pp: int = field(default_factory=lambda: _env_int("SKYRL_CWS_PP", 1))
    inference_tp: int = field(default_factory=lambda: _env_int("SKYRL_CWS_INFER_TP", 8))
    vllm_ep: bool = field(default_factory=lambda: os.environ.get("SKYRL_CWS_VLLM_EP", "0") == "1")
    buffer_gb: float = field(default_factory=lambda: float(os.environ.get("SKYRL_CWS_BUFFER_GB", "1.0")))
    num_syncs: int = field(default_factory=lambda: _env_int("SKYRL_CWS_NUM_SYNCS", 2))
    gpu_mem_util: float = field(default_factory=lambda: float(os.environ.get("SKYRL_CWS_GMU", "0.6")))
    # Eager by default: boots faster, and sidesteps an intermittent vLLM compile/autotune
    # hang seen on concurrent boots. Set 0 to cover CUDA graphs.
    enforce_eager: bool = field(default_factory=lambda: os.environ.get("SKYRL_CWS_EAGER", "1") == "1")
    # Extra dotted overrides, JSON: {"generator.inference_engine.weight_sync_num_ipc_buffers": 2, ...}
    overrides: dict = field(default_factory=lambda: json.loads(os.environ.get("SKYRL_CWS_OVERRIDES", "{}")))


def _set_dotted(cfg, key, value):
    obj = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def _make_cfg(case: _Case, overrides: dict | None = None) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = case.model
    cfg.trainer.logger = "console"
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = case.num_gpus // case.num_nodes
    cfg.trainer.placement.policy_num_nodes = case.num_nodes
    mc = cfg.trainer.policy.megatron_config
    mc.tensor_model_parallel_size = case.tp
    mc.expert_model_parallel_size = case.ep
    mc.pipeline_model_parallel_size = case.pp
    mc.expert_tensor_parallel_size = 1
    ie = cfg.generator.inference_engine
    ie.tensor_parallel_size = case.inference_tp
    ie.num_engines = case.num_gpus // case.inference_tp
    ie.run_engines_locally = True
    ie.distributed_executor_backend = "ray"
    ie.weight_transfer_threshold_cuda_ipc_GB = case.buffer_gb
    ie.gpu_memory_utilization = case.gpu_mem_util
    ie.enforce_eager = case.enforce_eager
    if case.vllm_ep:
        ie.engine_init_kwargs = dict(ie.engine_init_kwargs or {}, enable_expert_parallel=True)
    ie.engine_init_kwargs = dict(ie.engine_init_kwargs or {}, max_model_len=4096)
    for k, v in {**case.overrides, **(overrides or {})}.items():
        _set_dotted(cfg, k, v)
    validate_cfg(cfg)
    return cfg


async def _rpc_all(client, method: str, **kwargs) -> list:
    """``/collective_rpc`` on every server; the per-worker results, server-major."""
    out = await client._call_all_servers("/collective_rpc", {"method": method, "kwargs": kwargs})
    results = []
    for url in client.server_urls:
        results.extend(out[url]["body"]["results"])
    return results


def _diff_checksums(ref: list, got: list) -> list:
    bad = []
    for w, (r, g) in enumerate(zip(ref, got)):
        for name, val in r.items():
            if name not in g:
                bad.append(f"worker{w} {name}: missing")
            elif list(g[name]) != list(val):
                bad.append(f"worker{w} {name}: {val[2]} {val[3]} differs")
        for name in g.keys() - r.keys():
            bad.append(f"worker{w} {name}: unexpected")
    return bad


@dataclass
class _Harness:
    cfg: SkyRLTrainConfig
    client: object
    policy: object
    dispatch: WorkerDispatch

    async def sync_and_measure(self, label: str) -> dict:
        await _rpc_all(self.client, "skyrl_debug_memory_stats", reset_peak=True)
        t0 = time.perf_counter()
        await self.dispatch.save_weights_for_sampler()
        wall = time.perf_counter() - t0
        trainer = ray.get(self.policy.async_run_ray_method("pass_through", "get_last_weight_sync_stats"))
        vllm = await _rpc_all(self.client, "skyrl_debug_memory_stats")
        rec = {
            "label": label,
            "wall_seconds": wall,
            "transfer_seconds": self.dispatch.last_weight_sync_seconds,
            "trainer": trainer,
            "vllm": vllm,
        }
        summary = {
            "label": label,
            "wall_s": round(wall, 2),
            "transfer_s": round(self.dispatch.last_weight_sync_seconds or 0, 2),
            "trainer_max_alloc_GiB": max((p["max_allocated"] for t in trainer if t for p in t["phases"]), default=None),
            "trainer_device_peak_GiB": max(
                (p.get("device_peak_used", 0) for t in trainer if t for p in t["phases"]), default=None
            ),
            "vllm_max_alloc_GiB": max(v["device"]["max_allocated"] for v in vllm),
            "counters_rank0": trainer[0]["counters"] if trainer and trainer[0] else None,
        }
        summary["vllm_receive_w0"] = vllm[0].get("receive")
        summary["vllm_weights_awake_GiB_w0"] = (vllm[0].get("cumem") or {}).get("weights", {}).get("awake")
        print("CWS_SUMMARY " + json.dumps(summary, default=str), flush=True)
        print("CWS_RECORD " + json.dumps(rec, default=str), flush=True)
        return rec


async def _build(cfg: SkyRLTrainConfig, case: _Case, engines) -> _Harness:
    client, pg = engines.client, engines.pg
    await client.sleep()
    policy = init_worker_with_type(
        "policy",
        shared_pg=pg,
        colocate_all=True,
        num_gpus_per_node=case.num_gpus // case.num_nodes,
        num_nodes=case.num_nodes,
        cfg=cfg,
    )
    dispatch = WorkerDispatch(cfg=cfg, policy_actor_group=policy, inference_engine_client=client)
    # As the trainer does after building colocated models.
    policy.offload_to_cpu()
    dispatch.mark_all_offloaded()
    dispatch.init_weight_sync_state(client)
    return _Harness(cfg=cfg, client=client, policy=policy, dispatch=dispatch)


def _engine_cm(cfg: SkyRLTrainConfig, case: _Case):
    return InferenceEngineState.create(
        cfg=cfg,
        model=case.model,
        use_local=True,
        colocate_all=True,
        backend="vllm",
        sleep_level=2,
        gpu_memory_utilization=case.gpu_mem_util,
        engine_init_kwargs=cfg.generator.inference_engine.engine_init_kwargs,
    )


def _ray_env():
    env = {"SKYRL_COLOCATED_SYNC_DEBUG": "1"}
    # A fresh FlashInfer autotune cache per run. vLLM caches autotune results per rank;
    # a run killed mid-autotune leaves some ranks cached and others not, and the next
    # boot hangs in profile_run (cached ranks skip the synchronized autotune).
    env["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = os.environ.get(
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", f"/tmp/skyrl-cws-fi-autotune-{os.getpid()}-{int(time.time())}"
    )
    for k, v in os.environ.items():
        # The trainer cap, multi-node NCCL transport settings, and offline uv for worker envs.
        if k == "SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB" or k.startswith("NCCL_") or k == "UV_OFFLINE":
            env[k] = v
    return env


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_colocated_sync_byte_parity():
    """Every sync (level-2 sleep + reload) must reproduce the freshly loaded vLLM bytes."""
    case = _Case()
    with ray_init(extra_env_vars=_ray_env()):
        cfg = _make_cfg(case)
        async with _engine_cm(cfg, case) as engines:
            client = engines.client
            ref = await _rpc_all(client, "skyrl_debug_weight_checksums")
            print(f"CWS reference: {sum(len(r) for r in ref)} tensors over {len(ref)} workers", flush=True)
            ref_tokens = await _greedy(client)
            again = await _greedy(client)
            print(
                f"CWS fresh-engine greedy repeat: common prefix {_prefix_lens(ref_tokens, again)} of "
                f"{[len(t) for t in ref_tokens]} tokens",
                flush=True,
            )
            h = await _build(cfg, case, engines)
            for i in range(case.num_syncs):
                await h.sync_and_measure(f"sync{i}")
                got = await _rpc_all(client, "skyrl_debug_weight_checksums")
                bad = _diff_checksums(ref, got)
                assert not bad, f"sync{i}: {len(bad)} tensors differ from the fresh load, e.g. {bad[:10]}"
                tokens = await _greedy(client)
                prefix = _prefix_lens(ref_tokens, tokens)
                print(f"CWS sync{i}: byte parity OK; greedy common prefix vs fresh engine {prefix}", flush=True)
                # Exact equality is not a valid gate: with CUDA graphs even today's path drifts
                # at near-tie tokens with byte-identical weights. Broken weights diverge at once.
                assert min(prefix) >= 8, f"sync{i}: generations diverge early from the fresh engine: {prefix}"


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_colocated_sync_after_train_step_matches_baseline():
    """After a training step, the configured path must write the same vLLM bytes as today's path.

    Runs today's path (no overrides) and the case's overrides back to back on the same
    trained weights, by re-initializing the trainer engine between them.
    """
    case = _Case()
    if not case.overrides:
        pytest.skip("no feature overrides configured (SKYRL_CWS_OVERRIDES)")
    with ray_init(extra_env_vars=_ray_env()):
        base_cfg = _make_cfg(case, overrides={k: _default_of(k) for k in case.overrides})
        feat_cfg = _make_cfg(case)
        async with _engine_cm(feat_cfg, case) as engines:
            client = engines.client
            h = await _build(base_cfg, case, engines)
            batch = make_dummy_training_batch(batch_size=h.policy.actor_infos[0].rank.dp_size)
            h.dispatch.forward_backward("policy", batch)
            h.dispatch.optim_step("policy")
            await h.sync_and_measure("baseline_after_step")
            base = await _rpc_all(client, "skyrl_debug_weight_checksums")
            # Swap the trainer onto the feature config and rebuild its engine.
            h.dispatch.cfg = feat_cfg
            ray.get(h.policy.async_run_ray_method("pass_through", "reconfigure_for_test", feat_cfg))
            h.dispatch.init_weight_sync_state(client)
            await h.sync_and_measure("feature_after_step")
            got = await _rpc_all(client, "skyrl_debug_weight_checksums")
            bad = _diff_checksums(base, got)
            assert not bad, f"{len(bad)} tensors differ from today's path, e.g. {bad[:10]}"
            print("CWS after-step parity OK", flush=True)


def _default_of(key: str):
    obj = SkyRLTrainConfig()
    for p in key.split("."):
        obj = getattr(obj, p)
    return obj


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(test_colocated_sync_byte_parity())
