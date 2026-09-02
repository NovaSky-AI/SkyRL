"""
GPU CI tests for weight synchronization from trainer to inference server.

Each case drives the production trainer-side path: ``build_trainer_engine`` picks
the init info and builds the client, then ``engine.send_weights()`` owns the round
trip over vLLM's native RLHF routes. The trainer actor stands in for
FSDP/Megatron only -- it holds a plain HF model on one GPU, which
``FsdpWeightSource`` handles unchanged.

1. Non-colocated NCCL broadcast, TP=2, plus a 1P1D PD variant. Covers
   ``NCCLTrainerWeightTransferEngine`` against ``skyrl_nccl``, and the per-server
   ``rank_offset`` rewrite in ``nccl_init_payloads``.
2. Colocated CUDA IPC, TP=1. Covers ``IPCTrainerWeightTransferEngine`` (packed)
   against ``skyrl_ipc``, and the ``nccl`` + ``colocate_all`` -> ``ipc``
   resolution.
3. Non-colocated sharded RDT (NIXL pull), TP=1. Covers
   ``ShardedRDTTrainerWeightTransferEngine``, the ownership-aware source, and the
   ``replica_rank`` rewrite in ``rdt_init_payloads``.

Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v -s
"""

import asyncio
import os
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = os.environ.get("SKYRL_RDT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PROMPT = {
    "model": MODEL,
    "prompt": "What is the capital of France?",
    "max_tokens": 32,
    "temperature": 0.0,
}


class WeightSyncTrainerBase:
    """Single-GPU stand-in for a training worker, driving the real engine path.

    No ``torch.distributed`` group, which is the rank-0-only case: the engines
    resolve ``is_sender`` from ``init_info.rank``, and the IPC handle all-gather
    and delta result gather both no-op without a group.

    Not ``@ray.remote`` itself -- each backend needs different Ray resources, so
    the decorators are applied per backend below.
    """

    def __init__(
        self, model_name, weight_sync_backend, colocate_all, server_urls, data_parallel_size, inference_world_size
    ):
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        # The two config values the backend is resolved from, so this exercises
        # the same resolution the driver uses to configure the servers.
        self._ie_cfg = SimpleNamespace(weight_sync_backend=weight_sync_backend, model_dtype="bfloat16")
        self._colocate_all = colocate_all
        self._server_urls = list(server_urls)
        self._data_parallel_size = int(data_parallel_size)
        self._inference_world_size = int(inference_world_size)
        self._model_name = model_name
        self._engine = None

    def ready(self):
        return True

    def _build_source(self, dtype, backend):
        """The ``source_factory`` build_trainer_engine calls once the backend is known."""
        if backend == "sharded_rdt":
            from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
                make_fsdp_weight_source,
            )

            return make_fsdp_weight_source(self._model, dtype)

        from skyrl.backends.skyrl_train.weight_sync.sources import FsdpWeightSource

        return FsdpWeightSource(self._model, dtype)

    def sync_once(self):
        """Rendezvous on the first call, then run one full weight sync."""
        from skyrl.backends.skyrl_train.weight_sync.trainer_engines import (
            build_trainer_engine,
        )

        if self._engine is None:
            self._engine = build_trainer_engine(
                ie_cfg=self._ie_cfg,
                colocate_all=self._colocate_all,
                rank=0,
                inference_world_size=self._inference_world_size,
                source_factory=self._build_source,
                server_urls=self._server_urls,
                data_parallel_size=self._data_parallel_size,
                base_model_path=self._model_name,
            )
        self._engine.send_weights()

    def shutdown(self):
        if self._engine is not None:
            from skyrl.backends.skyrl_train.weight_sync.trainer_engines import (
                teardown_engine,
            )

            teardown_engine(self._engine)
            self._engine = None


# A whole GPU for NCCL and RDT; RDT additionally needs it so its engine can pin
# the producer sidecar it spawns to this GPU for CUDA IPC.
NcclTrainer = ray.remote(num_gpus=1)(WeightSyncTrainerBase)
RdtTrainer = ray.remote(num_gpus=1, max_concurrency=4)(WeightSyncTrainerBase)
# IPC shares a GPU with the colocated server, so a fraction.
IpcTrainer = ray.remote(WeightSyncTrainerBase)


async def _completion(http_client, router_url):
    resp = await http_client.post(f"{router_url}/v1/completions", json=PROMPT)
    assert resp.status_code == 200
    return resp.json()["choices"][0]["text"]


async def _assert_sync_replaces_dummy_weights(env, timeout_s: float = 120.0):
    """Dummy weights -> sync -> real weights.

    ``load_format="dummy"`` starts the server with garbage, so "Paris" appearing
    proves the transfer landed *and* that ``process_weights_after_loading`` ran
    (an unfinalized layerwise reload produces garbage too).
    """
    router_url = env["router_url"]
    trainer = env["trainer"]

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as http_client:
        text_before = await _completion(http_client, router_url)
        print(f"[step 1] dummy weights output: {text_before!r}")
        assert "Paris" not in text_before, "Dummy weights unexpectedly produced the correct answer"

        print("[step 2] trainer.sync_once() -- rendezvous + one full send_weights()")
        await asyncio.to_thread(lambda: ray.get(trainer.sync_once.remote()))

        text_after = await _completion(http_client, router_url)
        print(f"[step 3] synced weights output: {text_after!r}")
        assert "Paris" in text_after, f"Weight sync failed - expected 'Paris' but got: {text_after!r}"


async def _make_env(cfg, create_kwargs, trainer_cls, weight_sync_backend, *, colocate_with_engine=False):
    """Bring up the servers plus a trainer actor, and yield the pair."""
    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        client = engines.client
        inference_world_size, _ = await client.get_world_size()
        options = {}
        if colocate_with_engine:
            options = dict(
                num_gpus=0.2,
                num_cpus=0.2,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=engines.pg,
                    placement_group_bundle_index=0,
                ),
            )
        trainer = trainer_cls.options(**options).remote(
            MODEL,
            weight_sync_backend,
            # From create_kwargs, not cfg: InferenceEngineState.create deep-copies
            # cfg before applying the override, so the outer cfg still has the
            # default. This must be the value the servers were built with.
            create_kwargs["colocate_all"],
            client.server_urls,
            client.data_parallel_size,
            inference_world_size,
        )
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": client,
            "router_url": client.proxy_url,
        }

        ray.get(trainer.shutdown.remote())
        await client.teardown()
        ray.kill(trainer)
    # cleanup manually in colocated case
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


# -----------------------------------------------------------------
# Non-colocated NCCL broadcast
# -----------------------------------------------------------------


@pytest_asyncio.fixture(
    scope="class",
    params=[
        pytest.param({"enable_pd": False}, id="no_pd"),
        pytest.param(
            {"enable_pd": True, "num_prefill": 1, "num_decode": 1},
            id="pd_1P1D_non_colocated",
        ),
    ],
)
async def weight_update_env(class_scoped_ray_init_fixture, request):
    """
    Create environment for weight update testing (non-colocated, NCCL broadcast).

    - no_pd: TP=2 server on its own GPUs, trainer on separate GPU(s) (4 GPUs).
    - pd_1P1D_non_colocated: 1P1D (2 engines, TP=1), trainer on separate GPU (3 GPUs).
      Exercises non-colocated PD path in create_inference_servers with separate
      prefill/decode placement groups, and -- being two deployments -- the
      per-deployment ``rank_offset`` advance, which a single deployment cannot
      distinguish from a constant.
    """
    pd_cfg = request.param
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL

    if pd_cfg["enable_pd"]:
        num_prefill = pd_cfg["num_prefill"]
        num_decode = pd_cfg["num_decode"]
        create_kwargs = dict(
            model=MODEL,
            tp_size=1,
            num_inference_engines=num_prefill + num_decode,
            colocate_all=False,
            gpu_memory_utilization=0.5,
            engine_init_kwargs={
                "load_format": "dummy",
                "kv_transfer_config": {
                    "kv_connector": "NixlConnector",
                },
            },
            enable_pd=True,
            num_prefill=num_prefill,
        )
    else:
        create_kwargs = dict(
            model=MODEL,
            tp_size=2,
            colocate_all=False,
            gpu_memory_utilization=0.5,
            engine_init_kwargs={"load_format": "dummy"},
        )

    async for env in _make_env(cfg, create_kwargs, NcclTrainer, "nccl"):
        yield env


@pytest.mark.asyncio(loop_scope="class")
class TestWeightUpdateFlow:
    """Weight sync via NCCL broadcast (non-colocated)."""

    async def test_update_weights_flow(self, weight_update_env):
        """``send_weights()`` runs the inference-side ``update_weights``
        concurrently with the trainer-side broadcast, so a mis-sized receive
        buffer or a mismatched ``packed`` hangs rather than failing. Treat a
        timeout here as that bug, not as flake."""
        await _assert_sync_replaces_dummy_weights(weight_update_env)


# -----------------------------------------------------------------
# Colocated CUDA IPC
# -----------------------------------------------------------------


@pytest_asyncio.fixture(scope="class")
async def ipc_weight_update_env(class_scoped_ray_init_fixture):
    """Create environment for colocated IPC weight update testing."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    create_kwargs = dict(
        model=MODEL,
        tp_size=1,
        colocate_all=True,
        gpu_memory_utilization=0.5,
        engine_init_kwargs={"load_format": "dummy"},
    )

    # weight_sync_backend="nccl" + colocate_all resolves to ipc, exactly as the
    # driver resolves it for the servers.
    async for env in _make_env(cfg, create_kwargs, IpcTrainer, "nccl", colocate_with_engine=True):
        yield env


@pytest.mark.asyncio(loop_scope="class")
class TestColocatedIpcWeightUpdateFlow:
    """Weight sync via CUDA IPC (colocated, TP=1)."""

    async def test_update_weights_ipc(self, ipc_weight_update_env):
        """Packed IPC: the producer streams through one reusable buffer and the
        consumer clones out of it. A wrong refcount contract surfaces as garbage
        weights -- the buffer reused under a reader -- which "Paris" catches."""
        await _assert_sync_replaces_dummy_weights(ipc_weight_update_env)


# -----------------------------------------------------------------
# Sharded RDT (NIXL pull)
# -----------------------------------------------------------------


@pytest_asyncio.fixture(scope="class")
async def rdt_weight_update_env(class_scoped_ray_init_fixture):
    """Non-colocated sharded_rdt (NIXL pull) environment, TP=1.

    The trainer actor (1 GPU) drives the engine, which spawns its own producer
    sidecar on that GPU; the vLLM server (TP=1,
    distributed_executor_backend=ray) runs on another GPU. 2 GPUs + the sidecar.
    """
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    # Selects the sharded_rdt backend: build_vllm_cli_args reads this and sets
    # WeightTransferConfig(backend="sharded_rdt") + executor=ray.
    cfg.generator.inference_engine.weight_sync_backend = "sharded_rdt"

    create_kwargs = dict(
        model=MODEL,
        tp_size=1,
        colocate_all=False,
        gpu_memory_utilization=0.5,
        engine_init_kwargs={"load_format": "dummy"},
    )

    async for env in _make_env(cfg, create_kwargs, RdtTrainer, "sharded_rdt"):
        yield env


@pytest.mark.asyncio(loop_scope="class")
class TestShardedRdtWeightUpdateFlow:
    """Weight sync via the sharded_rdt (NIXL pull) backend (non-colocated, TP=1)."""

    async def test_update_weights_rdt(self, rdt_weight_update_env):
        """The first ``sync_once`` spawns the producer sidecar and bakes the
        replay plan on the inference side, inside the engine's
        ``init_transfer_engine``. Then the workers pull their slices over NIXL."""
        await _assert_sync_replaces_dummy_weights(rdt_weight_update_env, timeout_s=180.0)
