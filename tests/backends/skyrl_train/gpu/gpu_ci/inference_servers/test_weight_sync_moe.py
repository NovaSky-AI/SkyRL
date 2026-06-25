"""
Weight sync tests for a small MoE model to trigger https://github.com/vllm-project/vllm/issues/42821


Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync_moe.py -v -s
"""

# Lazy (string) annotations: the Ray actors below are pickled by value, which serializes
# their methods' `__annotations__`. A real annotation object like `CudaIpcInitInfo` drags
# `torch.distributed` -> `torch.backends` (an unpicklable lazy `GenericModule`) into the
# pickle; keeping annotations as strings avoids serializing those class objects at all.
from __future__ import annotations

import glob
import pickle
from collections.abc import Iterator
from pathlib import Path

import pytest
import ray
import torch
from huggingface_hub import snapshot_download
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.inference_servers.common import (
    get_open_port,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import WorkerWrap
from skyrl.backends.skyrl_train.weight_sync import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    WeightChunk,
)
from skyrl.train.utils import get_ray_pg_ready_with_timeout

# NOTE: `vllm` and its submodules are imported lazily inside functions/methods, never at
# module top level. The Ray actors below (MoeEngineActor / MoeTrainer) are pickled by value
# by Ray, which serializes their methods' module globals; a top-level `import vllm` pollutes
# those globals with vLLM/transformers lazy `GenericModule` objects (e.g. `torch.backends`)
# that cloudpickle cannot serialize. Keeping vLLM imports local mirrors `test_weight_sync.py`.

MOE_MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MOE_PROMPT = "The capital of France is"
MOE_MAX_TOKENS = 64


class WorkerWrapRepro(WorkerWrap):
    """`WorkerWrap` that yields safetensors from disk as its own `_weight_receiver`.

    If we used a real NCCL/IPC receiver
    (e.g. `BroadcastWeightTransferReceiver` / `CudaIpcWeightTransferReceiver`),
    it would require a Ray-actor trainer holding the model in its own GPU memory,
    either on a second GPU (NCCL) or colocated (IPC); so this stub trades that
    for a safetensors read from disk, allowing a single GPU/no Ray-actor test.
    """

    def init_weight_update_communicator(self, init_info: bytes) -> None:
        self._snapshot_files = sorted(glob.glob(str(Path(init_info.decode("utf-8")) / "*.safetensors")))
        self._weight_receiver = self

    def receive_weights(self, request: object) -> Iterator[tuple[str, torch.Tensor]]:
        from vllm.model_executor.model_loader.weight_utils import (
            safetensors_weights_iterator,
        )

        yield from safetensors_weights_iterator(self._snapshot_files, use_tqdm_on_load=False)

    def teardown(self) -> None:
        pass


@pytest.mark.asyncio
async def test_worker_wrap_load_weights_preserves_moe_forward(ray_init_fixture) -> None:
    """Weight sync must not corrupt MoE forward output.

    Runs decode -> weight sync -> decode and asserts upon the outputs.
    Decodes are greedy for test determinism/repeatability.

    Regression test for https://github.com/NovaSky-AI/SkyRL/issues/1680.
    """
    import vllm
    from vllm.sampling_params import SamplingParams

    engine = vllm.AsyncLLMEngine.from_engine_args(
        vllm.AsyncEngineArgs(
            model=MOE_MODEL,
            # Cap the max seq length per request to fit the below 40% budget of GPU RAM
            max_model_len=4096,
            # `reload_weights` calls `initialize_layerwise_reload` to materialize fresh
            # per-layer GPU buffers; leave headroom for these buffers alongside the model
            gpu_memory_utilization=0.4,
            # Skip CUDAGraph capture as the test only needs two short prefills
            enforce_eager=True,
            worker_extension_cls=f"{WorkerWrapRepro.__module__}.{WorkerWrapRepro.__name__}",
            distributed_executor_backend="ray",
        )
    )

    sampling = SamplingParams(temperature=0, max_tokens=MOE_MAX_TOKENS)

    async for output in engine.generate(MOE_PROMPT, sampling, request_id="pre"):
        if output.finished:
            text_pre = output.outputs[0].text
            break

    # Temperature 0 decoding is deterministic given fixed model/dtype/tensor parallel
    assert (
        text_pre == "______.\nA. London\nB. Madrid\nC. Rome\nD. Paris\n答案:\nD"
    ), "Test requires a known output to confirm the second generation"

    snapshot = snapshot_download(repo_id=MOE_MODEL, allow_patterns=["*.safetensors", "*.json"])
    await engine.collective_rpc("init_weight_update_communicator", args=(snapshot.encode("utf-8"),))
    # Pass placeholder bytes args to satisfy `WorkerWrap.load_weights`
    await engine.collective_rpc("load_weights", args=(pickle.dumps(None),))

    async for output in engine.generate(MOE_PROMPT, sampling, request_id="post"):
        if output.finished:
            text_post = output.outputs[0].text
            break

    assert text_pre == text_post, (
        f"weight sync corrupted forward output\nPRE  ({len(text_pre)} chars): {text_pre!r}\n"
        f"POST ({len(text_post)} chars): {text_post!r}"
    )


# Carried in `WeightUpdateRequest.names` to capture the vLLM EngineCore
# subprocess's layerwise-reload warning for test assertions
_LAYERWISE_WARNING_NEEDLE = "Failed to load weights"

# Fractional GPU split for the colocated (engine + trainer) placement-group bundle.
# The single bundle owns 1.0 GPU; the engine actor, its vLLM TP worker, and the
# trainer actor each reserve a slice so they share one physical GPU (CUDA IPC
# requires sender and receiver on the same device). Sum stays < 1.0.
_ENGINE_ACTOR_GPUS = 0.1
_VLLM_WORKER_GPUS = 0.4
_TRAINER_GPUS = 0.4


@ray.remote
class MoeEngineActor:
    """Ray actor that owns the colocated vLLM `AsyncLLMEngine`.

    Created inside a placement-group bundle (with `placement_group_capture_child_tasks=True`)
    so vLLM's Ray executor lands its TP worker on that same bundle. `VLLM_RAY_*` env vars
    pin the worker to the bundle with a fractional GPU, leaving room for the trainer actor
    on the same physical GPU. Weight-sync RPCs are forwarded to the worker via
    `collective_rpc`, mirroring the real inference-server control plane.
    """

    def __init__(self, model: str, bundle_index: int) -> None:
        import os

        import vllm

        # Pin the vLLM Ray TP worker to this bundle with a fractional GPU reservation
        # (read by vLLM's RayDistributedExecutor at engine init). See
        # skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py:setup_envvars_for_vllm
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(_VLLM_WORKER_GPUS)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = str(bundle_index)
        # ray backend discovers GPUs via Ray scheduling, not CUDA_VISIBLE_DEVICES
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.engine = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=model,
                # See rationale for these hypers in
                # `test_worker_wrap_load_weights_preserves_moe_forward`
                max_model_len=4096,
                gpu_memory_utilization=0.4,
                enforce_eager=True,
                worker_extension_cls=f"{WorkerWrap.__module__}.{WorkerWrap.__name__}",
                distributed_executor_backend="ray",
            )
        )

    def ready(self) -> bool:
        return True

    async def generate(self, request_id: str) -> str:
        from vllm.sampling_params import SamplingParams

        sampling = SamplingParams(temperature=0, max_tokens=MOE_MAX_TOKENS)
        async for output in self.engine.generate(MOE_PROMPT, sampling, request_id=request_id):
            if output.finished:
                return output.outputs[0].text
        raise RuntimeError("engine.generate finished without producing output")

    async def init_weight_update_communicator(self, init_info: bytes) -> None:
        await self.engine.collective_rpc("init_weight_update_communicator", args=(init_info,))

    async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        # Call SkyRL's worker method, not vLLM's native Worker.start_weight_update
        # (added in vLLM 0.22.0+) which an unprefixed name now resolves to.
        await self.engine.collective_rpc("skyrl_start_weight_update", args=(is_checkpoint_format,))

    async def load_weights(self, request: bytes) -> None:
        await self.engine.collective_rpc("load_weights", args=(request,))

    async def finish_weight_update(self) -> None:
        await self.engine.collective_rpc("skyrl_finish_weight_update")


async def _run_legacy_ipc_send(engine_actor, snapshot_path: str, init_info) -> None:
    """Drive a legacy CUDA-IPC weight send from the trainer to the colocated engine.

    Kept as a module-level function (not a Ray-actor method) so cloudpickle serializes it
    by reference when shipping `MoeTrainer`, rather than walking its body — which references
    `torch` and vLLM lazy submodules that don't cloudpickle cleanly.
    """
    import torch

    import skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy as cuda_ipc_strategy

    # This runs in its own actor process, so force the legacy sender path here
    # (the test-process monkeypatch in the old in-process version wouldn't reach us).
    cuda_ipc_strategy._SKYRL_USE_NEW_INFERENCE = False

    # The legacy CUDA-IPC sender uses `torch.distributed` barriers / all_gather;
    # initialize a 1-rank group in this actor (sender side, rank 0).
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{get_open_port()}",
            world_size=1,
            rank=0,
        )

    class _EngineActorInferenceClient:
        """Routes the sender's control plane to the colocated engine actor."""

        async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
            await engine_actor.start_weight_update.remote(is_checkpoint_format)

        async def update_named_weights(self, request: object) -> None:
            await engine_actor.load_weights.remote(pickle.dumps(request))

        async def finish_weight_update(self) -> None:
            await engine_actor.finish_weight_update.remote()

    sender = CudaIpcTransferStrategy.create_sender(init_info, _EngineActorInferenceClient())

    snapshot_files = sorted(glob.glob(str(Path(snapshot_path) / "*.safetensors")))
    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator,
    )

    def one_chunk_per_param() -> Iterator[WeightChunk]:
        # group_by_module=False regime: one parameter per chunk (~hundreds of chunks),
        # streamed to GPU lazily so peak footprint stays at a single parameter
        for name, tensor in safetensors_weights_iterator(snapshot_files, use_tqdm_on_load=False):
            gpu_tensor = tensor.to("cuda", torch.bfloat16).contiguous()
            yield WeightChunk(
                names=[name],
                dtypes=["bfloat16"],
                shapes=[list(gpu_tensor.shape)],
                tensors=[gpu_tensor],
            )

    await sender.send_chunks(one_chunk_per_param())


@ray.remote
class MoeTrainer:
    """Ray actor that holds the (legacy CUDA-IPC) weight-send logic.

    Colocated on the same bundle/physical GPU as `MoeEngineActor`. Streams the
    on-disk checkpoint one parameter per chunk through the real
    `CudaIpcWeightTransferSender` (legacy path), forwarding the per-chunk control
    RPCs to the engine actor. This stands in for the training side of weight sync.
    """

    def __init__(self, engine_actor: "ray.actor.ActorHandle") -> None:
        self._engine = engine_actor

    def ready(self) -> bool:
        return True

    async def send_weights(self, snapshot_path: str, init_info: CudaIpcInitInfo) -> None:
        await _run_legacy_ipc_send(self._engine, snapshot_path, init_info)


@pytest.mark.asyncio
async def test_worker_wrap_multichunk_reload_preserves_moe_forward(
    capfd: pytest.CaptureFixture[str],
    ray_init_fixture,
) -> None:
    """Multi-chunk weight sync through the real CUDA-IPC legacy sender must not corrupt MoE.

    Same regression coverage as the in-process variant, but the send logic now lives
    in a `MoeTrainer` Ray actor colocated on the *same* physical GPU as the vLLM engine:
    both the engine (its TP worker) and the trainer take a fractional slice of a single
    1-GPU placement-group bundle, so CUDA IPC works between them.

    Regression test for https://github.com/NovaSky-AI/SkyRL/issues/1680.
    """
    # Single 1-GPU bundle shared by the engine worker and the trainer (colocated).
    pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=60)

    engine_actor = None
    trainer = None
    try:
        engine_actor = MoeEngineActor.options(
            num_gpus=_ENGINE_ACTOR_GPUS,
            num_cpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                # Capture vLLM's Ray TP worker (a child task) into this PG bundle
                placement_group_capture_child_tasks=True,
            ),
        ).remote(MOE_MODEL, bundle_index=0)
        ray.get(engine_actor.ready.remote())

        text_pre = await engine_actor.generate.remote("pre")

        snapshot = snapshot_download(repo_id=MOE_MODEL, allow_patterns=["*.safetensors", "*.json"])

        init_info = CudaIpcInitInfo(model_dtype_str="bfloat16", override_existing_receiver=True)
        # Real CUDA-IPC receiver on the worker
        await engine_actor.init_weight_update_communicator.remote(pickle.dumps(init_info))

        # Trainer colocated on the same bundle / physical GPU as the engine worker.
        trainer = MoeTrainer.options(
            num_gpus=_TRAINER_GPUS,
            num_cpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
            ),
        ).remote(engine_actor)
        ray.get(trainer.ready.remote())

        # Flush engine-init / first-generate output so the capture window holds only the sync.
        # The vLLM worker (a Ray actor) streams its layerwise-reload warnings to this
        # driver process's stdout/stderr, which capfd captures.
        capfd.readouterr()
        await trainer.send_weights.remote(snapshot, init_info)
        out_after, err_after = capfd.readouterr()

        text_post = await engine_actor.generate.remote("post")
    finally:
        if trainer is not None:
            ray.kill(trainer)
        if engine_actor is not None:
            ray.kill(engine_actor)
        ray.util.remove_placement_group(pg)

    # Filter out unrelated warnings
    warning_lines = [
        ln
        for ln in (out_after + err_after).splitlines()
        if _LAYERWISE_WARNING_NEEDLE in ln and "RotaryEmbedding" not in ln
    ]
    assert not warning_lines, (
        f"this identity sync emitted {len(warning_lines)} {_LAYERWISE_WARNING_NEEDLE!r} "
        "warnings: per-chunk reloads restored non-chunk layers; a correctly bracketed sync "
        "emits none. First 3:\n  " + "\n  ".join(warning_lines[:3])
    )
    assert text_post == text_pre, (
        "multi-chunk weight sync corrupted forward output\n"
        f"PRE  ({len(text_pre)} chars): {text_pre!r}\n"
        f"POST ({len(text_post)} chars): {text_post!r}"
    )
