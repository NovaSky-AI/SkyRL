"""
Weight sync tests for a small MoE model to trigger https://github.com/vllm-project/vllm/issues/42821


Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync_moe.py -v -s
"""

import glob
import pickle
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import vllm
from huggingface_hub import snapshot_download
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.sampling_params import SamplingParams

from skyrl.backends.skyrl_train.inference_servers.common import (
    get_open_port,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import WorkerWrap
from skyrl.backends.skyrl_train.weight_sync import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    WeightChunk,
)

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


@pytest.mark.asyncio
async def test_worker_wrap_multichunk_reload_preserves_moe_forward(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    ray_init_fixture,
) -> None:
    """Multi-chunk weight sync through the real CUDA-IPC legacy sender must not corrupt MoE.

    Regression test for https://github.com/NovaSky-AI/SkyRL/issues/1680.
    """
    # Test is covering the legacy sender path, so force it here
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy._SKYRL_USE_NEW_INFERENCE",
        False,
    )

    engine = vllm.AsyncLLMEngine.from_engine_args(
        vllm.AsyncEngineArgs(
            model=MOE_MODEL,
            # See rationale for these hypers in the above
            # `test_worker_wrap_load_weights_preserves_moe_forward`
            max_model_len=4096,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            worker_extension_cls=f"{WorkerWrap.__module__}.{WorkerWrap.__name__}",
            distributed_executor_backend="ray",
        )
    )

    sampling = SamplingParams(temperature=0, max_tokens=MOE_MAX_TOKENS)

    async for output in engine.generate(MOE_PROMPT, sampling, request_id="pre"):
        if output.finished:
            text_pre = output.outputs[0].text
            break

    snapshot = snapshot_download(repo_id=MOE_MODEL, allow_patterns=["*.safetensors", "*.json"])
    snapshot_files = sorted(glob.glob(str(Path(snapshot) / "*.safetensors")))

    # Real CUDA-IPC receiver on the worker
    await engine.collective_rpc(
        "init_weight_update_communicator",
        args=(pickle.dumps(CudaIpcInitInfo(model_dtype_str="bfloat16", override_existing_receiver=True)),),
    )

    # Since this test covers code with `torch.distributed` barriers, initialize a 1-rank group
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{get_open_port()}",
            world_size=1,
            rank=0,
        )

    class _InProcessInferenceEngineClient:
        async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
            await engine.collective_rpc("start_weight_update", args=(is_checkpoint_format,))

        async def update_named_weights(self, request: object) -> None:
            await engine.collective_rpc("load_weights", args=(pickle.dumps(request),))

        async def finish_weight_update(self) -> None:
            await engine.collective_rpc("finish_weight_update")

    sender = CudaIpcTransferStrategy.create_sender(
        CudaIpcInitInfo(model_dtype_str="bfloat16", override_existing_receiver=True),
        _InProcessInferenceEngineClient(),
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

    # Flush engine-init / first-generate output so the capture window holds only the sync
    capfd.readouterr()
    await sender.send_chunks(one_chunk_per_param())
    out_after, err_after = capfd.readouterr()

    async for output in engine.generate(MOE_PROMPT, sampling, request_id="post"):
        if output.finished:
            text_post = output.outputs[0].text
            break

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
