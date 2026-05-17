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

from skyrl.backends.skyrl_train.inference_servers.vllm_worker import WorkerWrap

# Small unquantized MoE that triggers https://github.com/vllm-project/vllm/issues/42821
MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
PROMPT = "The capital of France is"
MAX_TOKENS = 64


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
async def test_worker_wrap_load_weights_preserves_moe_forward() -> None:
    """Weight sync must not corrupt MoE forward output.

    Runs decode -> weight sync -> decode and asserts upon the outputs.
    Decodes are greedy for test determinism/repeatability.

    Regression test for https://github.com/NovaSky-AI/SkyRL/issues/1680.
    """
    engine = vllm.AsyncLLMEngine.from_engine_args(
        vllm.AsyncEngineArgs(
            model=MODEL,
            # Cap the max seq length per request to fit the below 40% budget of GPU RAM
            max_model_len=4096,
            # `reload_weights` calls `initialize_layerwise_reload` to materialize fresh
            # per-layer GPU buffers; leave headroom for these buffers alongside the model
            gpu_memory_utilization=0.4,
            # Skip CUDAGraph capture as the test only needs two short prefills
            enforce_eager=True,
            worker_extension_cls=f"{WorkerWrapRepro.__module__}.{WorkerWrapRepro.__name__}",
        )
    )

    sampling = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    async for output in engine.generate(PROMPT, sampling, request_id="pre"):
        if output.finished:
            text_pre = output.outputs[0].text
            break

    # Temperature 0 decoding is deterministic given fixed model/dtype/tensor parallel
    assert (
        text_pre == "______.\nA. London\nB. Madrid\nC. Rome\nD. Paris\n答案:\nD"
    ), "Test requires a known output to confirm the second generation"

    snapshot = snapshot_download(repo_id=MODEL, allow_patterns=["*.safetensors", "*.json"])
    await engine.collective_rpc("init_weight_update_communicator", args=(snapshot.encode("utf-8"),))
    # Pass placeholder bytes args to satisfy `WorkerWrap.load_weights`
    await engine.collective_rpc("load_weights", args=(pickle.dumps(None),))

    async for output in engine.generate(PROMPT, sampling, request_id="post"):
        if output.finished:
            text_post = output.outputs[0].text
            break

    assert text_pre == text_post, (
        f"weight sync corrupted forward output\nPRE  ({len(text_pre)} chars): {text_pre!r}\n"
        f"POST ({len(text_post)} chars): {text_post!r}"
    )
