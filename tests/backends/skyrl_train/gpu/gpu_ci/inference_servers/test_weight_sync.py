"""
GPU CI tests for weight synchronization from trainer to inference server.

1. Non-colocated (NCCL broadcast), TP=2:
    - Trainer on GPUs 0-1, server (TP=2) on GPUs 2-3 (4 GPUs total)
    - Uses NCCL broadcast for weight sync via HTTP router

2. Colocated (CUDA IPC), TP=1:
    - Trainer and server share GPU 0 (2 GPUs total, 1 shared)
    - Uses CUDA IPC handles for zero-copy weight transfer

3. Legacy `WorkerWrap.load_weights` MoE reload, TP=1:
    - Server on GPU 0 (1 GPU total, no separate trainer process)
    - The NCCL or CUDA-IPC receiver is stubbed with safetensors-from-disk
      to skip trainer-side sender setup

Run:
    uv run pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v -s
"""

import base64
import glob
import pickle
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import ray
import torch
import vllm
from huggingface_hub import snapshot_download
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.sampling_params import SamplingParams

from skyrl.backends.skyrl_train.inference_servers.common import (
    get_node_ip,
    get_open_port,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import WorkerWrap
from skyrl.backends.skyrl_train.weight_sync import (
    BroadcastInitInfo,
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    WeightChunk,
)
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@ray.remote
class Trainer:
    """
    Simple trainer emulator that holds the real model weights.

    This is a simplified version of the trainer side for testing weight sync
    via NCCL broadcast in non-colocated scenarios.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pg = None
        self.model_name = model_name

    def ready(self):
        """Check if the trainer is ready."""
        return True

    def init_weight_sync(self, master_address: str, master_port: int, world_size: int, group_name: str):
        """Initialize the weight sync process group as rank 0 (trainer)."""
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        self.pg = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            )
        )
        return True

    def get_weight_info(self) -> dict:
        """
        Get weight metadata (names, dtypes, shapes) without doing NCCL.

        Returns:
            dict with names, dtypes, shapes for the weight update request.
        """
        names = []
        dtypes = []
        shapes = []

        for name, param in self.model.named_parameters():
            names.append(name)
            dtypes.append(str(param.dtype).split(".")[-1])  # e.g. "bfloat16"
            shapes.append(list(param.shape))

        return {"names": names, "dtypes": dtypes, "shapes": shapes}

    def broadcast_weights(self):
        """
        Broadcast all model weights to inference workers via NCCL.

        This is a blocking operation - server must call receive concurrently.
        """
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        params = list(self.model.named_parameters())
        print(
            f"[Trainer.broadcast_weights] Starting send of {len(params)} params, pg={self.pg}, pg.rank={self.pg.rank}, pg.world_size={self.pg.world_size}"
        )
        try:
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(params),
                trainer_args={"group": self.pg, "packed": True},
            )
            torch.cuda.synchronize()
            print("[Trainer.broadcast_weights] Send complete")
        except Exception as e:
            print(f"[Trainer.broadcast_weights] ERROR: {e}")
            raise


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
      prefill/decode placement groups.
    """
    pd_cfg = request.param
    enable_pd = pd_cfg["enable_pd"]
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL

    if enable_pd:
        num_prefill = pd_cfg["num_prefill"]
        num_decode = pd_cfg["num_decode"]
        create_kwargs = dict(
            model=MODEL,
            tp_size=1,
            num_inference_engines=num_prefill + num_decode,
            colocate_all=False,
            gpu_memory_utilization=0.5,
            use_new_inference_servers=True,
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
            use_new_inference_servers=True,
            engine_init_kwargs={"load_format": "dummy"},
        )

    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        trainer = Trainer.options(num_gpus=1.0).remote(MODEL)
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": engines.client,
            "router_url": engines.client.proxy_url,
        }

        await engines.client.teardown()
        ray.kill(trainer)
    # cleanup manually in colocated case
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


@pytest.mark.asyncio(loop_scope="class")
class TestWeightUpdateFlow:
    """Tests for weight synchronization from trainer to inference server (non-colocated)."""

    async def test_update_weights_flow(self, weight_update_env):
        """
        Full E2E weight sync test (non-colocated, NCCL broadcast):
        1. Query with dummy weights → gibberish
        2. Init weight transfer (both sides concurrently via client)
        3. Broadcast weights from trainer (concurrent with server receive)
        4. Finalize weight update
        5. Query again → correct output
        """
        router_url = weight_update_env["router_url"]
        trainer = weight_update_env["trainer"]
        client = weight_update_env["client"]

        print("\n[TEST] Running non-colocated weight sync test")

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
            # ===== Step 1: Verify dummy weights produce gibberish =====
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_before = resp.json()["choices"][0]["text"]
            print(f"[Step 1] Dummy weights output: {text_before!r}")

            # Dummy weights should NOT produce coherent output about Paris
            assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 2: Init weight transfer (both sides concurrently) =====
            master_address = get_node_ip()
            master_port = get_open_port()

            # Query all servers for world_size via client (fans out to all backends)
            inference_world_size, _ = await client.get_world_size()
            world_size = 1 + inference_world_size  # 1 trainer + all inference workers
            group_name = f"weight_sync_test_{master_port}"

            print(f"[Step 2] Init weight transfer: master={master_address}:{master_port}, world_size={world_size}")

            init_info = BroadcastInitInfo(
                master_addr=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
                group_name=group_name,
                backend="nccl",
                model_dtype_str="bfloat16",
                override_existing_receiver=True,
            )

            # Both sides must init concurrently (NCCL blocks until all ranks join)
            # Start trainer init (returns immediately, runs in Ray actor)
            trainer_init_ref = trainer.init_weight_sync.remote(master_address, master_port, world_size, group_name)

            # Await server init via client (fans out to all backends)
            result = await client.init_weight_update_communicator(init_info)
            for server_url, resp in result.items():
                assert resp["status"] == 200, f"Server {server_url} init failed: {resp}"

            # Trainer should be done now (NCCL group formed)
            ray.get(trainer_init_ref)
            print("[Step 2] Both sides init complete")

            # ===== Step 3: Broadcast weights (concurrent send/receive) =====
            print("[Step 3] Broadcasting weights from trainer to server...")

            # Get weight metadata first (no NCCL yet)
            weight_info = ray.get(trainer.get_weight_info.remote())
            print(f"[Step 3] Weight info: {len(weight_info['names'])} parameters")

            # Start trainer broadcast (returns immediately, runs in Ray actor)
            print("[Step 3] Launching trainer broadcast_weights.remote()...")
            trainer_broadcast_ref = trainer.broadcast_weights.remote()

            # Await server receive via client (fans out to all backends)
            dtype_names = [(d.split(".")[-1] if "." in d else d) for d in weight_info["dtypes"]]
            update_info = {
                "names": weight_info["names"],
                "dtype_names": dtype_names,
                "shapes": weight_info["shapes"],
                "packed": True,
            }
            print(
                f"[Step 3] Calling update_named_weights with {len(update_info['names'])} names, packed={update_info['packed']}"
            )
            result = await client.update_named_weights(update_info)
            print(f"[Step 3] update_named_weights returned: {list(result.keys())}")
            for server_url, resp in result.items():
                assert resp["status"] == 200, f"Server {server_url} update weights failed: {resp}"

            # Trainer should be done now (NCCL broadcast complete)
            ray.get(trainer_broadcast_ref)
            print("[Step 3] Weight sync complete")

            # ===== Step 4: Query again - should produce correct output =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 5] Real weights output: {text_after!r}")

            assert "Paris" in text_after, f"Weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] Non-colocated weight sync test passed!")


# -----------------------------------------------------------------
# Colocated CUDA IPC Weight Sync Test
# -----------------------------------------------------------------


@ray.remote
class IpcTrainer:
    """
    Trainer emulator that creates CUDA IPC handles for weight transfer.

    Unlike the NCCL Trainer, this does not create a process group.
    Instead it creates per-tensor IPC handles that the colocated
    inference server opens to read weights directly from GPU memory.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self._tensor_refs: list = []

    def ready(self):
        return True

    def create_ipc_update_info(self) -> dict:
        """Create IPC handles for all model parameters.

        Returns a dict matching the /update_weights API contract:
        names, dtype_names, shapes, and ipc_handles_pickled (base64).
        """
        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        names, dtype_names, shapes = [], [], []
        ipc_handles = []
        tensor_refs = []

        for name, param in self.model.named_parameters():
            weight = param.detach().contiguous()
            tensor_refs.append(weight)
            handle = reduce_tensor(weight)
            ipc_handles.append({gpu_uuid: handle})
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))

        # Prevent GC so IPC handles remain valid
        self._tensor_refs = tensor_refs

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": pickled,
        }


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
        use_new_inference_servers=True,
        engine_init_kwargs={"load_format": "dummy"},
    )

    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        # Trainer on same PG bundle as server (colocated) with fractional GPU
        trainer = IpcTrainer.options(
            num_gpus=0.2,
            num_cpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=engines.pg,
                placement_group_bundle_index=0,
            ),
        ).remote(MODEL)
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": engines.client,
            "router_url": engines.client.proxy_url,
        }

        await engines.client.teardown()
        ray.kill(trainer)
    # cleanup manually in colocated case
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


@pytest.mark.asyncio(loop_scope="class")
class TestColocatedIpcWeightUpdateFlow:
    """Tests for weight synchronization via CUDA IPC (colocated, TP=1)."""

    async def test_update_weights_ipc(self, ipc_weight_update_env):
        """
        Full E2E weight sync test (colocated, CUDA IPC):
        1. Query with dummy weights → gibberish
        2. Init IPC weight transfer engine (no-op for IPC)
        3. Create IPC handles from trainer weights and send to server
        4. Query again → correct output
        """
        router_url = ipc_weight_update_env["router_url"]
        trainer = ipc_weight_update_env["trainer"]
        client = ipc_weight_update_env["client"]

        print("\n[TEST] Running colocated IPC weight sync test")

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
            # ===== Step 1: Verify dummy weights produce gibberish =====
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_before = resp.json()["choices"][0]["text"]
            print(f"[Step 1] Dummy weights output: {text_before!r}")
            assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 2: Init IPC engine (no-op but verifies endpoint) =====
            init_info = CudaIpcInitInfo(
                model_dtype_str="bfloat16",
                override_existing_receiver=True,
            )
            result = await client.init_weight_update_communicator(init_info)
            for server_url, resp_data in result.items():
                assert resp_data["status"] == 200, f"Server {server_url} IPC init failed: {resp_data}"
            print("[Step 2] IPC engine init complete (no-op)")

            # ===== Step 3: Create IPC handles and send to server =====
            print("[Step 3] Creating IPC handles from trainer weights...")
            update_info = ray.get(trainer.create_ipc_update_info.remote())
            print(f"[Step 3] Created handles for {len(update_info['names'])} parameters")

            result = await client.update_named_weights(update_info)
            for server_url, resp_data in result.items():
                assert resp_data["status"] == 200, f"Server {server_url} IPC update failed: {resp_data}"
            print("[Step 3] IPC weight update complete")

            # ===== Step 4: Query again — should produce correct output =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 4] Real weights output: {text_after!r}")
            assert "Paris" in text_after, f"IPC weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] Colocated IPC weight sync test passed!")


# Small unquantized MoE that triggers https://github.com/vllm-project/vllm/issues/42821
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
async def test_worker_wrap_load_weights_preserves_moe_forward() -> None:
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
            await engine.collective_rpc("skyrl_start_weight_update", args=(is_checkpoint_format,))

        async def update_named_weights(self, request: object) -> None:
            await engine.collective_rpc("load_weights", args=(pickle.dumps(request),))

        async def finish_weight_update(self) -> None:
            await engine.collective_rpc("skyrl_finish_weight_update")

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
