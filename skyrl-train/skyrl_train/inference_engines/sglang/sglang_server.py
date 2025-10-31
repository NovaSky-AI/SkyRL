import os
import sys
import base64
import pickle
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI, Request

from sglang.srt.server_args import prepare_server_args, ServerArgs
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import kill_process_tree
from sglang.srt.managers.tokenizer_manager import (
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

from sglang.srt.utils import MultiprocessingSerializer


def build_engine_from_args(server_args: ServerArgs) -> Engine:
    kwargs = {
        "model_path": server_args.model_path,
        "tp_size": server_args.tp_size,
        "mem_fraction_static": server_args.mem_fraction_static,
        "random_seed": server_args.random_seed,
        "disable_radix_cache": server_args.disable_radix_cache,
        "dtype": server_args.dtype,
        "trust_remote_code": server_args.trust_remote_code,
        "max_prefill_tokens": server_args.max_prefill_tokens,
        "max_running_requests": server_args.max_running_requests,
        "mm_attention_backend": server_args.mm_attention_backend,
        "attention_backend": server_args.attention_backend,
        "enable_memory_saver": server_args.enable_memory_saver,
        # Ensure token-in-token-out
        "skip_tokenizer_init": True,
        # Register our custom CUDA IPC weight loader so load_format is recognized
        "custom_weight_loader": ["skyrl_train.inference_engines.sglang.sglang_engine.update_weights_cuda_ipc"],
    }
    return Engine(**kwargs)


def create_app(engine: Engine, server_args: ServerArgs) -> FastAPI:
    app = FastAPI()

    @app.get("/get_model_info")
    async def _get_model_info():
        return {"model": server_args.model_path, "tp_size": server_args.tp_size}

    @app.post("/generate")
    async def _generate(request: Request):
        data = await request.json()
        input_ids = data.get("input_ids")
        sampling_params = data.get("sampling_params", {})
        outputs = await engine.async_generate(input_ids=input_ids, sampling_params=sampling_params)
        return outputs

    @app.post("/init_weights_update_group")
    async def _init_weights_update_group(request: Request):
        data = await request.json()
        obj = InitWeightsUpdateGroupReqInput(
            master_address=data.get("master_address"),
            master_port=data.get("master_port"),
            rank_offset=data.get("rank_offset"),
            world_size=data.get("world_size"),
            group_name=data.get("group_name"),
            backend=data.get("backend"),
        )
        success, message = await engine.tokenizer_manager.init_weights_update_group(obj, None)
        return {"status": "ok", "success": success, "message": message}

    @app.post("/update_weights_from_distributed")
    async def _update_weights_from_distributed(request: Request):
        data = await request.json()
        obj = UpdateWeightsFromDistributedReqInput(
            name=data.get("name"), dtype=data.get("dtype"), shape=data.get("shape")
        )
        success, message = await engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        return {"status": "ok", "success": success, "message": message}

    @app.post("/update_weights_cuda_ipc")
    async def _update_weights_cuda_ipc(request: Request):
        data = await request.json()
        names: List[str] = data.get("names", [])
        dtypes: List[str] = data.get("dtypes", [])
        shapes: List[List[int]] = data.get("shapes", [])
        ipc_handles_b64: List[str] = data.get("ipc_handles_b64", [])

        # Decode IPC handle dicts
        ipc_handles_list: List[Dict[str, Any]] = [pickle.loads(base64.b64decode(s)) for s in ipc_handles_b64]

        # Build a NamedWeightsUpdateRequest-like payload expected by our custom loader.
        request_obj: Dict[str, Any] = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "extras": [{"ipc_handles": h} for h in ipc_handles_list],
        }

        # Serialize into a byte tensor as expected by the custom loader
        req_bytes = pickle.dumps(request_obj)
        req_b64 = base64.b64encode(req_bytes)
        end_marker = b"__END_OF_REQUEST__"
        data_with_marker = req_b64 + end_marker
        tensor_data = bytearray(data_with_marker)
        # Pad to 4-byte alignment (optional but consistent with client-side code)
        pad = (-len(tensor_data)) % 4
        if pad:
            tensor_data.extend(b"\x00" * pad)
        tensor_array = torch.tensor(list(tensor_data), dtype=torch.uint8)

        named_tensors = [("ipc_request", tensor_array)]
        serialized = MultiprocessingSerializer.serialize(named_tensors)
        # SGLang expects a list per TP rank; but HTTP server is single process with TP=1.
        # Use length 1 list regardless of tp_size to avoid mismatches.
        serialized_list = [serialized]

        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_list,
            load_format="skyrl_train.inference_engines.sglang.sglang_engine.update_weights_cuda_ipc",
            flush_cache=False,
        )

        success, message = await engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        return {"status": "ok" if success else "error", "message": message}

    @app.post("/destroy_weights_update_group")
    async def _destroy_weights_update_group(request: Request):  # noqa: F841
        # No-op; SGLang does not expose a destroy API here.
        return {"status": "ok"}

    @app.post("/resume_memory_occupation")
    async def _resume_memory_occupation(request: Request):
        data = await request.json()
        obj = ResumeMemoryOccupationReqInput(tags=data.get("tags", None))
        await engine.tokenizer_manager.resume_memory_occupation(obj, None)
        return {"status": "ok"}

    @app.post("/release_memory_occupation")
    async def _release_memory_occupation(request: Request):
        data = await request.json()
        obj = ReleaseMemoryOccupationReqInput(tags=data.get("tags", None))
        await engine.tokenizer_manager.release_memory_occupation(obj, None)
        return {"status": "ok"}

    @app.post("/flush_cache")
    async def _flush_cache():
        await engine.tokenizer_manager.flush_cache()
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    args = sys.argv[1:]
    # SGLang requires `skip-tokenizer-init` to do token-in-token-out with `/generate` endpoint
    if "--skip-tokenizer-init" not in args:
        args.append("--skip-tokenizer-init")
    server_args = prepare_server_args(args)

    # Initialize SGLang Engine
    engine = build_engine_from_args(server_args)

    # Build FastAPI app
    app = create_app(engine, server_args)

    try:
        uvicorn.run(app, host=server_args.host or "0.0.0.0", port=server_args.port or 8000, log_level="info")
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
