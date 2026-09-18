#!/usr/bin/env bash
# Quick vLLM LLM init smoke inside a ROCm container (sync engine via Ray actor, like GRPO).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_NAME:=Qwen/Qwen2.5-0.5B-Instruct}"
export VLLM_SMOKE_GPU="${VLLM_SMOKE_GPU:-0}"
export HIP_VISIBLE_DEVICES="${VLLM_SMOKE_GPU}"
export ROCR_VISIBLE_DEVICES="${VLLM_SMOKE_GPU}"
export CUDA_VISIBLE_DEVICES="${VLLM_SMOKE_GPU}"

export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

bash integrations/rocm_amd/gpu_cleanup.sh

python3 - <<'PY'
import os
import ray
import torch

print("torch", torch.__version__, "hip", torch.version.hip)
print("VLLM_USE_V1", os.environ.get("VLLM_USE_V1"))
print("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))

model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
free, total = torch.cuda.mem_get_info(0)
# Shared nodes may have less free VRAM; cap utilization from available memory.
max_util = float(os.environ.get("VLLM_SMOKE_GPU_UTIL", "0.12"))
dynamic_util = min(max_util, (free * 0.75) / total)
print(f"GPU0 free={free/1e9:.1f}GiB total={total/1e9:.1f}GiB gpu_memory_utilization={dynamic_util:.3f}")

runtime_env = {
    "env_vars": {
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "0"),
        "VLLM_TARGET_DEVICE": "rocm",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "1"),
        "VLLM_DISABLE_COMPILE_CACHE": os.environ.get("VLLM_DISABLE_COMPILE_CACHE", "1"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", "0"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
}

ray.init(num_gpus=1, include_dashboard=False, logging_level="error", runtime_env=runtime_env)


@ray.remote(num_gpus=1)
def _vllm_smoke(model_name: str, gpu_util: float) -> str:
    import os
    import vllm
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_V1"] = "0"
    print("worker vllm", vllm.__version__)
    print("worker CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=gpu_util,
        max_model_len=512,
        trust_remote_code=True,
    )
    out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
    return out[0].outputs[0].text


text = ray.get(_vllm_smoke.remote(model, dynamic_util))
ray.shutdown()
print("generated:", text)
print("PASS: vLLM LLM smoke (Ray actor)")
PY
