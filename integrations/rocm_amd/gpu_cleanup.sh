#!/usr/bin/env bash
# Best-effort cleanup between verify steps on shared GPU nodes.
set -euo pipefail

ray stop --force 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLMServerActor" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2

python3 - <<'PY' || true
import gc

try:
    import torch

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU0 after cleanup: free={free/1e9:.1f}GiB total={total/1e9:.1f}GiB")
except Exception as exc:
    print("gpu_cleanup:", exc)
PY
