"""Opt-in LoRA sync manifests: name, shape, dtype, bytes and sha256 of every adapter tensor.

Set ``SKYRL_LORA_MANIFEST_DIR`` to record what the trainer put on the wire
(the LoRA weight sources, rank 0) and what each vLLM worker built the adapter
from (``patch_lora_in_memory``). Unset, nothing is computed.

Hashing the full adapter costs tens of seconds per sync on large MoE adapters
(sender and every worker). ``SKYRL_LORA_MANIFEST_EVERY=N`` records the first
sync and every N-th one after it: each process counts its own syncs, and the
sender and the workers see the same sequence of syncs, so the recorded rounds
still pair up by their per-process counter.
"""

import hashlib
import itertools
import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

ENV = "SKYRL_LORA_MANIFEST_DIR"
EVERY_ENV = "SKYRL_LORA_MANIFEST_EVERY"
_COUNTER = itertools.count()
_SYNCS = itertools.count()


def enabled() -> bool:
    """Whether to record this sync. Call once per sync (it counts them)."""
    if not os.environ.get(ENV):
        return False
    every = max(1, int(os.environ.get(EVERY_ENV) or 1))
    return next(_SYNCS) % every == 0


def tensor_entry(t: torch.Tensor) -> Dict[str, Any]:
    data = t.detach().contiguous()
    raw = data.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()  # reshape: 0-dim tensors too
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype).removeprefix("torch."),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write(side: str, lora_name: str, entries: Dict[str, Dict[str, Any]], config: Optional[dict] = None) -> str:
    out_dir = os.environ[ENV]
    os.makedirs(out_dir, exist_ok=True)
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(name.encode())
        digest.update(entries[name]["sha256"].encode())
    path = os.path.join(out_dir, f"{side}-{lora_name}-pid{os.getpid()}-{next(_COUNTER)}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "side": side,
                "lora_name": lora_name,
                "count": len(entries),
                "total_bytes": sum(e["bytes"] for e in entries.values()),
                "adapter_sha256": digest.hexdigest(),
                "config": config,
                "tensors": entries,
            },
            f,
            indent=1,
        )
    return path


def record(side: str, lora_name: str, items: Iterable[Tuple[str, torch.Tensor]], config: Optional[dict] = None):
    return write(side, lora_name, {name: tensor_entry(t) for name, t in items}, config)
