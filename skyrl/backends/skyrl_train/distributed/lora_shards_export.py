"""Rebuild the PEFT-format adapter of an FSDP LoRA checkpoint from its per-rank shards.

With ``lora.checkpoint_export=final`` only the last checkpoint carries ``lora_adapter/``. A checkpoint
written before that (a run that crashed, or one picked for evaluation) holds the adapter as the
``model_world_size_<W>_rank_<r>.pt`` shards plus ``lora_adapter_config.json`` and
``lora_param_aliases.json`` (both written with every save that skips the export). This writes the same
``lora_adapter/adapter_model.safetensors`` + ``adapter_config.json`` that
``FSDPStrategy._save_lora_adapters`` would have written: same names (PEFT drops the adapter name), same
dtype, same values (the shards are FSDP2 ``Shard(0)`` pieces, concatenated in rank order).

CPU only, no process group. usage::

    python -m skyrl.backends.skyrl_train.distributed.lora_shards_export <ckpt>/global_step_N/policy [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from collections import OrderedDict
from typing import Dict, List

import torch

ADAPTER_CONFIG_FILE = "lora_adapter_config.json"
#: ``{name: name in the shards}`` for LoRA parameters tied to another one (shared-outer expert factors:
#: the shards hold the tensor once, the PEFT export once per expert).
ALIASES_FILE = "lora_param_aliases.json"


def lora_param_aliases(model: torch.nn.Module) -> Dict[str, str]:
    """Trainable LoRA parameter names that are the same parameter as an earlier name."""
    first: Dict[int, str] = {}
    aliases: Dict[str, str] = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        if not p.requires_grad or "lora_" not in name:
            continue
        if id(p) in first:
            aliases[name] = first[id(p)]
        else:
            first[id(p)] = name
    return aliases


_SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def _local(t: torch.Tensor) -> torch.Tensor:
    to_local = getattr(t, "to_local", None)
    return to_local() if callable(to_local) else t


def merge_rank_shards(
    shards: List[Dict[str, torch.Tensor]], aliases: Dict[str, str] | None = None, adapter_name: str = "default"
) -> "OrderedDict[str, torch.Tensor]":
    """Full LoRA tensors under PEFT names from the rank-ordered trainable state dicts.

    ``Shard(0)`` DTensor values are concatenated along dim 0 (empty trailing shards included);
    plain tensors are taken from rank 0. Names drop ``.<adapter_name>`` as
    ``get_peft_model_state_dict`` does. Each alias gets its own copy of the tensor it names.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    names = list(shards[0])
    for r, sd in enumerate(shards[1:], 1):
        if list(sd) != names:
            raise ValueError(f"rank {r} holds different parameters than rank 0")
    for name in names:
        if "lora_" not in name:
            continue
        v0 = shards[0][name]
        if hasattr(v0, "to_local"):
            full = torch.cat([_local(sd[name]).reshape(-1, *v0.shape[1:]) for sd in shards], dim=0)
            if tuple(full.shape) != tuple(v0.shape):
                raise ValueError(f"{name}: merged shape {tuple(full.shape)} != global {tuple(v0.shape)}")
        else:
            full = v0
        out[name.replace(f".{adapter_name}", "")] = full.detach().contiguous()
    for alias, name in (aliases or {}).items():
        out[alias.replace(f".{adapter_name}", "")] = out[name.replace(f".{adapter_name}", "")].clone()
    return out


def load_rank_shards(policy_dir: str) -> List[Dict[str, torch.Tensor]]:
    files = {}
    for p in glob.glob(os.path.join(policy_dir, "model_world_size_*_rank_*.pt")):
        m = _SHARD_RE.search(p)
        files[int(m.group(2))] = (int(m.group(1)), p)
    if not files:
        raise FileNotFoundError(f"no model_world_size_*_rank_*.pt in {policy_dir}")
    world = {w for w, _ in files.values()}
    if len(world) != 1 or sorted(files) != list(range(world.pop())):
        raise ValueError(f"incomplete or mixed rank shards in {policy_dir}: {sorted(files)}")
    return [torch.load(files[r][1], map_location="cpu", weights_only=False) for r in sorted(files)]


def export(policy_dir: str, out_dir: str | None = None) -> str:
    from safetensors.torch import save_file

    out_dir = out_dir or os.path.join(policy_dir, "lora_adapter")
    config_path = os.path.join(policy_dir, ADAPTER_CONFIG_FILE)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} is missing (written with each save that skips the PEFT export)")
    aliases_path = os.path.join(policy_dir, ALIASES_FILE)
    aliases = json.load(open(aliases_path)) if os.path.exists(aliases_path) else {}
    tensors = merge_rank_shards(load_rank_shards(policy_dir), aliases)
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copyfile(config_path, os.path.join(out_dir, "adapter_config.json"))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("policy_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = export(a.policy_dir, a.out)
    print(json.dumps({"out": out}))


if __name__ == "__main__":
    main()
