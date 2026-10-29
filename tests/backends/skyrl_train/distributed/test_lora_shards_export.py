"""CPU test: rebuild the PEFT adapter from FSDP LoRA rank shards (``lora_shards_export``).

Each gloo rank saves its trainable state dict the way ``FSDPStrategy.save_checkpoint`` does
(``Shard(0)`` DTensors, ``model_world_size_<W>_rank_<r>.pt``); the converter then runs in this
process, with no process group, and must return the full tensors under PEFT names.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/distributed/test_lora_shards_export.py
"""

import json
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
# lora_A (r, in), lora_B (out, r); (3, 4) leaves trailing ranks empty
SHAPES = {
    "base_model.model.layers.0.q_proj.lora_A.default.weight": (4, 10),
    "base_model.model.layers.0.q_proj.lora_B.default.weight": (3, 4),
    "base_model.model.layers.0.o_proj.lora_A.default.weight": (4, 6),
    "base_model.model.layers.0.o_proj.lora_B.default.weight": (10, 4),
}


def _full():
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(*s, generator=g) for k, s in SHAPES.items()}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _save_worker(rank: int, port: int, policy_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        mesh = init_device_mesh("cpu", (WORLD,))
        sd = {k: distribute_tensor(v, mesh, [Shard(0)]).detach() for k, v in _full().items()}
        torch.save(sd, os.path.join(policy_dir, f"model_world_size_{WORLD}_rank_{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestLoraShardsExport:
    def test_export_from_rank_shards(self, tmp_path):
        from safetensors.torch import load_file

        from skyrl.backends.skyrl_train.distributed.lora_shards_export import (
            ADAPTER_CONFIG_FILE,
            ALIASES_FILE,
            export,
        )

        policy = tmp_path / "policy"
        policy.mkdir()
        mp.spawn(_save_worker, args=(_free_port(), str(policy)), nprocs=WORLD, join=True)
        config = {"r": 4, "lora_alpha": 8, "peft_type": "LORA"}
        (policy / ADAPTER_CONFIG_FILE).write_text(json.dumps(config))
        # a shared-outer factor: saved once, exported under every expert's name
        alias = "base_model.model.layers.0.o_proj_e1.lora_A.default.weight"
        canon = "base_model.model.layers.0.o_proj.lora_A.default.weight"
        (policy / ALIASES_FILE).write_text(json.dumps({alias: canon}))

        out = export(str(policy))
        got = load_file(os.path.join(out, "adapter_model.safetensors"))
        expected = {k.replace(".default", ""): v for k, v in _full().items()}
        expected[alias.replace(".default", "")] = expected[canon.replace(".default", "")]
        assert set(got) == set(expected)
        for k, v in expected.items():
            assert got[k].dtype == v.dtype and torch.equal(got[k], v), k
        assert json.loads(open(os.path.join(out, "adapter_config.json")).read()) == config

    def test_aliases_of_tied_parameters(self):
        from torch import nn

        from skyrl.backends.skyrl_train.distributed.lora_shards_export import (
            lora_param_aliases,
        )

        m = nn.ModuleDict(
            {"e0": nn.ModuleDict({"lora_A": nn.Linear(2, 2)}), "e1": nn.ModuleDict({"lora_A": nn.Linear(2, 2)})}
        )
        m["e1"]["lora_A"].weight = m["e0"]["lora_A"].weight
        assert lora_param_aliases(m) == {"e1.lora_A.weight": "e0.lora_A.weight"}
