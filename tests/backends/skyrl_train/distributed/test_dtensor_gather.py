"""CPU (gloo) test: bucketed gather of FSDP2-style Shard(0) DTensors.

The in-memory LoRA sync gathered every adapter tensor with its own ``full_tensor()``
(55k collectives per sync on the full ML4 model). ``iter_full_tensors`` must return the
same bytes with one all-gather per bucket, and ``FsdpLoraAdapterSource`` must use it.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/distributed/test_dtensor_gather.py
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
# dim 0 not divisible by 4, smaller than 4 (empty trailing shards), 1-D, and a large one
SHAPES = [(32, 7168), (4096, 32), (10, 3), (3, 5), (5,), (1, 9), (64, 64), (7, 2, 3)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# PEFT (lora_A (r, in), lora_B (out, r)) pairs, r = 4; B (3, 4) leaves a trailing rank empty
LORA_SHAPES = {
    "m0": ((4, 7168), (4096, 4)),
    "m1": ((4, 10), (3, 4)),
    "m2": ((4, 6), (10, 4)),
}


def _tensors(mode: str):
    g = torch.Generator().manual_seed(0)
    if mode == "padding":
        # one row each: at world 4 every tensor gathers 4 rows (3 of them padding)
        return {"a": torch.randn(1, 2, generator=g), "b": torch.randn(1, 2, generator=g)}
    if mode == "gather":
        return {f"t{i}": torch.randn(*s, generator=g) for i, s in enumerate(SHAPES)}
    out = {}
    for m, (a, b) in LORA_SHAPES.items():
        out[f"base_model.model.{m}.lora_A.weight"] = torch.randn(*a, generator=g)
        out[f"base_model.model.{m}.lora_B.weight"] = torch.randn(*b, generator=g)
    return out


def _worker(rank: int, port: int, mode: str, out: dict) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard, distribute_tensor

        mesh = init_device_mesh("cpu", (WORLD,))
        full = _tensors(mode)
        shards = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in full.items()}
        calls = {"full_tensor": 0, "all_gather_into_tensor": 0}
        orig_full, orig_ag = DTensor.full_tensor, dist.all_gather_into_tensor

        def counting_full(self, *a, **k):
            calls["full_tensor"] += 1
            return orig_full(self, *a, **k)

        def counting_ag(*a, **k):
            calls["all_gather_into_tensor"] += 1
            return orig_ag(*a, **k)

        DTensor.full_tensor = counting_full
        dist.all_gather_into_tensor = counting_ag
        try:
            if mode in ("gather", "padding"):
                from skyrl.backends.skyrl_train.distributed.dtensor_gather import (
                    iter_full_tensors,
                )

                # padding: each (1, 2) fp32 tensor needs 4 * 1 * 2 * 4 = 32 receive bytes
                got = dict(iter_full_tensors(shards.items(), bucket_bytes=32 if mode == "padding" else 2**20))
            else:
                from skyrl.backends.skyrl_train.weight_sync.sources import (
                    FsdpLoraAdapterSource,
                )

                src = FsdpLoraAdapterSource(
                    dtype=torch.bfloat16,
                    adapter_state=lambda: shards,
                    adapter_config={"r": 4, "lora_alpha": 8},
                )
                src.prepare()
                got = {n: t.clone() for n, t in src}
                # the per-tensor reference: the old loop's full_tensor(), folds and cast
                full = {n: src.finalize_adapter({n: orig_full(v)})[0][n].to(torch.bfloat16) for n, v in shards.items()}
        finally:
            DTensor.full_tensor = orig_full
            dist.all_gather_into_tensor = orig_ag
        out[f"contiguous{rank}"] = all(v.is_contiguous() for v in got.values())
        out[f"equal{rank}"] = all(torch.equal(got[k], full[k]) for k in full)
        if rank == 0:
            out["equal"] = {k: torch.equal(got[k], full[k]) and got[k].shape == full[k].shape for k in full}
            out["order"] = list(got) == list(full)
            out["calls"] = calls
    finally:
        dist.destroy_process_group()


def _run(mode: str) -> dict:
    with mp.Manager() as m:
        out = m.dict()
        mp.spawn(_worker, args=(_free_port(), mode, out), nprocs=WORLD, join=True)
        return dict(out)


class TestBucketedGather:
    def test_bucket_budget_counts_padding(self):
        out = _run("padding")
        assert all(out[f"equal{r}"] and out[f"contiguous{r}"] for r in range(WORLD))
        # two 32-byte receive buffers do not fit one 32-byte bucket
        assert out["calls"]["all_gather_into_tensor"] == 2

    def test_iter_full_tensors_bitwise_with_few_collectives(self):
        out = _run("gather")
        assert out["order"]
        assert all(out["equal"].values()), out["equal"]
        assert all(out[f"equal{r}"] and out[f"contiguous{r}"] for r in range(WORLD))
        assert out["calls"]["full_tensor"] == 0
        # 1 MiB buckets: (32, 7168) fp32 alone fills one bucket; the rest share at most two more
        assert 1 <= out["calls"]["all_gather_into_tensor"] <= 3

    def test_fsdp_lora_source_batches_and_matches_per_tensor(self):
        pytest.importorskip("vllm.distributed.weight_transfer.base")
        out = _run("source")
        assert out["order"]
        assert all(out["equal"].values()), out["equal"]
        # the old source called full_tensor() once per tensor (6 here)
        assert out["calls"]["full_tensor"] == 0
        assert out["calls"]["all_gather_into_tensor"] == 1
