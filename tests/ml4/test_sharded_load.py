"""``ModelConfig.sharded_load_fn`` path: meta build via auto_map + PEFT + FSDP2 + per-rank load.

Two spawned ranks use the in-process fake process group (no gloo on Windows CPU builds), so
nothing here runs a real collective: every rank's local shards are compared with a plain
single-process load.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LOAD_FN = "loading.load_sharded"


def _worker(rank, world, model_dir, out_dir):
    from conftest import build_ml4_wrapper, install_stubs

    install_stubs()
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from torch.testing._internal.distributed.fake_pg import FakeStore

    from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
        fsdp2_load_sharded,
        resolve_sharded_load_fn,
    )
    from skyrl.train.config import FSDPConfig

    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    wrapper = build_ml4_wrapper(model_dir)
    load_fn = resolve_sharded_load_fn(LOAD_FN, model_dir)
    fsdp2_load_sharded(wrapper.model, {"mesh": mesh}, FSDPConfig(), load_fn, model_dir, torch.device("cpu"))
    from torch.distributed.fsdp import FSDPModule

    assert isinstance(wrapper.model.get_output_embeddings(), FSDPModule)
    out = {}
    for n, p in wrapper.model.named_parameters():
        _, off = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
        out[n] = (off[0], p.to_local().clone())
    out.update({n: (0, b.clone()) for n, b in wrapper.model.named_buffers()})
    torch.save(out, os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="no torch.distributed")
def test_sharded_load_matches_single_process(ml4_model_dir, tmp_path):
    from conftest import load_ml4_wrapper
    from ml4_train.lora import adapter_module_names

    mp.spawn(_worker, args=(2, ml4_model_dir, str(tmp_path)), nprocs=2)

    wrapper, lora_init = load_ml4_wrapper(ml4_model_dir)
    ref = wrapper.model
    assert lora_init and not any(t.is_meta for t in lora_init.values())
    ref_t = dict(ref.named_parameters()) | dict(ref.named_buffers())

    rows = dict.fromkeys(ref_t, 0)
    for r in range(2):
        got = torch.load(tmp_path / f"rank{r}.pt")
        assert set(got) == set(ref_t)
        for n, (off, t) in got.items():
            torch.testing.assert_close(t, ref_t[n][off : off + t.shape[0]], atol=0, rtol=0, msg=n)
            if ref_t[n].dim() > 0 and "lerope" not in n:
                rows[n] += t.shape[0]
    assert all(rows[n] == ref_t[n].shape[0] for n in rows if rows[n])

    # PEFT init survived sharding: kaiming A, zero B; every adapter is a contract module.
    a = [t for n, t in lora_init.items() if ".lora_A." in n]
    b = [t for n, t in lora_init.items() if ".lora_B." in n]
    assert all(t.abs().sum() > 0 for t in a) and all(t.abs().sum() == 0 for t in b)
    from peft.utils.save_and_load import get_peft_model_state_dict

    assert adapter_module_names(get_peft_model_state_dict(ref))


def _host_peak_worker(rank, world, nbytes, q):
    import resource
    from types import SimpleNamespace

    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    from skyrl.backends.skyrl_train.distributed import fsdp_utils
    from skyrl.train.config import FSDPConfig

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(64, 64, device="meta")
            self.adapter = nn.Parameter(torch.randn(nbytes // 4 // 1024, 1024))

    class Model(nn.Module):
        _no_split_modules = ["Block"]

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)
            self.block = Block()

    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    model = Model()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def load_fn(m, _path):
        for p in m.block.base.parameters():
            p.to_local().zero_()

    fsdp_utils.drop_checkpoint_page_cache = lambda _path: None
    fsdp_utils.fsdp2_load_sharded(model, {"mesh": mesh}, FSDPConfig(), load_fn, "unused", torch.device("cpu"))
    q.put(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - before)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or os.name == "nt", reason="needs torch.distributed and ru_maxrss")
def test_sharded_load_keeps_one_host_copy_of_initialized_tensors():
    """Pre-initialized (LoRA) tensors are held by reference while sharding: the host peak grows by
    about one rank's shard, not by a second full copy (ML4 full adapters are ~50 GB per rank)."""
    nbytes = 512 * 2**20
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_host_peak_worker, args=(0, 8, nbytes, q))
    p.start()
    grew = q.get(timeout=300)
    p.join()
    assert p.exitcode == 0
    assert grew < nbytes / 2, f"host peak grew by {grew / 2**20:.0f} MiB for a {nbytes / 2**20:.0f} MiB adapter"
