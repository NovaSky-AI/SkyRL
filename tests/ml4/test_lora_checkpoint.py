"""FSDP LoRA checkpoint: adapter + optimizer + scheduler + RNG only, never the frozen base.

Two spawned ranks on the fake process group (as in ``test_sharded_load``). Each rank: sharded
load -> one optimizer step on the adapters -> ``save_checkpoint``; then a fresh model is built
the way resume builds it (base from the original checkpoint via ``sharded_load_fn``, PEFT-initialized
adapters) and ``load_checkpoint`` must restore the adapter, optimizer, scheduler and RNG exactly.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LOAD_FN = "loading.load_sharded"


def _build(model_dir, mesh, strategy, shared_outer):
    from conftest import build_ml4_wrapper
    from transformers.trainer import get_scheduler

    from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
        fsdp2_load_sharded,
        resolve_sharded_load_fn,
    )

    wrapper = build_ml4_wrapper(model_dir, shared_outer)
    load_fn = resolve_sharded_load_fn(LOAD_FN, model_dir)
    fsdp2_load_sharded(wrapper.model, {"mesh": mesh}, strategy.fsdp_config, load_fn, model_dir, torch.device("cpu"))
    optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=1e-3)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=2, num_training_steps=10)
    return wrapper, optimizer, scheduler


def _local(model, trainable: bool):
    return {n: p.to_local().clone() for n, p in model.named_parameters() if p.requires_grad == trainable}


def _assert_outer_tied(model):
    """Shared-outer factors are still one tensor per MoE layer after FSDP2 sharding and loading."""
    from test_shared_outer_lora import OUTER, _factor, _moe_layers

    for layer in _moe_layers(model):
        e0 = layer.mlp.experts[0]
        assert all(_factor(e, p, s) is _factor(e0, p, s) for e in layer.mlp.experts for p, s in OUTER)


def _worker(rank, world, model_dir, ckpt_dir, shared_outer):
    from conftest import LORA_RANK, install_stubs

    install_stubs()
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl.train.config import FSDPConfig, ModelConfig

    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world)
    torch.cuda.synchronize = lambda *a, **k: None  # CPU-only build
    mesh = init_device_mesh("cpu", (world,))
    model_config = ModelConfig()
    model_config.lora.rank = LORA_RANK
    strategy = FSDPStrategy(FSDPConfig(), model_config=model_config)
    strategy.world_size = world

    wrapper, optimizer, scheduler = _build(model_dir, mesh, strategy, shared_outer)
    if shared_outer:
        _assert_outer_tied(wrapper.model)
    for p in wrapper.model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    adapter, base = _local(wrapper.model, True), _local(wrapper.model, False)
    assert adapter and base
    strategy.save_checkpoint(wrapper, ckpt_dir, node_local_rank=rank, optimizer=optimizer, scheduler=scheduler)
    after_save = torch.rand(4)

    saved = torch.load(os.path.join(ckpt_dir, f"model_world_size_{world}_rank_{rank}.pt"), weights_only=False)
    assert set(saved) == set(adapter), "checkpoint must hold exactly the trainable params"

    torch.manual_seed(1234)  # resume starts from a different RNG state
    new, new_opt, new_sched = _build(model_dir, mesh, strategy, shared_outer)
    assert any(not torch.equal(t, adapter[n]) for n, t in _local(new.model, True).items())
    strategy.load_checkpoint(new, ckpt_dir, optimizer=new_opt, scheduler=new_sched)

    for n, t in _local(new.model, True).items():
        torch.testing.assert_close(t, adapter[n], atol=0, rtol=0, msg=n)
    for n, t in _local(new.model, False).items():
        torch.testing.assert_close(t, base[n], atol=0, rtol=0, msg=n)  # base rebuilt from the model path
    old_sd, new_sd = optimizer.state_dict(), new_opt.state_dict()
    assert old_sd["param_groups"] == new_sd["param_groups"]
    assert len(new_sd["state"]) == len(adapter)
    for i, st in old_sd["state"].items():
        for k, v in st.items():
            w = new_sd["state"][i][k]
            v, w = (x.to_local() if hasattr(x, "to_local") else x for x in (v, w))
            torch.testing.assert_close(w, v, atol=0, rtol=0)
    assert new_sched.state_dict() == scheduler.state_dict()
    if shared_outer:
        _assert_outer_tied(new.model)
    assert torch.equal(torch.rand(4), after_save)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="no torch.distributed")
@pytest.mark.parametrize("shared_outer", [False, True], ids=["per-expert", "shared-outer"])
def test_lora_checkpoint_round_trip(ml4_model_dir, tmp_path, shared_outer):
    mp.spawn(_worker, args=(2, ml4_model_dir, str(tmp_path), shared_outer), nprocs=2)
