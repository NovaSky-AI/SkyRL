"""Shared-outer expert LoRA ties are checked where they can silently break: after FSDP init
(both paths) and after a checkpoint load. Two spawned ranks on the fake process group."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LOAD_FN = "loading.load_sharded"


def _strategy(world, mesh, model_dir, sharded_load_fn):
    from conftest import LORA_RANK
    from torch.distributed.fsdp import OffloadPolicy

    from skyrl.backends.skyrl_train.distributed import fsdp_strategy
    from skyrl.train.config import FSDPConfig, ModelConfig

    # CPU-only build: cpu_offload selects the CPU device; the no-op policy keeps FSDP from pinning
    fsdp_strategy.CPUOffloadPolicy = lambda **kw: OffloadPolicy()

    model_config = ModelConfig()
    model_config.lora.rank = LORA_RANK
    model_config.path, model_config.sharded_load_fn = model_dir, sharded_load_fn
    strategy = fsdp_strategy.FSDPStrategy(FSDPConfig(cpu_offload=True), model_config=model_config)
    strategy.world_size, strategy.device_mesh = world, mesh
    return strategy


def _worker(rank, world, model_dir, ckpt_dir, path, q):
    from conftest import build_ml4_wrapper, install_stubs, load_ml4_wrapper

    install_stubs()
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world)
    torch.cuda.synchronize = lambda *a, **k: None  # CPU-only build
    mesh = init_device_mesh("cpu", (world,))
    try:
        if path == "full-state":
            # no sharded_load_fn: the base-model full state dict path
            wrapper, _ = load_ml4_wrapper(model_dir, True)
            strategy = _strategy(world, mesh, model_dir, None)
            strategy._fsdp_init_model(wrapper, is_wrapped=True)
            q.put((rank, "ok"))
            return
        wrapper = build_ml4_wrapper(model_dir, True)
        strategy = _strategy(world, mesh, model_dir, LOAD_FN)
        model = strategy._fsdp_init_model(wrapper, is_wrapped=True)
        wrapper.model = model
        assert wrapper.shared_outer_factors > 0
        strategy.save_checkpoint(wrapper, ckpt_dir, node_local_rank=rank)
        strategy.load_checkpoint(wrapper, ckpt_dir, load_optimizer_states=False, load_lr_scheduler_states=False)
        if path == "load-unties":
            from test_shared_outer_lora import _moe_layers

            load = strategy._load_trainable_state_dict

            def load_then_untie(model, sd):  # a load that rebuilds one expert's shared factor
                load(model, sd)
                lin = _moe_layers(model)[0].mlp.experts[1].down_proj
                lin.lora_B["default"].weight = torch.nn.Parameter(lin.lora_B["default"].weight.detach().clone())

            strategy._load_trainable_state_dict = load_then_untie
            strategy.load_checkpoint(wrapper, ckpt_dir, load_optimizer_states=False, load_lr_scheduler_states=False)
        q.put((rank, "ok"))
    except RuntimeError as e:
        q.put((rank, str(e)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="no torch.distributed")
@pytest.mark.parametrize(
    "path,want",
    [
        ("sharded", "ok"),
        ("full-state", "after FSDP full-state init"),
        ("load-unties", "after checkpoint load"),
    ],
    ids=["sharded-load-and-resume", "full-state-init", "load-unties"],
)
def test_ties_checked(ml4_model_dir, tmp_path, path, want):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world = 1 if path == "full-state" else 2  # the fake PG has no rank-0 broadcast for the full state
    mp.spawn(_worker, args=(world, ml4_model_dir, str(tmp_path), path, q), nprocs=world)
    got = dict(q.get(timeout=60) for _ in range(world))
    assert all(want in v for v in got.values()), got
