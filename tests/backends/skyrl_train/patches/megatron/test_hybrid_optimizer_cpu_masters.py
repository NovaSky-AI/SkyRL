"""``patch_hybrid_optimizer_cpu_masters``: a fully offloaded optimizer keeps no FP32 masters on the GPU.

CPU only. GPU parameters are faked with a tensor subclass that reports ``is_cuda``, and the CUDA
stream/event calls the HybridDeviceOptimizer makes are replaced by no-ops, so megatron-core's real
``HybridDeviceOptimizer`` runs a step end to end: host masters in, BF16 "GPU" params updated.

Run with:
uv run --isolated --extra dev --extra megatron pytest \
    tests/backends/skyrl_train/patches/megatron/test_hybrid_optimizer_cpu_masters.py
"""

import contextlib

import pytest
import torch

pytestmark = pytest.mark.megatron


class FakeCuda(torch.Tensor):
    """A host tensor that claims to live on the GPU (what the HDO branches on)."""

    @property
    def is_cuda(self):
        return True


class _FakeEvent:
    def synchronize(self):
        pass

    def wait(self, stream=None):
        pass


class _FakeStream:
    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, other):
        pass

    def record_event(self):
        return _FakeEvent()


@pytest.fixture
def no_cuda_streams(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())


@pytest.fixture
def patched(monkeypatch):
    """Apply the patch for one test; monkeypatch restores every patched attribute afterwards."""
    import megatron.core.optimizer as optimizer_module
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_hybrid_optimizer_cpu_masters as mod,
    )

    for owner, name in (
        (HybridDeviceOptimizer, "_init_sub_optimizers"),
        (HybridDeviceOptimizer, "_get_sub_optimizer_param_groups"),
        (DistributedOptimizer, "__init__"),
        (optimizer_module, "_get_megatron_optimizer_based_on_param_groups"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(mod, "_APPLIED", False)
    return mod


def _fake_gpu_param(values, dtype=torch.bfloat16):
    return torch.tensor(values, dtype=dtype).as_subclass(FakeCuda)


def _hdo(params, offload_fraction=1.0, **kwargs):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )

    return HybridDeviceOptimizer(
        [{"params": params}],
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        pin_cpu_grads=False,  # no accelerator to pin against
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=0.1,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        **kwargs,
    )


def test_route_main_params_only_when_fully_offloaded():
    from megatron.core.optimizer import OptimizerConfig

    from skyrl.backends.skyrl_train.patches.megatron.patch_hybrid_optimizer_cpu_masters import (
        route_main_params_to_hybrid_optimizer,
    )

    def config(**kwargs):
        base = dict(optimizer="adam", lr=1e-5, bf16=True, use_distributed_optimizer=True)
        base.update(kwargs)
        return OptimizerConfig(**base)

    full = config(optimizer_cpu_offload=True, optimizer_offload_fraction=1.0)
    assert not full.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert route_main_params_to_hybrid_optimizer(full)
    assert full.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not full.use_precision_aware_optimizer  # the user-facing knob is untouched
    assert not route_main_params_to_hybrid_optimizer(full)  # already routed

    for cfg in (
        config(optimizer_cpu_offload=True, optimizer_offload_fraction=0.5),
        config(optimizer_cpu_offload=False, optimizer_offload_fraction=1.0),
        config(optimizer_cpu_offload=True, optimizer_offload_fraction=1.0, use_distributed_optimizer=False),
    ):
        assert not route_main_params_to_hybrid_optimizer(cfg)
        assert not cfg.use_precision_aware_optimizer_no_fp8_or_ds_fp8


def test_host_copy_casts_in_chunks(monkeypatch):
    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_hybrid_optimizer_cpu_masters as mod,
    )

    monkeypatch.setattr(mod, "_COPY_CHUNK_NUMEL", 7)
    src = torch.randn(5, 6).to(torch.bfloat16)
    dst = mod.host_copy(src, torch.float32, pin=True)
    assert dst.dtype == torch.float32 and dst.device.type == "cpu" and dst.shape == src.shape
    assert torch.equal(dst, src.float())
    assert dst.data_ptr() != src.data_ptr()


def test_offloaded_hdo_has_one_host_master_per_param_and_steps(patched, no_cuda_streams):
    patched.apply_hybrid_optimizer_cpu_masters_patch()
    assert patched.apply_hybrid_optimizer_cpu_masters_patch()  # idempotent

    init_a, init_b = [0.5, -1.0, 2.0, 0.25], [[1.0, 2.0], [3.0, -4.0]]
    a, b = _fake_gpu_param(init_a), _fake_gpu_param(init_b)
    hdo = _hdo([a, b])

    # One FP32 host tensor per offloaded param: the CPU optimizer's param, the fp32 master and
    # the "CPU copy" are the same object, and nothing stays on the (fake) GPU.
    assert hdo.gpu_optimizer is None
    for p in (a, b):
        master = hdo.param_to_fp32_param[p]
        assert master is hdo.gpu_params_map_cpu_copy[p]
        assert hdo.cpu_copys_map_gpu_param[master] is p
        assert type(master) is torch.Tensor and not master.is_cuda
        assert master.dtype == torch.float32 and master.shape == p.shape
        assert torch.equal(master, p.float())
    cpu_params = [q for opt in hdo.cpu_optimizers for g in opt.param_groups for q in g["params"]]
    assert {id(q) for q in cpu_params} == {id(hdo.param_to_fp32_param[p]) for p in (a, b)}

    # Grads arrive as BF16 decoupled_grad views (the precision-aware DistributedOptimizer path).
    grad_a, grad_b = torch.tensor([0.1, -0.2, 0.3, 0.4]), torch.tensor([[1.0, -1.0], [0.5, 0.0]])
    a.decoupled_grad = grad_a.to(torch.bfloat16).as_subclass(FakeCuda)
    b.decoupled_grad = grad_b.to(torch.bfloat16).as_subclass(FakeCuda)
    hdo.step()

    ref_a = torch.tensor(init_a, dtype=torch.bfloat16).float().requires_grad_()
    ref_b = torch.tensor(init_b, dtype=torch.bfloat16).float().requires_grad_()
    ref = torch.optim.AdamW([ref_a, ref_b], lr=0.1, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    ref_a.grad = grad_a.to(torch.bfloat16).float()
    ref_b.grad = grad_b.to(torch.bfloat16).float()
    ref.step()

    for p, r in ((a, ref_a), (b, ref_b)):
        assert torch.allclose(hdo.param_to_fp32_param[p], r.detach(), atol=0, rtol=0)  # FP32 master
        assert torch.equal(p.as_subclass(torch.Tensor), r.detach().to(torch.bfloat16))  # copied back
    assert all("master_param" in hdo.state[p] and "exp_avg" in hdo.state[p] for p in (a, b))


def test_partial_offload_keeps_upstream_split(patched, no_cuda_streams):
    patched.apply_hybrid_optimizer_cpu_masters_patch()
    params = [_fake_gpu_param([1.0, 2.0]), _fake_gpu_param([3.0, 4.0]), _fake_gpu_param([5.0, 6.0])]
    hdo = _hdo(params, offload_fraction=0.5)
    # Upstream offloads in order until the threshold is reached: 2 of 6 elements < 3, 4 >= 3.
    assert set(hdo.gpu_params_map_cpu_copy) == {params[0], params[1]}
    gpu_side = hdo.param_to_fp32_param[params[2]]
    assert gpu_side.is_cuda and gpu_side.dtype == torch.float32  # FP32 copy for the GPU optimizer
    assert hdo.gpu_optimizer is not None


def test_throwaway_hdo_is_not_materialized(patched, no_cuda_streams, monkeypatch):
    """The HDO built before the DistributedOptimizer is deferred; the one it builds is not."""
    import megatron.core.optimizer as optimizer_module
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    def fake_dist_init(self, optimizer):
        # What DistributedOptimizer.__init__ does with an HDO: rebuild one over its shards.
        self.first = optimizer
        self.optimizer = HybridDeviceOptimizer(params=optimizer.param_groups, **optimizer.defaults)

    def fake_build(config, params):
        return DistributedOptimizer(_hdo(params))

    monkeypatch.setattr(DistributedOptimizer, "__init__", fake_dist_init)
    monkeypatch.setattr(optimizer_module, "_get_megatron_optimizer_based_on_param_groups", fake_build)
    patched.apply_hybrid_optimizer_cpu_masters_patch()

    class Cfg:
        optimizer_cpu_offload = True
        use_distributed_optimizer = True

    params = [_fake_gpu_param([1.0, 2.0])]
    dist = optimizer_module._get_megatron_optimizer_based_on_param_groups(Cfg(), params)
    assert dist.first._skyrl_init_deferred
    assert not hasattr(dist.first, "cpu_optimizers")  # no host copies were made
    assert not dist.optimizer._skyrl_init_deferred
    assert set(dist.optimizer.gpu_params_map_cpu_copy) == set(params)

    # Without a DistributedOptimizer to replace it, a deferred HDO is completed on the way out.
    monkeypatch.setattr(optimizer_module, "_get_megatron_optimizer_based_on_param_groups", lambda c, p: _hdo(p))
    monkeypatch.setattr(patched, "_APPLIED", False)
    patched.apply_hybrid_optimizer_cpu_masters_patch()
    plain = optimizer_module._get_megatron_optimizer_based_on_param_groups(Cfg(), params)
    assert not plain._skyrl_init_deferred and set(plain.gpu_params_map_cpu_copy) == set(params)


class FakeCudaBf16Param(torch.nn.Parameter):
    """A BF16 model param that reports the CUDA tensor type the DistributedOptimizer checks."""

    @property
    def is_cuda(self):
        return True

    def type(self, *args, **kwargs):
        if args or kwargs:
            return super().type(*args, **kwargs)
        return "torch.cuda.BFloat16Tensor"


@pytest.mark.parametrize("routed", [False, True])
def test_distributed_optimizer_builds_no_fp32_main_params_when_routed(routed):
    """The init OOM site: ``_build_model_and_main_param_groups`` FP32-clones every owned shard."""
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range

    from skyrl.backends.skyrl_train.patches.megatron.patch_hybrid_optimizer_cpu_masters import (
        route_main_params_to_hybrid_optimizer,
    )

    config = OptimizerConfig(
        optimizer="adam",
        lr=1e-5,
        bf16=True,
        use_distributed_optimizer=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
    )
    if routed:
        assert route_main_params_to_hybrid_optimizer(config)

    param = FakeCudaBf16Param(torch.arange(12, dtype=torch.float32).to(torch.bfloat16))
    dtype = (torch.bfloat16, torch.bfloat16)
    gbuf_ranges = [{dtype: [{"param_map": {param: {"param": Range(2, 10)}}}]}]
    group = {"params": [param]}
    opt_group_ranges = [{"params": [param], "orig_group": group}]
    _, _, shard_float16, _, shard_fp32_from_float16 = DistributedOptimizer._build_model_and_main_param_groups(
        gbuf_ranges, {param: (0, dtype, 0)}, opt_group_ranges, config
    )

    shard = shard_float16[0][0]
    assert shard.dtype == torch.bfloat16 and shard.data_ptr() == param.data_ptr() + 2 * 2  # a view
    main = shard_fp32_from_float16[0][0]
    if routed:
        # No main param: the HDO is handed the BF16 param-buffer shard and owns the FP32 master.
        assert main is None and param.main_param is None
        assert group["params"] == [shard]
    else:
        # Upstream default: an FP32 clone on the param's device (the GPU), handed to the HDO.
        assert main.dtype == torch.float32 and main.data_ptr() != shard.data_ptr()
        assert group["params"] == [main]
