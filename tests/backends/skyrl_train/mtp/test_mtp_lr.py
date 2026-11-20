"""CPU tests for the MTP heads' own learning rate (trainer.mtp.lr).

The heads get their own Megatron optimizer parameter group, matched by name, and Megatron's scheduler drives it
between ``mtp_lr`` and a proportionally scaled ``min_lr`` with the policy's warmup and decay.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.optimizer import OptimizerConfig, _get_param_groups  # noqa: E402
from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,  # noqa: E402
)

from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (  # noqa: E402
    get_megatron_optimizer,
    mtp_heads_lr,
    mtp_lr_config_overrides,
)
from skyrl.train.config.config import (  # noqa: E402
    MegatronOptimizerOffloadToDiskConfig,
)

pytestmark = pytest.mark.megatron


@pytest.fixture(scope="module", autouse=True)
def _single_process_group():
    if not torch.distributed.is_initialized():
        store = tempfile.NamedTemporaryFile(delete=False)
        torch.distributed.init_process_group("gloo", init_method=f"file://{store.name}", rank=0, world_size=1)
        yield
        torch.distributed.destroy_process_group()
        if os.path.exists(store.name):  # gloo's FileStore may have removed it already
            os.unlink(store.name)
    else:
        yield


class _Toy(torch.nn.Module):
    """Megatron-style names: decoder layers, the MTP block, and a 1-D norm in each."""

    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.proj = torch.nn.Linear(4, 4, bias=False)
        self.decoder.norm = torch.nn.LayerNorm(4, bias=False)
        self.mtp = torch.nn.Module()
        self.mtp.eh_proj = torch.nn.Linear(8, 4, bias=False)
        self.mtp.enorm = torch.nn.LayerNorm(4, bias=False)


def _groups(lr=1e-6, min_lr=5e-7, mtp_lr=3e-5):
    config = OptimizerConfig(lr=lr, min_lr=min_lr, weight_decay=0.01)
    model = _Toy()
    groups = _get_param_groups([model], config, mtp_lr_config_overrides(config, mtp_lr))
    by_param = {}
    for g in groups:
        for p in g["params"]:
            by_param[id(p)] = g
    return model, groups, by_param


def test_heads_get_their_own_lr_and_scaled_min_lr():
    model, _, by_param = _groups()
    for p in (model.mtp.eh_proj.weight, model.mtp.enorm.weight):
        assert by_param[id(p)]["max_lr"] == pytest.approx(3e-5)
        assert by_param[id(p)]["min_lr"] == pytest.approx(1.5e-5)  # the policy's min_lr / lr = 0.5
    for p in (model.decoder.proj.weight, model.decoder.norm.weight):
        assert by_param[id(p)]["max_lr"] == pytest.approx(1e-6)
        assert by_param[id(p)]["min_lr"] == pytest.approx(5e-7)
    # The standard overrides merge: 1-D params keep Megatron's zero weight decay, in the heads' group too.
    assert by_param[id(model.mtp.enorm.weight)]["wd_mult"] == 0.0
    assert by_param[id(model.mtp.eh_proj.weight)]["wd_mult"] == 1.0
    assert by_param[id(model.decoder.norm.weight)]["wd_mult"] == 0.0


def _scheduler(optimizer):
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=5e-7,
        lr_warmup_steps=2,
        lr_decay_steps=100,
        lr_decay_style="constant",
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=100,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )


def test_scheduler_drives_the_heads_group_at_its_own_lr():
    _, groups, _ = _groups()
    sched = _scheduler(SimpleNamespace(param_groups=groups))
    sched.step(1)  # halfway through warmup: both groups at half their peak
    lrs = sorted({round(g["lr"], 12) for g in groups})
    assert lrs == [pytest.approx(5e-7), pytest.approx(1.5e-5)]
    sched.step(5)  # past warmup: each at its own peak
    lrs = sorted({round(g["lr"], 12) for g in groups})
    assert lrs == [pytest.approx(1e-6), pytest.approx(3e-5)]


def test_mtp_lr_metric_reads_the_heads_group():
    model, groups, by_param = _groups()
    optimizer = SimpleNamespace(param_groups=groups)
    _scheduler(optimizer).step(1)  # halfway through warmup
    assert mtp_heads_lr(optimizer, 3e-5) == by_param[id(model.mtp.eh_proj.weight)]["lr"] == pytest.approx(1.5e-5)
    # Without an own lr the heads sit in the policy's group.
    assert mtp_heads_lr(optimizer, None) == by_param[id(model.decoder.proj.weight)]["lr"] == pytest.approx(5e-7)
    # A miss fails rather than silently dropping the metric.
    with pytest.raises(AssertionError, match="no optimizer param group"):
        mtp_heads_lr(optimizer, 7e-5)


def test_disk_offloaded_optimizer_refuses_an_own_mtp_lr():
    with pytest.raises(ValueError, match="does not support trainer.mtp.lr"):
        get_megatron_optimizer(
            model=[],
            config=OptimizerConfig(lr=1e-6),
            offload_to_disk=MegatronOptimizerOffloadToDiskConfig(),
            mtp_lr=3e-5,
        )
