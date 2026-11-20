"""CPU test: expert_nvfp4's stochastic weight rounding takes its version from the policy's LR scheduler, so a worker
resumed from a checkpoint draws as one that ran continuously.

Run with:
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron -- pytest tests/backends/skyrl_train/workers/megatron/quantization/test_expert_nvfp4_resume.py
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import stochastic as sr
from skyrl.backends.skyrl_train.workers.megatron.quantization import expert_nvfp4
from skyrl.train.config.config import OptimizerConfig

try:
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
        get_megatron_optimizer_param_scheduler,
    )
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronPolicyWorkerBase,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)

SEED = 7


class _Strategy:
    """MegatronStrategy's scheduler contract: one scheduler step per optimizer step; a checkpoint load restores it."""

    def __init__(self, saved_scheduler: dict | None = None):
        self.saved_scheduler = saved_scheduler

    def optimizer_step(self, optimizer, model, scheduler, name):
        scheduler.step(1)
        return None

    def load_checkpoint(self, model, optimizer, scheduler, ckpt_dir, load_optimizer_states, load_lr_scheduler_states):
        scheduler.load_state_dict(self.saved_scheduler)
        return ckpt_dir, {}


def _start_worker(monkeypatch, strategy: _Strategy) -> MegatronPolicyWorkerBase:
    """An uninitialized policy worker with a real scheduler, in a fresh process (install resets the rounding)."""
    monkeypatch.setattr(expert_nvfp4, "_WEIGHT_ROUNDING", expert_nvfp4._WeightRounding())
    expert_nvfp4.set_weight_rounding(True, SEED)
    worker = MegatronPolicyWorkerBase.__new__(MegatronPolicyWorkerBase)
    worker.cfg = SimpleNamespace(
        log_mfu=False, policy=SimpleNamespace(megatron_config=SimpleNamespace(expert_nvfp4=True))
    )
    worker.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    worker.scheduler = get_megatron_optimizer_param_scheduler(worker.optimizer, OptimizerConfig())
    worker.model = SimpleNamespace(run_pending_grad_sync=lambda: None)
    worker.actor_module = []
    worker.strategy = strategy
    return worker


@pytest.mark.megatron
def test_resumed_worker_draws_as_a_continuous_one(monkeypatch):
    w = torch.nn.Parameter(torch.zeros(1))
    monkeypatch.setattr(expert_nvfp4, "_WEIGHT_STREAMS", {id(w): 5})
    resume_at, total = 3, 5

    continuous = _start_worker(monkeypatch, _Strategy())
    draws, saved = [expert_nvfp4.weight_draw([w])], None
    for step in range(1, total + 1):
        continuous.optim_step()
        draws.append(expert_nvfp4.weight_draw([w]))
        if step == resume_at:
            saved = copy.deepcopy(continuous.scheduler.state_dict())
    assert [d.seed for d in draws] == [sr.weight_seed(SEED, v) for v in range(total + 1)]

    resumed = _start_worker(monkeypatch, _Strategy(saved))
    assert expert_nvfp4.weight_draw([w]) == draws[0]
    resumed.load_checkpoint("ckpt")
    assert expert_nvfp4.weight_draw([w]) == draws[resume_at]
    for step in range(resume_at + 1, total + 1):
        resumed.optim_step()
        assert expert_nvfp4.weight_draw([w]) == draws[step]
