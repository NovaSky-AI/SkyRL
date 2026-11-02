"""CPU tests for set_optimizer_hparams: writing the request's AdamParams into a live optimizer.

No Ray cluster and no GPU: this pins the param-group walk shared by both worker families
(the base implementation) and the FSDP per-step clip hook. Megatron's ChainedOptimizer walk
and its optimizer-config clip are exercised end to end by the GPU warm-runtime and Megatron
LoRA tests, which run real optim_steps.

Run:
  uv run --isolated --extra fsdp --extra tinker --extra dev \\
    pytest tests/tinker/skyrl_train/test_optim_step_hparams.py
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.workers.worker import Worker  # noqa: E402
from skyrl.tinker import types  # noqa: E402

ADAM = types.AdamParams(learning_rate=1e-4, beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.05, grad_clip_norm=0.5)


def _two_group_optimizer() -> torch.optim.AdamW:
    """AdamW with one decaying and one no-decay group, the way Megatron builds them."""
    return torch.optim.AdamW(
        [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "weight_decay": 0.01},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "weight_decay": 0.0},
        ],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def test_set_optimizer_hparams_walks_every_param_group():
    """The base set_optimizer_hparams updates all groups and the strategy's per-step clip."""
    worker = Worker.__new__(Worker)
    worker.optimizer = _two_group_optimizer()
    worker.strategy = SimpleNamespace(max_norm=1.0)  # an FSDP-style strategy

    worker.set_optimizer_hparams(ADAM)

    decaying, zero = worker.optimizer.param_groups
    assert (decaying["lr"], decaying["betas"], decaying["eps"], decaying["weight_decay"]) == (
        1e-4,
        (0.8, 0.95),
        1e-12,
        0.05,
    )
    # A no-decay group keeps its 0; everything else is the request's value.
    assert (zero["lr"], zero["betas"], zero["eps"], zero["weight_decay"]) == (1e-4, (0.8, 0.95), 1e-12, 0.0)
    # The strategy the FSDP worker clips with reads max_norm per step (fsdp_strategy.py).
    assert worker.strategy.max_norm == 0.5


def test_set_optimizer_hparams_can_turn_decay_back_on():
    """A request's weight_decay=0 zeroes decay for that step, not forever."""
    worker = Worker.__new__(Worker)
    worker.optimizer = _two_group_optimizer()
    worker.strategy = SimpleNamespace(max_norm=1.0)

    # One step with the SDK-default weight_decay=0, then a request that decays again.
    worker.set_optimizer_hparams(ADAM.model_copy(update={"weight_decay": 0.0}))
    worker.set_optimizer_hparams(ADAM)

    decaying, zero = worker.optimizer.param_groups
    assert decaying["weight_decay"] == 0.05  # the decaying group decays again
    assert zero["weight_decay"] == 0.0  # a structurally no-decay group stays 0 forever


def test_set_optimizer_hparams_without_an_optimizer_is_a_no_op():
    """policy.inference_only_init builds no optimizer; applying hyperparameters must not raise."""
    worker = Worker.__new__(Worker)
    worker.optimizer = None
    worker.set_optimizer_hparams(ADAM)
