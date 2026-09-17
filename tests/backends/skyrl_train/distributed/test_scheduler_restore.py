"""In-place checkpoint restore must replace Core's nonzero scheduler counter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("megatron.core")

from skyrl.backends.skyrl_train.distributed.megatron import (
    megatron_strategy as strategy_module,
)
from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer_param_scheduler,
)


def test_production_checkpoint_load_restores_nonzero_scheduler(monkeypatch, tmp_path):
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(2))], lr=1e-6)
    scheduler = get_megatron_optimizer_param_scheduler(
        optimizer, SimpleNamespace(num_warmup_steps=0, weight_decay=0.01)
    )
    scheduler.step(1)
    saved = scheduler.state_dict()
    scheduler.step(4)
    strategy = object.__new__(strategy_module.MegatronStrategy)
    strategy.is_lora = False
    strategy.finalize_pending_saves = Mock()
    strategy.print = Mock()
    model = Mock(spec=["sharded_state_dict", "load_state_dict"])
    model.sharded_state_dict.return_value = {}
    monkeypatch.setattr(strategy_module, "get_default_load_sharded_strategy", lambda _: None)
    monkeypatch.setattr(strategy_module, "FullyParallelLoadStrategyWrapper", lambda *args: None)
    monkeypatch.setattr(strategy_module.mpu, "get_data_parallel_group", lambda **kwargs: None)
    monkeypatch.setattr(
        strategy_module.dist_checkpointing, "load", lambda **kwargs: {"model": {}, "lr_scheduler": saved}
    )
    for _ in range(2):
        strategy.load_checkpoint(SimpleNamespace(actor_module=[model]), str(tmp_path), scheduler=scheduler)
        assert scheduler.state_dict() == saved
        assert scheduler.num_steps == 1
        assert optimizer.param_groups[0]["lr"] == 1e-6
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
