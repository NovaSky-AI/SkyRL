import pytest
import torch

from skyrl.backends.skyrl_train.utils.lr_scheduler import ScaleAwareLRScheduler


def test_scale_aware_scheduler_scales_current_and_future_lrs():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    wrapped = ScaleAwareLRScheduler(scheduler, optimizer)

    wrapped.set_scale(0.25)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)

    optimizer.step()
    wrapped.step()

    assert wrapped.get_base_lr()[0] == pytest.approx(0.5)
    assert wrapped.get_last_lr()[0] == pytest.approx(0.125)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.125)


def test_scale_aware_scheduler_state_dict_round_trip():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    wrapped = ScaleAwareLRScheduler(scheduler, optimizer)
    wrapped.set_scale(0.5)

    optimizer.step()
    wrapped.step()
    state_dict = wrapped.state_dict()

    param2 = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer2 = torch.optim.AdamW([param2], lr=1.0)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.5)
    restored = ScaleAwareLRScheduler(scheduler2, optimizer2)
    restored.load_state_dict(state_dict)

    assert restored.get_scale() == pytest.approx(0.5)
    assert restored.get_base_lr()[0] == pytest.approx(wrapped.get_base_lr()[0])
    assert restored.get_last_lr()[0] == pytest.approx(wrapped.get_last_lr()[0])
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(wrapped.get_last_lr()[0])


def test_scale_aware_scheduler_supports_step_with_args():
    class DummyMegatronScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.last_increment = None

        def step(self, increment):
            self.last_increment = increment
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.5

        def state_dict(self):
            return {"last_increment": self.last_increment}

        def load_state_dict(self, state_dict):
            self.last_increment = state_dict.get("last_increment")

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = DummyMegatronScheduler(optimizer)
    wrapped = ScaleAwareLRScheduler(scheduler, optimizer)
    wrapped.set_scale(0.5)

    wrapped.step(1)

    assert scheduler.last_increment == 1
    assert wrapped.get_base_lr()[0] == pytest.approx(0.5)
    assert wrapped.get_last_lr()[0] == pytest.approx(0.25)
