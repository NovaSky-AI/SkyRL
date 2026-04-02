from typing import Any, Dict, List


def _iter_param_groups(optimizer) -> List[Dict[str, Any]]:
    if hasattr(optimizer, "chained_optimizers"):
        groups = []
        for chained_optimizer in optimizer.chained_optimizers:
            groups.extend(chained_optimizer.param_groups)
        return groups
    return list(optimizer.param_groups)


class ScaleAwareLRScheduler:
    """A thin wrapper that layers a persistent multiplicative LR scale on top of a scheduler."""

    def __init__(self, scheduler, optimizer, scale: float = 1.0):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self._scale = float(scale)
        self._base_lrs = self._read_optimizer_lrs()
        self._apply_scale()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def _read_optimizer_lrs(self) -> List[float]:
        return [float(param_group["lr"]) for param_group in _iter_param_groups(self.optimizer)]

    def _set_optimizer_lrs(self, lrs: List[float]) -> None:
        param_groups = _iter_param_groups(self.optimizer)
        if len(param_groups) != len(lrs):
            raise ValueError(
                f"Scheduler LR count ({len(lrs)}) does not match optimizer param groups ({len(param_groups)})."
            )
        for param_group, lr in zip(param_groups, lrs):
            param_group["lr"] = lr

    def _apply_scale(self) -> None:
        self._set_optimizer_lrs([base_lr * self._scale for base_lr in self._base_lrs])

    def step(self, *args, **kwargs):
        self._set_optimizer_lrs(self._base_lrs)
        try:
            result = self.scheduler.step(*args, **kwargs)
            self._base_lrs = self._read_optimizer_lrs()
        finally:
            self._apply_scale()
        return result

    def get_last_lr(self) -> List[float]:
        return [base_lr * self._scale for base_lr in self._base_lrs]

    def get_base_lr(self) -> List[float]:
        return list(self._base_lrs)

    def get_scale(self) -> float:
        return self._scale

    def set_scale(self, scale: float) -> None:
        self._scale = float(scale)
        self._apply_scale()

    def set_absolute_lr(self, learning_rate: float) -> None:
        self._base_lrs = [float(learning_rate) / self._scale for _ in self._base_lrs]
        if hasattr(self.scheduler, "base_lrs") and len(self.scheduler.base_lrs) == len(self._base_lrs):
            self.scheduler.base_lrs = list(self._base_lrs)
        self._apply_scale()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wrapped_scheduler_state": self.scheduler.state_dict(),
            "scale": self._scale,
            "base_lrs": list(self._base_lrs),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "wrapped_scheduler_state" not in state_dict:
            self.scheduler.load_state_dict(state_dict)
            self._scale = 1.0
            self._base_lrs = self._read_optimizer_lrs()
            self._apply_scale()
            return

        self.scheduler.load_state_dict(state_dict["wrapped_scheduler_state"])
        self._scale = float(state_dict.get("scale", 1.0))
        self._base_lrs = list(state_dict.get("base_lrs", self._read_optimizer_lrs()))
        self._apply_scale()
