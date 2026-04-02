from dataclasses import dataclass
from typing import Dict, Optional

from skyrl.train.config import ResponseLengthAdaptiveLRConfig


@dataclass
class ResponseLengthAdaptiveLRUpdate:
    scale: float
    triggered: bool
    metrics: Dict[str, float]


class ResponseLengthAdaptiveLRController:
    """Simple response-length-triggered LR decay controller.

    The controller tracks an EMA of average response length and decays a multiplicative
    LR scale when the latest response length exceeds the prior EMA by a configured ratio.
    """

    def __init__(self, cfg: ResponseLengthAdaptiveLRConfig, warmup_steps: int):
        self.cfg = cfg
        self.warmup_steps = warmup_steps
        self.current_scale = 1.0
        self.ema_response_length: Optional[float] = None
        self.last_observed_response_length: Optional[float] = None
        self.last_trigger_step: Optional[int] = None
        self.decay_count = 0

    def _monitoring_ready(self, step: int) -> bool:
        return step >= max(self.warmup_steps, self.cfg.min_monitor_steps)

    def _cooldown_elapsed(self, step: int) -> bool:
        if self.last_trigger_step is None:
            return True
        return (step - self.last_trigger_step) >= self.cfg.cooldown_steps

    def update(self, step: int, avg_response_length: float) -> ResponseLengthAdaptiveLRUpdate:
        current_length = float(avg_response_length)
        prev_ema = self.ema_response_length
        triggered = False
        response_length_ratio = 1.0

        if prev_ema is not None and prev_ema > 0:
            response_length_ratio = current_length / prev_ema

        if (
            prev_ema is not None
            and self._monitoring_ready(step)
            and self.decay_count < self.cfg.max_decays
            and self._cooldown_elapsed(step)
            and response_length_ratio >= self.cfg.trigger_ratio
        ):
            next_scale = max(self.current_scale * self.cfg.decay_factor, self.cfg.min_scale)
            if next_scale < self.current_scale:
                self.current_scale = next_scale
                self.decay_count += 1
                self.last_trigger_step = step
                triggered = True

        if prev_ema is None:
            self.ema_response_length = current_length
        else:
            alpha = self.cfg.ema_alpha
            self.ema_response_length = alpha * current_length + (1 - alpha) * prev_ema

        self.last_observed_response_length = current_length

        metrics = {
            "enabled": 1.0,
            "avg_response_length": current_length,
            "ema_response_length": float(self.ema_response_length),
            "response_length_ratio": response_length_ratio,
            "trigger_threshold": float(self.cfg.trigger_ratio),
            "lr_scale": self.current_scale,
            "decay_count": float(self.decay_count),
            "triggered": float(triggered),
        }
        return ResponseLengthAdaptiveLRUpdate(scale=self.current_scale, triggered=triggered, metrics=metrics)

    def state_dict(self) -> Dict[str, float]:
        return {
            "current_scale": self.current_scale,
            "ema_response_length": self.ema_response_length,
            "last_observed_response_length": self.last_observed_response_length,
            "last_trigger_step": self.last_trigger_step,
            "decay_count": self.decay_count,
        }

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        self.current_scale = float(state_dict.get("current_scale", 1.0))
        self.ema_response_length = state_dict.get("ema_response_length")
        self.last_observed_response_length = state_dict.get("last_observed_response_length")
        self.last_trigger_step = state_dict.get("last_trigger_step")
        self.decay_count = int(state_dict.get("decay_count", 0))
