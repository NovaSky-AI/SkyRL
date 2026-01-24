"""
Early stopping callback for training.
Monitors a metric and stops training if it doesn't improve for a specified number of steps.
"""

from typing import Optional, Callable
from loguru import logger
import numpy as np


class EarlyStopping:
    """
    Early stopping callback that monitors a metric and stops training
    if it doesn't improve for a specified patience period.
    
    Args:
        monitor: Metric name to monitor (e.g., "eval/reward_mean")
        patience: Number of steps to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: "min" to minimize metric, "max" to maximize metric
        restore_best_weights: Whether to restore best weights when stopping
    """
    
    def __init__(
        self,
        monitor: str = "eval/reward_mean",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best_weights: bool = False,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value: Optional[float] = None
        self.best_step: int = 0
        self.wait_count: int = 0
        self.stopped_epoch: int = 0
        self.best_weights: Optional[dict] = None
        
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.is_better = self._get_comparison_fn()
    
    def _get_comparison_fn(self) -> Callable[[float, float], bool]:
        """Get comparison function based on mode."""
        if self.mode == "min":
            return lambda current, best: current < (best - self.min_delta)
        else:  # mode == "max"
            return lambda current, best: current > (best + self.min_delta)
    
    def on_step_end(self, metrics: dict, step: int, model=None) -> bool:
        """
        Called at the end of each training step.
        
        Args:
            metrics: Dictionary of metrics from the current step
            step: Current training step
            model: Optional model to save/restore weights from
        
        Returns:
            True if training should continue, False if training should stop
        """
        if self.monitor not in metrics:
            logger.warning(f"Early stopping monitor '{self.monitor}' not found in metrics. Available: {list(metrics.keys())}")
            return True
        
        current_value = metrics[self.monitor]
        
        # Initialize best_value on first call
        if self.best_value is None:
            self.best_value = current_value
            self.best_step = step
            if self.restore_best_weights and model is not None:
                self._save_weights(model)
            return True
        
        # Check if current value is better
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_step = step
            self.wait_count = 0
            if self.restore_best_weights and model is not None:
                self._save_weights(model)
            logger.info(
                f"Early stopping: {self.monitor} improved to {current_value:.4f} "
                f"(best: {self.best_value:.4f} at step {self.best_step})"
            )
        else:
            self.wait_count += 1
            logger.debug(
                f"Early stopping: {self.monitor} did not improve. "
                f"Wait count: {self.wait_count}/{self.patience}"
            )
        
        # Check if we should stop
        if self.wait_count >= self.patience:
            logger.info(
                f"Early stopping triggered: {self.monitor} did not improve "
                f"for {self.patience} steps. Best value: {self.best_value:.4f} at step {self.best_step}"
            )
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                self._restore_weights(model)
                logger.info("Restored best weights from step {}".format(self.best_step))
            return False
        
        return True
    
    def _save_weights(self, model):
        """Save model weights for potential restoration."""
        try:
            # Try to get state dict
            if hasattr(model, "state_dict"):
                self.best_weights = model.state_dict().copy()
            elif hasattr(model, "module") and hasattr(model.module, "state_dict"):
                self.best_weights = model.module.state_dict().copy()
            else:
                logger.warning("Could not save model weights for early stopping")
        except Exception as e:
            logger.warning(f"Error saving model weights: {e}")
    
    def _restore_weights(self, model):
        """Restore saved model weights."""
        try:
            if self.best_weights is None:
                return
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(self.best_weights)
            elif hasattr(model, "module") and hasattr(model.module, "load_state_dict"):
                model.module.load_state_dict(self.best_weights)
            else:
                logger.warning("Could not restore model weights")
        except Exception as e:
            logger.warning(f"Error restoring model weights: {e}")
    
    def get_best_value(self) -> Optional[float]:
        """Get the best value seen so far."""
        return self.best_value
    
    def get_best_step(self) -> int:
        """Get the step at which the best value was seen."""
        return self.best_step
