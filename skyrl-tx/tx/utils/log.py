from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

try:
    import wandb
except ImportError:
    wandb = None


def _setup_root_logger() -> None:
    logger = logging.getLogger("tx")
    logger.setLevel(logging.DEBUG)
    handler = RichHandler(
        show_time=False,
        show_level=False,
        markup=True,
    )
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger("tx")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")

_setup_root_logger()
logger = logging.getLogger("tx")


class ExperimentTracker(str, Enum):
    wandb = "wandb"


class Tracker:

    def __init__(self, config: dict[str, str]):
        pass

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        data = {"step": step, **metrics} if step else metrics
        logger.info(", ".join(f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}" for key, value in data.items()))


class WandbTracker(Tracker):

    def __init__(self, config: dict[str, str]):
        if wandb is None:
            raise RuntimeError("wandb not installed")
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY environment variable not set")
        self.run = wandb.init(**config)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        super().log(metrics, step)
        if wandb is not None:
            wandb.log(metrics, step=step)

    def __del__(self):
        if wandb is not None:
            wandb.finish()


def get_tracker(tracker: ExperimentTracker | None, config: dict[str, str]) -> Tracker:
    match tracker:
        case None:
            return Tracker(config)
        case ExperimentTracker.wandb:
            return WandbTracker(config)
        case _:
            raise ValueError(f"Unsupported experiment tracker: {tracker}")


__all__ = ["logger"]
