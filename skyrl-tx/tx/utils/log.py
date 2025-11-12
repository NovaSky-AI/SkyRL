from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    wandb = None  # type: ignore[assignment]


def _setup_root_logger() -> None:
    logger = logging.getLogger("tx")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to root logger
    handler = RichHandler(
        show_time=False,
        show_level=False,
        markup=True,
    )
    formatter = logging.Formatter("%(levelname)s:     %(message)s")
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

    def __init__(self, config: dict[str, Any], **kwargs):
        logger.info(f"model config: {config}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        data = metrics if step is None else {"step": step, **metrics}
        logger.info(
            ", ".join(
                f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}" for key, value in data.items()
            )
        )


class WandbTracker(Tracker):

    def __init__(self, config: dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        if wandb is None:
            raise RuntimeError("wandb not installed")
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY environment variable not set")
        self.run = wandb.init(config=config, **kwargs)  # type: ignore[union-attr]

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        super().log(metrics, step)
        if wandb is not None:
            wandb.log(metrics, step=step)  # type: ignore[union-attr]

    def __del__(self):
        if wandb is not None:
            wandb.finish()  # type: ignore[union-attr]


def get_tracker(tracker: ExperimentTracker | None, config: dict[str, Any], **kwargs) -> Tracker:
    match tracker:
        case None:
            return Tracker(config, **kwargs)
        case ExperimentTracker.wandb:
            return WandbTracker(config, **kwargs)
        case _:
            raise ValueError(f"Unsupported experiment tracker: {tracker}")


__all__ = ["logger"]
