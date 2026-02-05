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


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"

RICH_HANDLER_KWARGS = {
    "show_time": False,
    "show_level": False,
    "show_path": False,
    "markup": False,
    "rich_tracebacks": True,
}


def _create_rich_handler() -> RichHandler:
    """Create a RichHandler with consistent configuration."""
    handler = RichHandler(**RICH_HANDLER_KWARGS)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _setup_root_logger() -> None:
    logger = logging.getLogger("tx")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to root logger
    logger.addHandler(_create_rich_handler())


def get_uvicorn_log_config() -> dict:
    """Get uvicorn logging config that uses the same RichHandler."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": LOG_FORMAT,
            },
        },
        "handlers": {
            "default": {
                "()": RichHandler,
                **RICH_HANDLER_KWARGS,
                "formatter": "default",
            },
        },
        "loggers": {
            # Main uvicorn logger (general server messages)
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            # Uvicorn error logger (startup, shutdown, exceptions)
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            # HTTP access logs (request/response logging)
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger("tx")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
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


__all__ = ["logger", "get_uvicorn_log_config"]
