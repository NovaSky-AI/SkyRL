"""
Logging configuration for SkyRL.

This module provides logging setup that separates training progress from infrastructure logs:
- All logs go to a file in SKYRL_LOG_DIR
- Only training progress logs go to stdout by default
- Set SKYRL_LOG_LEVEL=DEBUG to show all logs on stdout
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from skyrl_train.env_vars import SKYRL_LOG_DIR, SKYRL_LOG_LEVEL

# Logger names that are considered "training progress" and should go to stdout
# All other loggers are considered "infrastructure" and only go to the log file
TRAINING_PROGRESS_LOGGERS = frozenset(
    {
        "skyrl_train.trainer",
        "skyrl_train.fully_async_trainer",
        "skyrl_train.utils.tracking",
        "skyrl_train.evaluate",
    }
)

# Logger name prefixes to suppress from stdout (infrastructure loggers)
INFRA_LOGGER_PREFIXES = (
    "vllm",
    "ray",
    "transformers",
    "torch",
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "skyrl_train.workers",
    "skyrl_train.inference_engines",
    "skyrl_train.distributed",
    "skyrl_train.model_wrapper",
    "skyrl_train.utils.utils",
    "skyrl_train.utils.ppo_utils",
    "skyrl_train.dataset",
    "skyrl_train.generators",
)


class TrainingProgressFilter(logging.Filter):
    """Filter that only allows training progress logs through."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Allow if it's a training progress logger
        if record.name in TRAINING_PROGRESS_LOGGERS:
            return True

        # Allow if it starts with a training progress logger prefix
        for prefix in TRAINING_PROGRESS_LOGGERS:
            if record.name.startswith(prefix + "."):
                return True

        # Block if it's an infrastructure logger
        for prefix in INFRA_LOGGER_PREFIXES:
            if record.name.startswith(prefix):
                return False

        # Allow other loggers (e.g., root logger, unknown loggers)
        return True


def setup_logging(run_name: Optional[str] = None) -> Path:
    """
    Set up SkyRL logging with file and console handlers.

    All logs go to the log file. Only training progress logs go to stdout
    unless SKYRL_LOG_LEVEL=DEBUG, in which case all logs go to stdout.

    Args:
        run_name: Name of the training run for organizing logs. If None,
                  logs go directly to SKYRL_LOG_DIR.

    Returns:
        Path to the log directory.
    """
    # Create log directory
    log_dir = Path(SKYRL_LOG_DIR)
    if run_name:
        log_dir = log_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "logs.log"

    # Determine if we should show all logs on stdout
    verbose = SKYRL_LOG_LEVEL == "DEBUG"

    # Configure loguru for file output (all logs)
    logger.remove()

    # File handler - all logs with colors
    logger.add(
        str(log_file),
        colorize=True,
        level="DEBUG",  # Capture all levels in file
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # Console handler - filtered or all based on log level
    def console_filter(record):
        """Filter for console output - only training progress unless DEBUG."""
        if verbose:
            return True
        name = record["name"]
        # Check if it's a training progress logger
        if name in TRAINING_PROGRESS_LOGGERS:
            return True
        for prefix in TRAINING_PROGRESS_LOGGERS:
            if name.startswith(prefix + "."):
                return True
        # Check if it's an infra logger to block
        for prefix in INFRA_LOGGER_PREFIXES:
            if name.startswith(prefix):
                return False
        return True

    logger.add(
        sys.stderr,
        colorize=True,
        level=SKYRL_LOG_LEVEL,
        filter=console_filter,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # Configure stdlib logging to route through loguru
    _setup_stdlib_logging(log_file, verbose)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return log_dir


def _setup_stdlib_logging(log_file: Path, verbose: bool) -> None:
    """
    Configure stdlib logging to write to file and optionally stdout.

    This captures logs from vLLM, transformers, and other third-party libraries.
    """

    class InterceptHandler(logging.Handler):
        """Handler that routes stdlib logging to loguru."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where the logged message originated
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Remove existing handlers and add our interceptor
    logging.root.handlers = [InterceptHandler()]
    level = getattr(logging, SKYRL_LOG_LEVEL, logging.INFO)
    logging.root.setLevel(level)

    # Suppress noisy third-party loggers
    for name in ["httpx", "httpcore", "urllib3", "asyncio"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_worker_logging() -> None:
    """
    Configure logging for Ray workers.

    This is called within Ray workers to set up proper logging. Workers
    write to stderr which is captured by Ray.
    """
    # Use the same setup but without run_name (workers don't create log dirs)
    logger.remove()

    level_name = SKYRL_LOG_LEVEL

    # Console only for workers (Ray handles log forwarding)
    logger.level("INFO", color="<bold><green>")
    logger.add(
        sys.stderr,
        colorize=True,
        level=level_name,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # Route stdlib logging through loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    logging.root.handlers = [InterceptHandler()]
    level = getattr(logging, level_name, logging.INFO)
    logging.root.setLevel(level)
