"""
Random seed management utilities for reproducibility in SkyRL training.

This module provides utilities to set random seeds across all relevant libraries
(Python random, NumPy, PyTorch, CUDA) to ensure reproducible training runs.
"""

import os
import random
from typing import Optional

import numpy as np
import torch
from loguru import logger


def set_random_seed(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """
    Set random seed for reproducibility across all libraries.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA operations (if available)

    Args:
        seed: The random seed value to use
        deterministic: If True, sets PyTorch to deterministic mode (slower but reproducible)
        warn_only: If True, only warns instead of raising errors when deterministic operations fail

    Examples:
        >>> set_random_seed(42)
        >>> set_random_seed(42, deterministic=False)  # Faster but less reproducible
    """
    # Set Python random seed
    random.seed(seed)
    logger.debug(f"Set Python random seed to {seed}")

    # Set NumPy random seed
    np.random.seed(seed)
    logger.debug(f"Set NumPy random seed to {seed}")

    # Set PyTorch random seed
    torch.manual_seed(seed)
    logger.debug(f"Set PyTorch CPU random seed to {seed}")

    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.debug(f"Set PyTorch CUDA random seed to {seed}")

    # Set deterministic mode for PyTorch
    if deterministic:
        # Set deterministic algorithms (may be slower)
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            logger.debug("Enabled PyTorch deterministic algorithms")
        except Exception as e:
            if warn_only:
                logger.warning(f"Could not enable deterministic algorithms: {e}")
            else:
                raise

        # Set CUBLAS workspace to deterministic
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.debug("Set CUBLAS workspace to deterministic mode")

    # Disable cuDNN benchmarking for reproducibility
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.debug("Set cuDNN to deterministic mode")


def get_random_seed() -> Optional[int]:
    """
    Get the current random seed state.

    Returns:
        The current seed if set, None otherwise. Note that this only returns
        the Python random seed state, as other libraries don't expose their
        seed state directly.
    """
    return random.getstate()[1][0] if random.getstate()[0] == "MT19937" else None


def reset_random_seed(seed: int) -> None:
    """
    Reset all random seeds to a new value.

    This is useful when you want to change the seed mid-training or
    ensure a clean seed state.

    Args:
        seed: The new random seed value to use
    """
    set_random_seed(seed, deterministic=True, warn_only=True)
    logger.info(f"Reset random seed to {seed}")


def seed_worker(worker_id: int, base_seed: int) -> None:
    """
    Seed a dataloader worker process.

    This function is designed to be used as a `worker_init_fn` for PyTorch
    DataLoaders to ensure each worker has a different but deterministic seed.

    Args:
        worker_id: The ID of the worker process
        base_seed: The base seed value

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     worker_init_fn=lambda worker_id: seed_worker(worker_id, 42)
        ... )
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
    logger.debug(f"Seeded worker {worker_id} with seed {worker_seed}")
