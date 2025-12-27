"""
Simple logging utility for SkyRL Gym.
"""
import logging
import sys
import os

def get_logger(name: str):
    """Get a simple logger for the given module name."""
    # Configure the library's root logger once.
    # All loggers in the library will propagate to this logger.
    lib_logger = logging.getLogger("skyrl_gym")
    if not lib_logger.handlers:
        level = os.environ.get("SKYRL_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        lib_logger.addHandler(handler)
        lib_logger.setLevel(getattr(logging, level, logging.INFO))
        # Prevent double logging if the root logger is also configured.
        lib_logger.propagate = False
    
    return logging.getLogger(f"skyrl_gym.{name}")