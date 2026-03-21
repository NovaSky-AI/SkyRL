"""
ThunderAgent - VLLM proxy with program state tracking.

Module structure:
- backend: Backend state management
- program: Program state management
- scheduler: Request routing and proxying
"""

from .config import Config, get_config, set_config
from .backend import BackendState
from .program import ProgramState, ProgramStatus
from .scheduler import MultiBackendRouter

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "BackendState",
    "ProgramState",
    "ProgramStatus",
    "MultiBackendRouter",
]

__version__ = "0.2.0"
