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
from .app import get_program_id, register_routes

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "BackendState",
    "ProgramState",
    "ProgramStatus",
    "MultiBackendRouter",
    "get_program_id",
    "register_routes",
]

__version__ = "0.2.0"
