"""Helpers for the example-local FlashRL runtime contract."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Callable

FLASHRL_CONFIG_ENV_VAR = "FLASHRL_CONFIG"
FLASHRL_PATCH_FN_ENV_VAR = "SKYRL_FLASHRL_PATCH_FN"

DEFAULT_FLASHRL_PATCH_TARGET = "vllm.model_executor.layers.patch:apply_patch"
DEFAULT_FLASHRL_TRANSFORMERS_VERSION = "4.53.3"
DEFAULT_FLASHRL_VLLM_WHEEL_URL = (
    "https://github.com/NovaSky-AI/SkyRL/releases/download/"
    "skyrl_train-v0.1.0/vllm-0.1.dev7509+gcc487699a.d20250821-cp312-cp312-linux_x86_64.whl"
)

FLASHRL_README_PATH = "examples/train/flash_rl/README.md"


def get_flashrl_config() -> str:
    """Return the required FlashRL config path/name from the environment."""
    flashrl_config = os.environ.get(FLASHRL_CONFIG_ENV_VAR, "").strip()
    if not flashrl_config:
        raise RuntimeError(
            f"FlashRL requires `{FLASHRL_CONFIG_ENV_VAR}` to be set. "
            f"See {FLASHRL_README_PATH} for the expected setup."
        )
    return flashrl_config


def get_flashrl_patch_target() -> str:
    """Return the import target for the FlashRL patch hook."""
    return os.environ.get(FLASHRL_PATCH_FN_ENV_VAR, DEFAULT_FLASHRL_PATCH_TARGET).strip()


def _parse_patch_target(import_target: str) -> tuple[str, str]:
    module_name, separator, attr_name = import_target.partition(":")
    if not separator or not module_name or not attr_name:
        raise RuntimeError(
            f"`{FLASHRL_PATCH_FN_ENV_VAR}` must have the format `<module>:<callable>`, "
            f"got {import_target!r}."
        )
    return module_name, attr_name


@lru_cache(maxsize=1)
def load_flashrl_patch_fn() -> Callable[[], None]:
    """Import and return the FlashRL patch hook from the custom vLLM build."""
    import_target = get_flashrl_patch_target()
    module_name, attr_name = _parse_patch_target(import_target)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "FlashRL requires a custom vLLM wheel that exposes a patch hook at "
            f"`{import_target}`. Install a FlashRL-compatible wheel and, if the hook moved, "
            f"set `{FLASHRL_PATCH_FN_ENV_VAR}` accordingly. See {FLASHRL_README_PATH}."
        ) from exc

    try:
        patch_fn = getattr(module, attr_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Imported `{module_name}` but could not find `{attr_name}`. "
            f"Set `{FLASHRL_PATCH_FN_ENV_VAR}` to the correct `<module>:<callable>` for "
            "your FlashRL-compatible vLLM wheel."
        ) from exc

    if not callable(patch_fn):
        raise RuntimeError(
            f"`{import_target}` resolved successfully, but `{attr_name}` is not callable."
        )

    return patch_fn


def validate_flashrl_environment(validate_patch_fn: bool = True) -> str:
    """Validate the minimal environment required for the FlashRL example path."""
    flashrl_config = get_flashrl_config()
    if validate_patch_fn:
        load_flashrl_patch_fn()
    return flashrl_config
