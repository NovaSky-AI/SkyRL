"""
CLI override support for Pydantic configurations.

This module provides utilities to override Pydantic config fields from command-line
arguments, similar to Hydra's CLI override syntax.

Example:
    python script.py --trainer.epochs=30 --trainer.policy.model.path="Qwen/Qwen2.5-7B"
"""

import argparse
import sys
from typing import Any, List, Optional
from .configs import SkyRLConfig, set_nested_attr


def parse_override(override_str: str) -> tuple[str, Any]:
    """
    Parse a CLI override string into (key, value) pair.

    Args:
        override_str: Override string like "trainer.epochs=30"

    Returns:
        Tuple of (dotted_path, parsed_value)

    Examples:
        >>> parse_override("trainer.epochs=30")
        ("trainer.epochs", 30)
        >>> parse_override("trainer.policy.model.path=Qwen/Qwen2.5-1.5B")
        ("trainer.policy.model.path", "Qwen/Qwen2.5-1.5B")
    """
    if "=" not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'key=value'")

    key, value_str = override_str.split("=", 1)
    key = key.strip()
    value_str = value_str.strip()

    # Try to parse as Python literal (handles int, float, bool, lists, etc.)
    try:
        import ast
        value = ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If literal_eval fails, treat as string
        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            value = value_str[1:-1]
        else:
            value = value_str

    return key, value


def apply_overrides(config: SkyRLConfig, overrides: List[str]) -> SkyRLConfig:
    """
    Apply CLI overrides to a Pydantic config.

    Args:
        config: Base SkyRLConfig to modify
        overrides: List of override strings like ["trainer.epochs=30", ...]

    Returns:
        Modified config (same object, modified in place)

    Example:
        >>> cfg = create_default_config()
        >>> cfg = apply_overrides(cfg, ["trainer.epochs=30", "trainer.logger=console"])
    """
    for override in overrides:
        key, value = parse_override(override)
        set_nested_attr(config, key, value)

    return config


def collect_overrides(args: Optional[List[str]] = None) -> List[str]:
    """
    Collect config overrides from command-line arguments.

    Overrides are arguments in --key.path=value format.

    Args:
        args: Optional argument list (uses sys.argv[1:] if None)

    Returns:
        List of override strings (without leading --)

    Example:
        >>> # sys.argv = ["script.py", "--trainer.epochs=30", "--help"]
        >>> overrides = collect_overrides()
        >>> # overrides = ["trainer.epochs=30"]
    """
    if args is None:
        args = sys.argv[1:]

    overrides = []
    for arg in args:
        if arg.startswith("--") and "=" in arg:
            overrides.append(arg[2:])  # Strip leading --

    return overrides
