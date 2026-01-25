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


def create_override_parser(description: str = "SkyRL Training") -> argparse.ArgumentParser:
    """
    Create an ArgumentParser that captures remaining args as overrides.

    Args:
        description: Description for the argument parser

    Returns:
        ArgumentParser configured to capture overrides

    Example:
        >>> parser = create_override_parser("GSM8K Training")
        >>> parser.add_argument("--data-dir", default="~/data/gsm8k")
        >>> args, overrides = parser.parse_known_args()
        >>> # overrides = ["trainer.epochs=30", ...]
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config Overrides:
  Use --key=value or --key.nested.field=value syntax to override config fields.

  Examples:
    --trainer.epochs=30
    --trainer.policy.model.path="Qwen/Qwen2.5-7B"
    --trainer.placement.policy_num_gpus_per_node=8
    --data.train_data='["/path/to/train.parquet"]'
    --trainer.algorithm.use_kl_loss=true
    --generator.backend=sglang
        """
    )
    return parser


def parse_args_with_overrides(
    parser: Optional[argparse.ArgumentParser] = None,
    args: Optional[List[str]] = None
) -> tuple[argparse.Namespace, List[str]]:
    """
    Parse command-line arguments, separating known args from config overrides.

    Args:
        parser: Optional ArgumentParser (creates default if None)
        args: Optional argument list (uses sys.argv if None)

    Returns:
        Tuple of (parsed_args, override_list)

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--num-gpus", type=int, default=4)
        >>> args, overrides = parse_args_with_overrides(parser)
        >>> # args.num_gpus = 4
        >>> # overrides = ["trainer.epochs=30", ...]
    """
    if parser is None:
        parser = create_override_parser()

    if args is None:
        args = sys.argv[1:]

    # Separate known args from overrides (anything starting with --)
    known_args = []
    overrides = []

    i = 0
    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # Check if it's a key=value override
            if "=" in arg:
                # Remove leading --
                override_str = arg[2:]
                overrides.append(override_str)
                i += 1
            else:
                # It's a flag or argument with value
                # Check if it's defined in parser
                try:
                    # Try to parse just this arg to see if it's known
                    test_args = known_args + [arg]
                    if i + 1 < len(args) and not args[i + 1].startswith("--"):
                        test_args.append(args[i + 1])
                    parser.parse_known_args(test_args)
                    # If successful, it's a known arg
                    known_args.append(arg)
                    if i + 1 < len(args) and not args[i + 1].startswith("--"):
                        known_args.append(args[i + 1])
                        i += 2
                    else:
                        i += 1
                except:
                    # Unknown flag, might be override without =
                    # Treat as override if next item exists and doesn't start with --
                    if i + 1 < len(args) and not args[i + 1].startswith("--"):
                        override_str = f"{arg[2:]}={args[i + 1]}"
                        overrides.append(override_str)
                        i += 2
                    else:
                        known_args.append(arg)
                        i += 1
        else:
            known_args.append(arg)
            i += 1

    # Parse known args
    parsed_args = parser.parse_args(known_args)

    return parsed_args, overrides
