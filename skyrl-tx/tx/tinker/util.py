"""Common utilities for the tinker module."""

import argparse


def _add_engine_options(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add engine options to a parser.

    Args:
        parser: ArgumentParser to add engine options to.

    Returns:
        The argument group containing engine options.
    """
    engine_group = parser.add_argument_group("Engine Options")
    engine_group.add_argument(
        "--base-model",
        dest="base_model",
        default="Qwen/Qwen3-0.6B",
        help="Base model name (default: Qwen/Qwen3-0.6B)",
        metavar="MODEL",
    )
    engine_group.add_argument(
        "--checkpoints-base-path",
        dest="checkpoints_base_path",
        default="/tmp/tx_checkpoints",
        help="Base path where checkpoints will be stored (default: /tmp/tx_checkpoints)",
        metavar="PATH",
    )
    engine_group.add_argument(
        "--max-lora-adapters",
        dest="max_lora_adapters",
        type=int,
        default=32,
        help="Maximum number of LoRA adapters (default: 32)",
        metavar="NUM",
    )
    engine_group.add_argument(
        "--max-lora-rank",
        dest="max_lora_rank",
        type=int,
        default=32,
        help="Maximum LoRA rank (default: 32)",
        metavar="RANK",
    )
    return engine_group


def create_engine_option_parser() -> argparse.ArgumentParser:
    """Create an ArgumentParser with engine options.

    Returns:
        ArgumentParser configured with base-model, checkpoints-base-path,
        max-lora-adapters, and max-lora-rank options.
    """
    parser = argparse.ArgumentParser(description="Tinker engine")
    _add_engine_options(parser)
    return parser


def create_api_option_parser() -> tuple[argparse.ArgumentParser, argparse._ArgumentGroup]:
    """Create an ArgumentParser with both engine and API options.

    Returns:
        Tuple of (ArgumentParser, engine_group) where engine_group can be used
        to extract engine arguments.
    """
    parser = argparse.ArgumentParser(description="Tinker API server")
    engine_group = _add_engine_options(parser)

    # Add API options group
    api_group = parser.add_argument_group("API Server Options")
    api_group.add_argument(
        "--host",
        dest="host",
        default="0.0.0.0",
        help="Host to bind the API server to (default: 0.0.0.0)",
        metavar="HOST",
    )
    api_group.add_argument(
        "--port",
        dest="port",
        type=int,
        default=8000,
        help="Port to bind the API server to (default: 8000)",
        metavar="PORT",
    )

    return parser, engine_group


def get_engine_args_from_group(group: argparse._ArgumentGroup, namespace: argparse.Namespace) -> list[str]:
    """Extract engine arguments from an argument group.

    Args:
        group: Argument group containing engine options.
        namespace: Parsed namespace from an ArgumentParser.

    Returns:
        List of command-line arguments for the engine.
    """
    engine_args = []
    for action in group._group_actions:
        if action.dest and action.dest != "help":
            value = getattr(namespace, action.dest)
            # Use the first option string (the long form like '--base-model')
            engine_args.append(action.option_strings[0])
            engine_args.append(str(value))
    return engine_args
