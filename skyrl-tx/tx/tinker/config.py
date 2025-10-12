"""Configuration for the Tinker engine."""

import argparse
from pathlib import Path
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    checkpoints_base: Path = Field(
        default=Path("/tmp/tx_checkpoints"),
        description="Base path where checkpoints will be stored",
    )
    max_lora_adapters: int = Field(default=32, description="Maximum number of LoRA adapters")
    max_lora_rank: int = Field(default=32, description="Maximum LoRA rank")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism degree to use for the model")


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    """Add Pydantic model fields to an ArgumentParser.

    Args:
        parser: The ArgumentParser to add arguments to
        model: The Pydantic model class
    """
    for name, field in model.model_fields.items():
        kwargs = {
            "help": field.description,
        }

        # Add type if available
        if field.annotation is not None:
            kwargs["type"] = field.annotation

        # Check for default value
        if field.is_required():
            # Mark as required in argparse if no default is provided
            kwargs["required"] = True
        else:
            # For optional fields, provide the default value to argparse
            kwargs["default"] = field.default

        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)


def config_to_argv(cfg: BaseModel) -> list[str]:
    """This should 'unparse' a config parsed by an ArgumentParser constructed by add_model."""
    argv = []
    for field_name, value in cfg.model_dump().items():
        argv.append(f"--{field_name.replace('_', '-')}")
        argv.append(str(value))
    return argv
