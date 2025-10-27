"""Configuration for the Tinker engine."""

import argparse

from cloudpathlib import AnyPath
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    checkpoints_base: AnyPath = Field(
        default=AnyPath("/tmp/tx_checkpoints"),
        description="Base path where checkpoints will be stored",
    )
    database_url: str | None = Field(
        default=None,
        description="Database URL (e.g., postgresql://user:password@localhost:5432/tinker). If not set, uses TX_DATABASE_URL env var or defaults to SQLite",
    )
    max_lora_adapters: int = Field(default=32, description="Maximum number of LoRA adapters")
    max_lora_rank: int = Field(default=32, description="Maximum LoRA rank")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism degree to use for the model")
    micro_batch_size: int = Field(
        default=0,
        description="Micro-batch size for gradient accumulation; 0 means disabled (use full batch)",
    )
    enforce_eager: bool = Field(default=False, description="Disable JAX JIT compilation")
    shard_attention_heads: bool = Field(
        default=True,
        description="Whether to shard attention linear layers (qkvo projections) across tensor parallel devices",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Whether to use gradient checkpointing (full recomputation strategy)",
    )


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    """Add Pydantic model fields to an ArgumentParser.

    Args:
        parser: The ArgumentParser to add arguments to
        model: The Pydantic model class
    """
    import typing

    for name, field in model.model_fields.items():
        arg_name = name.replace("_", "-")
        kwargs = {
            "help": field.description,
        }

        if field.annotation is bool:
            # For boolean flags, use BooleanOptionalAction to support both --{arg_name} and --no-{arg_name}
            kwargs = {**kwargs, "action": argparse.BooleanOptionalAction, "dest": name, "default": field.default}
        else:
            # Add type if available
            if field.annotation is not None:
                # Handle Union types (e.g., str | None)
                origin = typing.get_origin(field.annotation)
                if origin is typing.Union:
                    # Get the non-None type from the union
                    args = typing.get_args(field.annotation)
                    if type(None) in args:
                        kwargs["type"] = next((arg for arg in args if arg is not type(None)), None)
                elif callable(field.annotation):
                    kwargs["type"] = field.annotation

            # Check for default value
            if field.is_required():
                # Mark as required in argparse if no default is provided
                kwargs["required"] = True
            else:
                # For optional fields, provide the default value to argparse
                kwargs["default"] = field.default

        parser.add_argument(f"--{arg_name}", **kwargs)


def config_to_argv(cfg: BaseModel) -> list[str]:
    """This should 'unparse' a config parsed by an ArgumentParser constructed by add_model."""
    argv = []
    for field_name, value in cfg.model_dump().items():
        field = cfg.model_fields[field_name]
        arg_name = field_name.replace("_", "-")

        if field.annotation is bool:
            argv.append(f"--{arg_name}" if value else f"--no-{arg_name}")
        else:
            # Skip None values - let them use defaults or environment variables
            if value is not None:
                argv.append(f"--{arg_name}")
                argv.append(str(value))
    return argv
