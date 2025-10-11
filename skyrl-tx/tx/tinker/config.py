"""Configuration for the Tinker engine."""

import argparse
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    checkpoints_base_path: str = Field(..., description="Base path where checkpoints will be stored")
    max_lora_adapters: int = Field(default=32, description="Maximum number of LoRA adapters")
    max_lora_rank: int = Field(default=32, description="Maximum LoRA rank")


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    """Add Pydantic model fields to an ArgumentParser.

    Args:
        parser: The ArgumentParser to add arguments to
        model: The Pydantic model class
    """
    fields = model.model_fields
    for name, field in fields.items():
        kwargs = {
            "dest": name,
            "help": field.description,
        }

        # Add type if available
        if field.annotation is not None:
            kwargs["type"] = field.annotation

        # Add default if it's not required
        if not field.is_required():
            kwargs["default"] = field.default

        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
