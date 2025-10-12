"""Configuration for the Tinker engine."""

import argparse
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    checkpoints_base_path: str = Field(..., description="Base path where checkpoints will be stored", json_schema_extra={"default": "/tmp/tx_checkpoints"})
    max_lora_adapters: int = Field(..., description="Maximum number of LoRA adapters", json_schema_extra={"default": 32})
    max_lora_rank: int = Field(..., description="Maximum LoRA rank", json_schema_extra={"default": 32})


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    """Add Pydantic model fields to an ArgumentParser.

    Args:
        parser: The ArgumentParser to add arguments to
        model: The Pydantic model class
    """
    fields = model.model_fields
    for name, field in fields.items():
        kwargs = {
            "help": field.description,
        }

        # Add type if available
        if field.annotation is not None:
            kwargs["type"] = field.annotation

        # Check for argparse default in json_schema_extra
        if field.json_schema_extra and "default" in field.json_schema_extra:
            kwargs["default"] = field.json_schema_extra["default"]
        elif field.is_required():
            # Mark as required in argparse if no default is provided
            kwargs["required"] = True

        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
