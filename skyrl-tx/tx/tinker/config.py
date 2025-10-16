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

        if field.annotation is bool:
            # For boolean flags, use 'store_true' if the default is False, 'store_false' if default is True
            if field.default is False:
                kwargs["action"] = "store_true"
            else:
                assert field.default is True
                # For True defaults, use --no-{name} flag with store_false
                parser.add_argument(f"--no-{name.replace('_', '-')}", action="store_false", dest=name, **kwargs)
                continue
        else:
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
        field = cfg.model_fields[field_name]

        if field.annotation is bool:
            # For boolean flags with False default, only add the flag if True (store_true)
            if field.default is False:
                if value:
                    argv.append(f"--{field_name.replace('_', '-')}")
            # For boolean flags with True default, only add --no-flag if False (store_false)
            else:
                assert field.default is True
                if not value:
                    argv.append(f"--no-{field_name.replace('_', '-')}")
        else:
            argv.append(f"--{field_name.replace('_', '-')}")
            argv.append(str(value))
    return argv
