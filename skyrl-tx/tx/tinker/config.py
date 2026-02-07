"""Configuration for the Tinker engine."""

import argparse
import json
import os
from pathlib import Path

from cloudpathlib import AnyPath
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    backend: str = Field(default="jax", description="Backend to use for training and inference")
    backend_config: dict = Field(
        default_factory=dict,
        description="Backend-specific configuration as JSON string",
        json_schema_extra={"argparse_type": json.loads},
    )
    checkpoints_base: AnyPath = Field(
        default=AnyPath("/tmp/tx_checkpoints"),
        description="Base path where checkpoints will be stored",
    )
    database_url: str = Field(
        default=f'sqlite:///{Path(__file__).parent / "tinker.db"}',
        description="Database URL (e.g., postgresql://user:password@localhost:5432/tinker). If not set, uses TX_DATABASE_URL env var or defaults to SQLite",
        json_schema_extra={"argparse_type": str, "env_var": "TX_DATABASE_URL"},
    )
    external_inference_url: str | None = Field(
        default=None,
        description="URL of the external inference engine. If set, sample requests will be sent to the external engine instead (currently only VLLM is supported).",
        json_schema_extra={"argparse_type": str},
    )
    external_inference_api_key: str = Field(
        default="EMPTY",
        description="API key for an external inference engine. If not provided will use vLLM 'EMPTY' key convention",
    )
    external_inference_lora_base: Path = Field(
        default=Path("/tmp/lora_models"),
        description="Directory where LoRA models will be extracted for external inference engines",
    )
    session_cleanup_interval_sec: int = Field(
        default=60,
        description="How often to check for stale sessions (seconds). Set to -1 to disable cleanup.",
    )
    # The tinker client sends heartbeats every 10 seconds by default.
    # https://github.com/thinking-machines-lab/tinker/blob/2d8e9d5e00f746f39148a5d0cb760dff3f2eed43/src/tinker/lib/internal_client_holder.py#L182
    session_timeout_sec: int = Field(
        default=300,
        description="Seconds without heartbeat before session is considered stale. Set to -1 to disable cleanup.",
    )


def convert_env_var(env_name: str, env_value: str, expected_type: type):
    """Convert environment variable to expected type."""
    if expected_type is bool:
        if env_value not in ("0", "1"):
            raise ValueError(
                f"Environment variable '{env_name}' for a boolean flag must be '0' or '1', but got '{env_value}'."
            )
        return env_value == "1"
    else:
        return env_value


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    """Add Pydantic model fields to an ArgumentParser.

    The priority order of how options are handled: 1. Explicitly specified command line options,
    2. environment variables and 3. default values.

    Args:
        parser: The ArgumentParser to add arguments to
        model: The Pydantic model class
    """
    for name, field in model.model_fields.items():
        arg_name = name.replace("_", "-")
        kwargs = {
            "help": field.description,
        }

        # Check for default value, with env_var support
        default_value = field.default
        if field.json_schema_extra and "env_var" in field.json_schema_extra:
            env_name = field.json_schema_extra["env_var"]
            if env_value := os.environ.get(env_name):
                default_value = convert_env_var(env_name, env_value, field.annotation)

        if field.annotation is bool:
            # For boolean flags, use BooleanOptionalAction to support both --{arg_name} and --no-{arg_name}
            kwargs = {**kwargs, "action": argparse.BooleanOptionalAction, "dest": name, "default": default_value}
        else:
            # Check if explicit argparse_type is specified in field metadata
            argparse_type = field.json_schema_extra.get("argparse_type") if field.json_schema_extra else None
            if argparse_type is not None:
                kwargs["type"] = argparse_type
            elif field.annotation is not None:
                kwargs["type"] = field.annotation

            if field.is_required():
                # Mark as required in argparse if no default is provided
                kwargs["required"] = True
            else:
                # For optional fields, provide the default value to argparse
                kwargs["default"] = default_value

        parser.add_argument(f"--{arg_name}", **kwargs)


def config_to_argv(cfg: BaseModel) -> list[str]:
    """This should 'unparse' a config parsed by an ArgumentParser constructed by add_model."""
    argv = []
    for field_name, value in cfg.model_dump().items():
        field = cfg.model_fields[field_name]
        arg_name = field_name.replace("_", "-")

        if field.annotation is bool:
            argv.append(f"--{arg_name}" if value else f"--no-{arg_name}")
        elif field.annotation is dict:
            # Serialize dict to JSON string
            if value:
                argv.append(f"--{arg_name}")
                argv.append(json.dumps(value))
        else:
            # Skip None values - let them use defaults or environment variables
            if value is not None:
                argv.append(f"--{arg_name}")
                argv.append(str(value))
    return argv
