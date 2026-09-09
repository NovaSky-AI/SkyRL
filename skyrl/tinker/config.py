"""Configuration for the Tinker engine."""

import argparse
import json
import os
from pathlib import Path

from cloudpathlib import AnyPath
from pydantic import BaseModel, ConfigDict, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    backend: str = Field(default="jax", description="Backend to use for training and inference")
    backend_config: dict = Field(
        default_factory=dict,
        description="Backend-specific configuration as JSON string",
        json_schema_extra={"argparse_type": json.loads},
    )
    checkpoints_base: AnyPath = Field(
        default=AnyPath("/tmp/skyrl_checkpoints"),
        description="Base path where checkpoints will be stored",
    )
    database_url: str = Field(
        default=f'sqlite:///{Path(__file__).parent / "tinker.db"}',
        description="Database URL (e.g., postgresql://user:password@localhost:5432/tinker). If not set, uses SKYRL_DATABASE_URL env var or defaults to SQLite",
        json_schema_extra={"argparse_type": str, "env_var": "SKYRL_DATABASE_URL"},
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
    forwarding_inference_max_connections: int | None = Field(
        default=None,
        description=(
            "Optional cap on the httpx connection pool used by "
            "SkyRLTrainInferenceForwardingClient to forward sample requests to "
            "the engine-managed vLLM. The natural backpressure chain is "
            "httpx pool -> vllm-router -> vLLM's max_num_seqs; this knob "
            "only sets the API-side connection ceiling. Default `None` is "
            "unlimited — vllm-router/vLLM are the only queues — which is "
            "usually what you want. Raise your host's `ulimit -n` for very "
            "high fan-out (the only hard cost of unlimited connections is "
            "file descriptors). Set an int to enforce a per-API-process cap."
        ),
        json_schema_extra={"argparse_type": lambda v: None if v == "None" else int(v)},
    )
    forwarding_inference_timeout_sec: float = Field(
        default=300.0,
        gt=0,
        description=(
            "Read timeout in seconds for API-side requests forwarded to the "
            "SkyRL-Train-managed inference engine. This must cover time spent "
            "queued behind other requests as well as generation time."
        ),
        json_schema_extra={
            "argparse_type": float,
            "env_var": "SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC",
        },
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
    torch_profiler: dict = Field(
        default_factory=dict,
        description=(
            "torch.profiler settings as JSON, e.g. "
            '{"export_dir": "s3://bucket/traces", "ranks": [0]}. '
            "Passing the flag enables the /start_profiling, /stop_profiling and "
            "/profiling_status endpoints; omitting it makes them return 404. "
            "See TinkerTorchProfilerConfig for the accepted fields."
        ),
        json_schema_extra={"argparse_type": json.loads},
    )
    """NOTE: annotated `dict`, not TinkerTorchProfilerConfig. `config_to_argv` dispatches on
    `field.annotation is dict` to JSON-serialize this for the engine subprocess; a BaseModel
    annotation falls through to `str(value)` and the engine gets a repr `json.loads` cannot
    read. Validate into the model at the point of use instead."""


class TinkerTorchProfilerConfig(BaseModel):
    """Operator-controlled torch profiler settings for the Tinker server.

    Split of responsibility: the operator fixes where traces land and which ranks
    pay the cost (here, at startup); the client picks the schedule and names the
    capture (per request, at /start_profiling). There is no `enabled` field --
    passing `--torch-profiler` at all is what turns the endpoints on -- so
    `export_dir` is required and there is no configured-but-off state.
    """

    model_config = ConfigDict(extra="forbid")

    export_dir: str = Field(
        ...,
        description="Where traces are written. Local absolute path, or a cloud URI (s3://, gs://, gcs://).",
    )
    ranks: list[int] = Field(default_factory=lambda: [0], description="Global ranks to profile.")
    max_session_duration_sec: int = Field(
        default=7200,
        gt=0,
        description=(
            "How long one client may hold the single profiling slot before the engine "
            "finalizes the session and releases it. Guards against a client that starts "
            "profiling and never calls /stop_profiling, which would otherwise lock out "
            "every other client."
        ),
    )

    def validate_startup(self) -> None:
        """Fail fast at server startup on settings that cannot work."""
        if not self.ranks:
            raise ValueError("`torch_profiler.ranks` must be non-empty.")
        from skyrl.backends.skyrl_train.utils.io.io import is_cloud_path

        if not is_cloud_path(self.export_dir) and not os.path.isabs(self.export_dir):
            raise ValueError(
                f"`torch_profiler.export_dir` must be an absolute local path or a cloud URI; "
                f"got {self.export_dir!r}. Ray workers run from a /tmp/ray runtime working dir, "
                f"so a relative path would write traces there."
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
