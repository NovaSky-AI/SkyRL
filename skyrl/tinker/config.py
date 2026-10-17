"""Configuration for the Tinker engine."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from cloudpathlib import AnyPath
from pydantic import BaseModel, ConfigDict, Field, model_validator

SKYRL_TRAIN_BACKENDS = ("fsdp", "megatron")

# Marks a server-controlled key that no `backend_config` value can match.
_ALWAYS_REJECTED = object()

_TORCH_PROFILER_CONFIG_KEY = "trainer.policy.torch_profiler_config"

# The request's `loss_fn` picks the policy loss function. `policy_loss_type` is still read by
# `ppo_policy_loss`, where "dual_clip" enables dual-clip PPO and any other value fails an assert.
_TINKER_POLICY_LOSS_TYPES = ("regular", "dual_clip")


def flatten_backend_config(backend_config: dict) -> dict[str, Any]:
    """Flatten nested mapping values in ``backend_config`` into dot-separated keys.

    ``SkyRLTrainConfig.from_cli_overrides`` reads ``{"trainer.placement": {"colocate_all": False}}``
    the same as ``{"trainer.placement.colocate_all": False}``. This returns the second form, so key
    lookups see both spellings. Non-mapping values and empty mappings are kept as they are.
    """
    flat = {}
    for key, value in backend_config.items():
        if isinstance(value, dict) and value:
            for sub_key, sub_value in flatten_backend_config(value).items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def validate_skyrl_train_backend_config(
    backend_config: dict,
    *,
    backend: str,
    runtime_role: str,
    base_model: str,
    base_model_checkpoint_path: str | None = None,
) -> None:
    """Reject ``backend_config`` values the Tinker server overrides on the SkyRL-Train backends.

    A key the server sets itself is rejected when its value differs from the one the server uses;
    keys with no fixed server value (such as the learning rate, which every ``optim_step`` replaces)
    are rejected for any value.

    Args:
        backend_config: The ``--backend-config`` dict. Nested mappings are checked as dot-keys.
        backend: ``fsdp`` or ``megatron``.
        runtime_role: ``trainer``, ``inference`` or ``combined``.
        base_model: The ``--base-model`` value.
        base_model_checkpoint_path: The ``--base-model-checkpoint-path`` value, if any.

    Raises:
        ValueError: Listing every rejected key and what sets its value instead.
    """
    model_path = base_model_checkpoint_path or base_model
    controlled: dict[str, tuple[Any, str]] = {
        "strategy": (backend, f"set by --backend ({backend!r})"),
        "trainer.strategy": (backend, f"set by --backend ({backend!r})"),
        "runtime_role": (runtime_role, f"set by --runtime-role ({runtime_role!r})"),
        "use_ray": (_ALWAYS_REJECTED, "only read by the jax backend"),
        "trainer.policy.model.path": (model_path, f"set by --base-model ({model_path!r})"),
        "trainer.critic.model.path": (model_path, f"set by --base-model ({model_path!r})"),
        "trainer.algorithm.use_kl_loss": (False, "the Tinker server does not support a KL loss"),
    }
    for role in ("policy", "critic"):
        prefix = f"trainer.{role}.optimizer_config"
        controlled[f"{prefix}.lr"] = (
            _ALWAYS_REJECTED,
            "replaced by AdamParams.learning_rate before every optim_step",
        )
        schedule_reason = "the client schedules the learning rate through AdamParams.learning_rate"
        controlled[f"{prefix}.scheduler"] = ("constant_with_warmup", schedule_reason)
        controlled[f"{prefix}.num_warmup_steps"] = (0, schedule_reason)
    if runtime_role != "inference":
        # An inference-only runtime builds its config without a client LoRA config, so there the
        # rank is what enables LoRA on the inference engines.
        controlled["trainer.policy.model.lora.rank"] = (
            _ALWAYS_REJECTED,
            "set by the LoRA rank the client passes to create_model",
        )
    if runtime_role != "combined":
        controlled["trainer.placement.colocate_all"] = (False, f"a {runtime_role}-only runtime never colocates")
    if base_model_checkpoint_path is not None:
        controlled["generator.inference_engine.served_model_name"] = (
            base_model,
            f"set to --base-model ({base_model!r}) when --base-model-checkpoint-path is given",
        )

    problems = []
    for key, value in sorted(flatten_backend_config(backend_config).items()):
        if key == _TORCH_PROFILER_CONFIG_KEY or key.startswith(_TORCH_PROFILER_CONFIG_KEY + "."):
            problems.append(
                f"{key}: torch profiling is controlled at runtime; start the server with --torch-profiler "
                "and call /start_profiling and /stop_profiling"
            )
        elif key in controlled:
            server_value, reason = controlled[key]
            if server_value is _ALWAYS_REJECTED or value != server_value:
                problems.append(f"{key}={value!r}: {reason}")
        elif key == "trainer.algorithm.policy_loss_type" and value not in _TINKER_POLICY_LOSS_TYPES:
            problems.append(
                f"{key}={value!r}: must be 'regular' or 'dual_clip'; the request's loss_fn picks the loss function"
            )
    if problems:
        raise ValueError(
            "--backend-config sets keys the Tinker server controls or does not support:\n"
            + "\n".join(f"  {problem}" for problem in problems)
        )


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    base_model_checkpoint_path: str | None = Field(
        default=None,
        description="Compatible base-weight directory to load instead of base_model",
        json_schema_extra={"argparse_type": str},
    )
    runtime_role: Literal["trainer", "inference", "combined"] = Field(
        default="combined", description="GPU runtime role", json_schema_extra={"argparse_type": str}
    )
    backend: str = Field(default="megatron", description="Backend to use for training and inference")
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
        default=2048.0,
        gt=0,
        description=(
            "Read timeout in seconds for API-side requests forwarded to the "
            "SkyRL-Train-managed inference engine. This must cover time spent "
            "queued behind other requests as well as generation time: with the "
            "default unlimited connection count a large rollout burst waits inside "
            "vLLM's queue, and 128x128 bursts routinely exceed 300s there."
        ),
        json_schema_extra={
            "argparse_type": float,
            "env_var": "SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC",
        },
    )
    external_future_retrieved_ttl_sec: float = Field(
        default=300.0,
        gt=0,
        description=(
            "How long a forwarded sample result stays in memory after it was delivered, so an "
            "SDK retry after a lost HTTP response still finds it. Must outlast the SDK's worst-case "
            "re-poll gap (45s poll timeout + up to 30s backoff, twice). Memory for long-output "
            "rollouts is roughly completion rate x result size x this window."
        ),
    )
    external_future_completed_ttl_sec: float = Field(
        default=600.0,
        gt=0,
        description="How long a completed but never-delivered forwarded sample result stays in memory.",
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

    @model_validator(mode="after")
    def _check_skyrl_train_backend_config(self) -> "EngineConfig":
        # Runs in the API process before the engine subprocess starts, so a rejected key stops
        # the server at launch.
        if self.backend in SKYRL_TRAIN_BACKENDS:
            validate_skyrl_train_backend_config(
                self.backend_config,
                backend=self.backend,
                runtime_role=self.runtime_role,
                base_model=self.base_model,
                base_model_checkpoint_path=self.base_model_checkpoint_path,
            )
        return self


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

    def validate_startup(self, backend: str) -> None:
        """Fail fast at server startup on settings that cannot work."""
        if backend == "jax":
            raise ValueError(
                "`--torch-profiler` is not supported for the jax backend. torch.profiler only "
                "records the SkyRL-Train policy workers; use `--backend fsdp` or `--backend megatron`."
            )
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
