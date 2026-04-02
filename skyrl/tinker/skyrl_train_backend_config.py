"""Validation and config helpers for Tinker's SkyRL-Train backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skyrl.tinker import types

if TYPE_CHECKING:
    from skyrl.train.config import SkyRLTrainConfig


def resolve_tinker_skyrl_train_strategy(backend: str) -> str:
    """Normalize Tinker backend names to the strategy used in SkyRL-Train config."""
    if backend in ("fsdp", "fsdp2"):
        return "fsdp2"
    if backend == "megatron":
        return "megatron"
    raise ValueError(f"Unsupported Tinker skyrl-train backend: {backend}")


_EXACT_KEY_REJECTIONS = {
    "strategy": "fixed by `--backend`; remove `strategy` from `--backend-config`.",
    "trainer.strategy": "fixed by `--backend` and cannot be overridden from `--backend-config`.",
    "trainer.policy.model.path": "fixed by `--base-model` and cannot be overridden from `--backend-config`.",
    "trainer.policy.optimizer_config.lr": (
        "ignored by this backend; use `OptimStepInput.adam_params.learning_rate` for step-time learning rates."
    ),
    "trainer.policy.optimizer_config.scheduler": "forced by the Tinker skyrl-train backend and cannot be overridden.",
    "trainer.policy.optimizer_config.num_warmup_steps": (
        "forced by the Tinker skyrl-train backend and cannot be overridden."
    ),
    "trainer.algorithm.policy_loss_type": (
        "chosen per `forward_backward` request; use `ForwardBackwardInput.loss_fn` instead."
    ),
    "trainer.algorithm.use_kl_loss": "currently forced to `False` by the Tinker skyrl-train backend.",
    "trainer.policy.model.lora.rank": "set from `CreateModelInput.lora_config.rank`, not `--backend-config`.",
    "trainer.policy.model.lora.alpha": "set from `CreateModelInput.lora_config.alpha`, not `--backend-config`.",
    "trainer.epochs": "ignored because Tinker does not run the standard SkyRL-Train epoch loop.",
    "trainer.train_batch_size": "ignored because Tinker batches requests dynamically through the API.",
    "trainer.policy_mini_batch_size": "ignored because Tinker does not use the standard PPO trainer loop.",
    "trainer.critic_mini_batch_size": "ignored because Tinker does not initialize a critic trainer loop.",
    "trainer.eval_batch_size": "ignored because Tinker does not run the standard evaluation loop.",
    "trainer.eval_before_train": "ignored because Tinker does not run the standard evaluation loop.",
    "trainer.eval_interval": "ignored because Tinker does not run the standard evaluation loop.",
    "trainer.ckpt_interval": "ignored because checkpointing in Tinker is driven by explicit API calls.",
    "trainer.hf_save_interval": "ignored because checkpointing in Tinker is driven by explicit API calls.",
    "trainer.resume_mode": "ignored because checkpoint loading in Tinker is driven by explicit API calls.",
    "trainer.resume_path": "ignored because checkpoint loading in Tinker is driven by explicit API calls.",
    "trainer.project_name": "ignored because Tinker does not use the standard trainer logging loop.",
    "trainer.run_name": "ignored because Tinker does not use the standard trainer logging loop.",
    "trainer.logger": "ignored because Tinker does not use the standard trainer logging loop.",
    "trainer.dump_data_batch": "ignored because Tinker does not use the standard trainer logging loop.",
    "trainer.dump_eval_results": "ignored because Tinker does not use the standard evaluation loop.",
    "trainer.placement.critic_num_nodes": "ignored because Tinker only initializes a policy actor group.",
    "trainer.placement.critic_num_gpus_per_node": "ignored because Tinker only initializes a policy actor group.",
    "trainer.placement.ref_num_nodes": "ignored because Tinker only initializes a policy actor group.",
    "trainer.placement.ref_num_gpus_per_node": "ignored because Tinker only initializes a policy actor group.",
}

_PREFIX_KEY_REJECTIONS = {
    "data.": "ignored because Tinker does not use SkyRL-Train dataset loading; send data through the Tinker API.",
    "trainer.ref.": "ignored because Tinker does not initialize a reference-model actor group.",
    "trainer.critic.": "ignored because Tinker does not initialize a critic actor group.",
    "trainer.fully_async.": "ignored because Tinker does not use the fully async trainer loop.",
}

_BACKEND_PREFIX_REJECTIONS = {
    "fsdp2": {
        "trainer.policy.megatron_config.": "only supported with `--backend megatron`.",
    },
    "megatron": {
        "trainer.policy.fsdp_config.": "only supported with `--backend fsdp`.",
    },
}


def _matches_prefix(key: str, prefix: str) -> bool:
    root = prefix[:-1]
    return key == root or key.startswith(prefix)


def collect_tinker_skyrl_train_backend_override_errors(backend: str, overrides: dict[str, Any]) -> list[tuple[str, str]]:
    """Return sorted `(key, reason)` pairs for unsupported backend-config overrides."""
    resolved_strategy = resolve_tinker_skyrl_train_strategy(backend)
    errors: dict[str, str] = {}

    for key in overrides:
        if key in _EXACT_KEY_REJECTIONS:
            errors[key] = _EXACT_KEY_REJECTIONS[key]
            continue

        backend_prefix_rejections = _BACKEND_PREFIX_REJECTIONS[resolved_strategy]
        matched_backend_prefix = next(
            (prefix for prefix in backend_prefix_rejections if _matches_prefix(key, prefix)),
            None,
        )
        if matched_backend_prefix is not None:
            errors[key] = backend_prefix_rejections[matched_backend_prefix]
            continue

        matched_prefix = next((prefix for prefix in _PREFIX_KEY_REJECTIONS if _matches_prefix(key, prefix)), None)
        if matched_prefix is not None:
            errors[key] = _PREFIX_KEY_REJECTIONS[matched_prefix]

    return sorted(errors.items())


def validate_tinker_skyrl_train_backend_overrides(backend: str, overrides: dict[str, Any]) -> None:
    """Raise a clear error when `--backend-config` contains misleading Tinker overrides."""
    errors = collect_tinker_skyrl_train_backend_override_errors(backend, overrides)
    if not errors:
        return

    formatted_errors = "\n".join(f"- {key}: {reason}" for key, reason in errors)
    raise ValueError(f"Invalid --backend-config for the Tinker skyrl-train backend:\n{formatted_errors}")


def build_tinker_skyrl_train_config(
    base_model: str,
    backend: str,
    overrides: dict[str, Any],
    lora_config: types.LoraConfig | None = None,
) -> "SkyRLTrainConfig":
    """Build the effective SkyRL-Train config used by Tinker's FSDP/Megatron backends."""
    resolved_strategy = resolve_tinker_skyrl_train_strategy(backend)
    validate_tinker_skyrl_train_backend_overrides(resolved_strategy, overrides)

    from skyrl.train.config import SkyRLTrainConfig

    user_overrides = dict(overrides)
    user_overrides.pop("strategy", None)
    user_overrides["trainer.policy.model.path"] = base_model

    cfg = SkyRLTrainConfig.from_cli_overrides(user_overrides)

    # Tinker manages optimizer stepping and learning rate externally.
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0

    # TODO(tyler): Support KL loss in the Tinker skyrl-train backend.
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.strategy = resolved_strategy

    if lora_config is not None and lora_config.rank > 0:
        cfg.trainer.policy.model.lora.rank = lora_config.rank
        cfg.trainer.policy.model.lora.alpha = int(lora_config.alpha)

    return cfg


def validate_and_build_tinker_skyrl_train_config(
    base_model: str,
    backend: str,
    overrides: dict[str, Any],
) -> "SkyRLTrainConfig":
    """Validate raw `--backend-config` keys and build the startup SkyRL-Train config."""
    return build_tinker_skyrl_train_config(base_model=base_model, backend=backend, overrides=overrides)
