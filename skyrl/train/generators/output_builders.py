from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.utils import (
    get_response_ids_and_loss_mask_from_messages,
    get_rollout_metrics,
)

RewardType = Union[float, List[float]]


@dataclass
class RetokenizedTrajectory:
    prompt_messages: ConversationType
    response_messages: ConversationType
    reward: RewardType
    stop_reason: Optional[str] = None
    assistant_message_logprobs: Optional[List[List[float]]] = None
    env_class: Optional[str] = None
    env_metrics: Optional[Dict[str, Any]] = None


@dataclass
class TokenizedTrajectory:
    prompt_token_ids: List[int]
    response_ids: List[int]
    loss_mask: List[int]
    reward: RewardType
    stop_reason: Optional[str] = None
    rollout_logprobs: Optional[List[float]] = None
    env_class: Optional[str] = None
    env_metrics: Optional[Dict[str, Any]] = None
    rollout_expert_indices: Optional[List[List[List[int]]]] = None


T = TypeVar("T")


def _normalize_optional_values(name: str, values: Sequence[Optional[T]]) -> Optional[List[T]]:
    has_values = [value is not None for value in values]
    if any(has_values) and not all(has_values):
        raise ValueError(f"`{name}` must be provided for all trajectories or omitted for all trajectories")
    if not any(has_values):
        return None
    return [value for value in values if value is not None]


def _validate_rollout_expert_indices(
    prompt_token_ids: Sequence[List[int]],
    response_ids: Sequence[List[int]],
    rollout_expert_indices: Optional[List[List[List[List[int]]]]],
) -> None:
    if rollout_expert_indices is None:
        return
    for i, (prompt_ids, cur_response_ids, cur_expert_indices) in enumerate(
        zip(prompt_token_ids, response_ids, rollout_expert_indices)
    ):
        expected_seq_len = len(prompt_ids) + len(cur_response_ids)
        if len(cur_expert_indices) != expected_seq_len:
            raise ValueError(
                "Expected `rollout_expert_indices` to cover the full prompt + response sequence for every "
                f"trajectory, but sample {i} had sequence length {len(cur_expert_indices)} and expected "
                f"{expected_seq_len}"
            )


def _validate_generator_output(
    *,
    prompt_token_ids: Sequence[List[int]],
    response_ids: Sequence[List[int]],
    rewards: Sequence[RewardType],
    loss_masks: Sequence[List[int]],
    rollout_logprobs: Optional[Sequence[List[float]]],
) -> None:
    if len(response_ids) <= 0:
        raise ValueError("No outputs generated")

    if len(prompt_token_ids) != len(response_ids):
        raise ValueError(
            "Mismatch between the number of prompts and responses in GeneratorOutput: "
            f"{len(prompt_token_ids)} prompt sequences vs {len(response_ids)} response sequences"
        )

    if len(loss_masks) != len(response_ids):
        raise ValueError(
            "Mismatch between the number of response IDs and loss masks in GeneratorOutput: "
            f"{len(response_ids)} responses vs {len(loss_masks)} loss masks"
        )

    if len(rewards) != len(response_ids):
        raise ValueError(
            "Mismatch between the number of response IDs and rewards in GeneratorOutput: "
            f"{len(response_ids)} responses vs {len(rewards)} rewards"
        )

    if rollout_logprobs is not None and len(rollout_logprobs) != len(response_ids):
        raise ValueError(
            "Mismatch between the number of response IDs and rollout logprob lists in GeneratorOutput: "
            f"{len(response_ids)} responses vs {len(rollout_logprobs)} rollout logprob lists"
        )

    rewards_are_token_level = [isinstance(reward, list) for reward in rewards]
    if any(rewards_are_token_level) and not all(rewards_are_token_level):
        raise ValueError("`rewards` must be either all response-level scalars or all token-level reward lists")

    for i, cur_response_ids in enumerate(response_ids):
        if len(cur_response_ids) != len(loss_masks[i]):
            raise ValueError(
                f"Response ids and loss masks must have the same length for sample {i}, "
                f"got {len(cur_response_ids)} and {len(loss_masks[i])}"
            )

        if rewards_are_token_level[i] and len(cur_response_ids) != len(rewards[i]):
            raise ValueError(
                f"Token-level rewards must match response length for sample {i}, "
                f"got {len(rewards[i])} rewards and {len(cur_response_ids)} response ids"
            )

        if rollout_logprobs is not None and len(cur_response_ids) != len(rollout_logprobs[i]):
            raise ValueError(
                f"Rollout logprobs must match response length for sample {i}, "
                f"got {len(rollout_logprobs[i])} rollout logprobs and {len(cur_response_ids)} response ids"
            )


def _finalize_generator_output(
    *,
    prompt_token_ids: List[List[int]],
    response_ids: List[List[int]],
    rewards: List[RewardType],
    loss_masks: List[List[int]],
    stop_reasons: Sequence[Optional[str]],
    rollout_logprobs: Sequence[Optional[List[float]]],
    env_classes: Sequence[Optional[str]],
    env_metrics: Sequence[Optional[Dict[str, Any]]],
    rollout_metrics: Optional[Dict[str, Any]] = None,
    rollout_expert_indices: Optional[Sequence[Optional[List[List[List[int]]]]]] = None,
    validate: bool = True,
) -> GeneratorOutput:
    if not prompt_token_ids:
        raise ValueError("Expected at least one trajectory when building GeneratorOutput")

    normalized_stop_reasons = _normalize_optional_values("stop_reasons", stop_reasons)
    normalized_rollout_logprobs = _normalize_optional_values("rollout_logprobs", rollout_logprobs)
    normalized_env_classes = _normalize_optional_values("env_class", env_classes)
    normalized_env_metrics = _normalize_optional_values("env_metrics", env_metrics)
    normalized_rollout_expert_indices = (
        _normalize_optional_values("rollout_expert_indices", rollout_expert_indices)
        if rollout_expert_indices is not None
        else None
    )

    if normalized_env_metrics is not None and normalized_env_classes is None:
        raise ValueError("`env_class` must be provided for all trajectories when `env_metrics` are provided")

    _validate_rollout_expert_indices(prompt_token_ids, response_ids, normalized_rollout_expert_indices)

    if rollout_metrics is None:
        rollout_metrics = get_rollout_metrics(
            response_ids,
            rewards,
            env_metrics=normalized_env_metrics,
            env_classes=normalized_env_classes,
        )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": prompt_token_ids,
        "response_ids": response_ids,
        "rewards": rewards,
        "loss_masks": loss_masks,
        "stop_reasons": normalized_stop_reasons,
        "rollout_metrics": rollout_metrics,
        "rollout_logprobs": normalized_rollout_logprobs,
        "rollout_expert_indices": normalized_rollout_expert_indices,
        "trajectory_ids": None,
        "is_last_step": None,
    }

    if validate:
        _validate_generator_output(
            prompt_token_ids=prompt_token_ids,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            rollout_logprobs=normalized_rollout_logprobs,
        )

    return generator_output


def build_generator_output_from_messages(
    trajectories: Sequence[RetokenizedTrajectory],
    tokenizer,
    *,
    chat_template: Optional[str] = None,
    rollout_metrics: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> GeneratorOutput:
    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[RewardType] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[Optional[str]] = []
    rollout_logprobs: List[Optional[List[float]]] = []
    env_classes: List[Optional[str]] = []
    env_metrics: List[Optional[Dict[str, Any]]] = []

    for trajectory in trajectories:
        cur_prompt_token_ids = tokenizer.apply_chat_template(
            trajectory.prompt_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            chat_template=chat_template,
        )
        cur_response_ids, cur_loss_mask, cur_rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            trajectory.response_messages,
            tokenizer,
            assistant_logprobs=trajectory.assistant_message_logprobs,
            chat_template=chat_template,
        )
        prompt_token_ids.append(cur_prompt_token_ids)
        response_ids.append(cur_response_ids)
        rewards.append(trajectory.reward)
        loss_masks.append(cur_loss_mask)
        stop_reasons.append(trajectory.stop_reason)
        rollout_logprobs.append(cur_rollout_logprobs)
        env_classes.append(trajectory.env_class)
        env_metrics.append(trajectory.env_metrics)

    return _finalize_generator_output(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_logprobs=rollout_logprobs,
        env_classes=env_classes,
        env_metrics=env_metrics,
        rollout_metrics=rollout_metrics,
        validate=validate,
    )


def build_generator_output_from_tokenized(
    trajectories: Sequence[TokenizedTrajectory],
    *,
    rollout_metrics: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> GeneratorOutput:
    prompt_token_ids = [trajectory.prompt_token_ids for trajectory in trajectories]
    response_ids = [trajectory.response_ids for trajectory in trajectories]
    rewards = [trajectory.reward for trajectory in trajectories]
    loss_masks = [trajectory.loss_mask for trajectory in trajectories]
    stop_reasons = [trajectory.stop_reason for trajectory in trajectories]
    rollout_logprobs = [trajectory.rollout_logprobs for trajectory in trajectories]
    env_classes = [trajectory.env_class for trajectory in trajectories]
    env_metrics = [trajectory.env_metrics for trajectory in trajectories]
    rollout_expert_indices = [trajectory.rollout_expert_indices for trajectory in trajectories]

    return _finalize_generator_output(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_logprobs=rollout_logprobs,
        env_classes=env_classes,
        env_metrics=env_metrics,
        rollout_metrics=rollout_metrics,
        rollout_expert_indices=rollout_expert_indices,
        validate=validate,
    )
