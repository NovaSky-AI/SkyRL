from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from skyrl.train.generators.base import TrajectoryID


@dataclass
class MiniSWEStepOutput:
    prompt_token_ids: List[int]
    response_ids: List[int]
    loss_mask: List[int]
    rewards: List[float]
    rollout_logprobs: Optional[List[float]]
    stop_reason: str
    trajectory_id: TrajectoryID
    is_last_step: bool


def build_stepwise_sampling_params(base_sampling_params: Dict[str, Any], trajectory_id: TrajectoryID) -> Dict[str, Any]:
    """Request exact chat-completion tokens and sticky-route the rollout."""
    sampling_params = deepcopy(base_sampling_params)
    sampling_params["logprobs"] = True
    sampling_params["top_logprobs"] = 1
    sampling_params["session_id"] = trajectory_id.to_string()

    extra_body = dict(sampling_params.get("extra_body") or {})
    extra_body["return_token_ids"] = True
    extra_body["return_tokens_as_token_ids"] = True
    sampling_params["extra_body"] = extra_body
    return sampling_params


def _get_first_choice(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Mini-SWE raw response is missing `choices[0]`.")
    if not isinstance(choices[0], dict):
        raise ValueError("Mini-SWE raw response `choices[0]` must be a dictionary.")
    return choices[0]


def _parse_token_id(token: Any) -> int:
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        for part in reversed(token.split(":")):
            if part.isdigit():
                return int(part)
    raise ValueError(f"Unable to parse token id from logprobs token entry: {token!r}")


def _extract_response_ids(choice: Dict[str, Any]) -> List[int]:
    token_ids = choice.get("token_ids")
    if isinstance(token_ids, list) and all(isinstance(token_id, int) for token_id in token_ids):
        return token_ids

    logprobs = choice.get("logprobs", {})
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if isinstance(content, list) and content:
        return [_parse_token_id(entry.get("token")) for entry in content]

    raise ValueError(
        "Mini-SWE raw response is missing completion token ids. "
        "Expected `choices[0].token_ids` or token-id annotated `choices[0].logprobs.content`."
    )


def _extract_rollout_logprobs(choice: Dict[str, Any], expected_length: int) -> Optional[List[float]]:
    logprobs = choice.get("logprobs", {})
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if content is None:
        return None
    if not isinstance(content, list):
        raise ValueError("Mini-SWE raw response `choices[0].logprobs.content` must be a list when provided.")

    rollout_logprobs = [float(entry["logprob"]) for entry in content]
    if len(rollout_logprobs) != expected_length:
        raise ValueError(
            "Mini-SWE raw response logprobs length does not match completion token length: "
            f"{len(rollout_logprobs)} != {expected_length}"
        )
    return rollout_logprobs


def _extract_prompt_token_ids(raw_response: Dict[str, Any], choice: Dict[str, Any]) -> List[int]:
    prompt_token_ids = raw_response.get("prompt_token_ids")
    if prompt_token_ids is None:
        prompt_token_ids = choice.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not all(isinstance(token_id, int) for token_id in prompt_token_ids):
        raise ValueError(
            "Mini-SWE raw response is missing prompt token ids. "
            "Expected `prompt_token_ids` at the response or choice level."
        )
    return prompt_token_ids


def _truncate_step_to_max_seq_len(
    prompt_token_ids: List[int],
    response_ids: List[int],
    rollout_logprobs: Optional[List[float]],
    stop_reason: str,
    max_seq_len: int,
) -> tuple[List[int], List[int], Optional[List[float]], str]:
    max_response_tokens = max(0, max_seq_len - len(prompt_token_ids))
    if len(response_ids) <= max_response_tokens:
        return prompt_token_ids, response_ids, rollout_logprobs, stop_reason

    truncated_response_ids = response_ids[:max_response_tokens]
    truncated_logprobs = rollout_logprobs[:max_response_tokens] if rollout_logprobs is not None else None
    return prompt_token_ids, truncated_response_ids, truncated_logprobs, "length"


def build_stepwise_outputs_from_messages(
    messages: List[Dict[str, Any]],
    reward: float,
    trajectory_id: TrajectoryID,
    max_seq_len: int,
) -> List[MiniSWEStepOutput]:
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not assistant_messages:
        raise ValueError("Found no assistant messages in Mini-SWE trajectory output.")

    outputs: List[MiniSWEStepOutput] = []
    for i, message in enumerate(assistant_messages):
        raw_response = message.get("extra", {}).get("response")
        if not isinstance(raw_response, dict):
            raise ValueError(
                "Mini-SWE assistant message is missing the raw LiteLLM response under `message['extra']['response']`."
            )

        choice = _get_first_choice(raw_response)
        prompt_token_ids = _extract_prompt_token_ids(raw_response, choice)
        response_ids = _extract_response_ids(choice)
        rollout_logprobs = _extract_rollout_logprobs(choice, expected_length=len(response_ids))
        stop_reason = choice.get("finish_reason") or "stop"

        prompt_token_ids, response_ids, rollout_logprobs, stop_reason = _truncate_step_to_max_seq_len(
            prompt_token_ids=prompt_token_ids,
            response_ids=response_ids,
            rollout_logprobs=rollout_logprobs,
            stop_reason=stop_reason,
            max_seq_len=max_seq_len,
        )

        loss_mask = [1] * len(response_ids)
        per_token_rewards = [0.0] * len(response_ids)
        is_last_step = i == len(assistant_messages) - 1
        if is_last_step and per_token_rewards:
            per_token_rewards[-1] = float(reward)

        outputs.append(
            MiniSWEStepOutput(
                prompt_token_ids=prompt_token_ids,
                response_ids=response_ids,
                loss_mask=loss_mask,
                rewards=per_token_rewards,
                rollout_logprobs=rollout_logprobs,
                stop_reason=stop_reason,
                trajectory_id=trajectory_id,
                is_last_step=is_last_step,
            )
        )

    return outputs
