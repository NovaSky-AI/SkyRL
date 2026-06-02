from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AddedMessageSpan:
    completion_start: int
    completion_end: int
    message_start: int
    message_end: int
    generation_prompt_start: int
    generation_prompt_end: int


@dataclass
class TerminalInteraction:
    prompt_id: int
    completion_id: int
    prompt_messages: list[dict[str, Any]]
    prompt_token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    completion_messages: list[dict[str, Any]] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    completion_masks: list[int] = field(default_factory=list)
    completion_observation_masks: list[int] = field(default_factory=list)
    completion_warning_masks: list[int] = field(default_factory=list)
    completion_env_output_masks: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    reward: float = 0.0
    correct: bool = False
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.prompt_messages + self.completion_messages

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.completion_token_ids

    def append_assistant(self, token_ids: list[int], logprobs: list[float], tokenizer) -> str:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        self.completion_messages.append({"role": "assistant", "content": text})
        self.completion_token_ids.extend(token_ids)
        self.completion_masks.extend([1] * len(token_ids))
        self.completion_observation_masks.extend([0] * len(token_ids))
        self.completion_warning_masks.extend([0] * len(token_ids))
        self.completion_env_output_masks.extend([0] * len(token_ids))
        self.completion_logprobs.extend(logprobs)
        return text

    def append_tokens(
        self,
        token_ids: list[int],
        masks: list[int],
        logprobs: list[float],
        role: str,
        text: str,
        observation_masks: list[int] | None = None,
        warning_masks: list[int] | None = None,
        env_output_masks: list[int] | None = None,
    ) -> None:
        if len(token_ids) != len(masks) or len(token_ids) != len(logprobs):
            raise ValueError("token_ids, masks, and logprobs must have the same length")
        for name, extra_masks in (
            ("observation_masks", observation_masks),
            ("warning_masks", warning_masks),
            ("env_output_masks", env_output_masks),
        ):
            if extra_masks is not None and len(extra_masks) != len(token_ids):
                raise ValueError(f"token_ids and {name} must have the same length")
        self.completion_messages.append({"role": role, "content": text})
        self.completion_token_ids.extend(token_ids)
        self.completion_masks.extend(masks)
        self.completion_observation_masks.extend(observation_masks or [0] * len(token_ids))
        self.completion_warning_masks.extend(warning_masks or [0] * len(token_ids))
        self.completion_env_output_masks.extend(env_output_masks or [0] * len(token_ids))
        self.completion_logprobs.extend(logprobs)
