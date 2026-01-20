from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from skyrl_train.inference_engines.base import ConversationType, InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend


DEFAULT_SUMMARY_PROMPT = (
    "Your context window is full. Summarize the conversation so far so another model can continue the task.\n"
    "Be concise and structured. Include:\n"
    "- Objective\n"
    "- Key facts/constraints\n"
    "- Current plan and open questions\n"
    "- Next action to take\n"
    "Return only the summary wrapped in <summary></summary>."
)


@dataclass
class FoldResult:
    folded: bool
    summary_text: Optional[str] = None
    summary_prompt_ids: Optional[List[int]] = None
    summary_output_ids: Optional[List[int]] = None
    summary_logprobs: Optional[List[float]] = None
    summary_stop_reason: Optional[str] = None
    new_chat_history: Optional[ConversationType] = None
    new_input_ids: Optional[List[int]] = None


class ContextFolder:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        inference_engine_client,
        backend: str,
        base_sampling_params: DictConfig,
        chat_template_kwargs: Dict[str, Any],
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.inference_engine_client = inference_engine_client
        self.backend = backend
        self.base_sampling_params = base_sampling_params
        self.chat_template_kwargs = chat_template_kwargs

        self.enabled = bool(cfg.get("enabled", False))
        self.trigger_ratio = float(cfg.get("trigger_ratio", 0.8))
        self.min_tokens = int(cfg.get("min_tokens", 0))
        self.max_folds = int(cfg.get("max_folds", 1))
        self.keep_initial_prompt_tokens = int(cfg.get("keep_initial_prompt_tokens", -1))
        self.keep_last_messages = int(cfg.get("keep_last_messages", 0))
        summary_prompt = cfg.get("summary_prompt", None)
        if summary_prompt is None:
            summary_prompt = DEFAULT_SUMMARY_PROMPT
        self.summary_prompt = str(summary_prompt)

        summary_prefix = cfg.get("summary_prefix", None)
        if summary_prefix is None:
            summary_prefix = "[Previous conversation summary]\n{summary}\n\nPlease continue the task."
        self.summary_prefix = str(summary_prefix)

        self.summary_role = str(cfg.get("summary_role", "user"))
        self.summary_max_tokens = cfg.get("summary_max_tokens", None)
        self.summary_sampling_params = cfg.get("summary_sampling_params", None)
        self.include_summary_in_training = bool(cfg.get("include_summary_in_training", False))
        self.summary_pattern = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)

        if self.summary_role not in {"user", "system"}:
            raise ValueError("context_folding.summary_role must be 'user' or 'system'")

    def fold_trigger(self, current_input_length: int, max_input_length: int, fold_count: int) -> bool:
        if not self.enabled:
            return False
        if fold_count >= self.max_folds:
            return False
        if current_input_length < self.min_tokens:
            return False
        threshold_length = int(max_input_length * self.trigger_ratio)
        return current_input_length >= threshold_length

    async def fold(
        self,
        chat_history: ConversationType,
        current_input_length: int,
        max_input_length: int,
        initial_chat_history_length: int,
        session_id: str,
        fold_count: int,
    ) -> FoldResult:
        if not self.fold_trigger(current_input_length, max_input_length, fold_count):
            return FoldResult(folded=False)

        keep_initial = self._resolve_keep_initial(initial_chat_history_length, len(chat_history))
        keep_last = self._resolve_keep_last(keep_initial, len(chat_history))

        if keep_initial + keep_last >= len(chat_history):
            logger.debug("Context folding skipped: not enough history to summarize")
            return FoldResult(folded=False)

        summary_request, summary_prompt_ids, tail_messages = self._build_summary_request(
            chat_history, keep_initial, keep_last, max_input_length
        )
        if summary_request is None or summary_prompt_ids is None:
            logger.warning("Context folding skipped: summary prompt exceeds max input length")
            return FoldResult(folded=False)

        summary_sampling_params = self._build_summary_sampling_params()
        summary_session_id = f"{session_id}_summary_{fold_count}"

        engine_input = InferenceEngineInput(
            prompt_token_ids=[summary_prompt_ids],
            session_ids=[summary_session_id],
            sampling_params=summary_sampling_params,
        )
        engine_output = await self.inference_engine_client.generate(engine_input)
        summary_text = engine_output["responses"][0]
        summary_output_ids = engine_output["response_ids"][0]
        summary_stop_reason = engine_output["stop_reasons"][0]
        summary_logprobs = None
        if engine_output.get("response_logprobs") is not None:
            summary_logprobs = engine_output["response_logprobs"][0]

        summary_text = self._extract_summary(summary_text)
        if not summary_text:
            logger.warning("Context folding skipped: empty summary")
            return FoldResult(folded=False)

        summary_message = {
            "role": self.summary_role,
            "content": self._render_summary_prefix(summary_text),
        }

        initial_messages = chat_history[:keep_initial]
        new_chat_history = initial_messages + [summary_message] + tail_messages
        new_input_ids = self.tokenizer.apply_chat_template(
            new_chat_history,
            add_generation_prompt=True,
            tokenize=True,
            **self.chat_template_kwargs,
        )

        logger.info(
            f"Context folded: {len(chat_history)} -> {len(new_chat_history)} messages "
            f"(summary tokens: {len(summary_output_ids)})"
        )

        return FoldResult(
            folded=True,
            summary_text=summary_text,
            summary_prompt_ids=summary_prompt_ids,
            summary_output_ids=summary_output_ids,
            summary_logprobs=summary_logprobs,
            summary_stop_reason=summary_stop_reason,
            new_chat_history=new_chat_history,
            new_input_ids=new_input_ids,
        )

    def _resolve_keep_initial(self, initial_chat_history_length: int, total_messages: int) -> int:
        keep_initial = self.keep_initial_prompt_tokens
        if keep_initial < 0:
            keep_initial = initial_chat_history_length
        keep_initial = max(0, min(keep_initial, total_messages))
        return keep_initial

    def _resolve_keep_last(self, keep_initial: int, total_messages: int) -> int:
        keep_last = max(0, self.keep_last_messages)
        if keep_initial + keep_last > total_messages:
            keep_last = max(0, total_messages - keep_initial)
        return keep_last

    def _build_summary_request(
        self,
        chat_history: ConversationType,
        keep_initial: int,
        keep_last: int,
        max_input_length: int,
    ) -> Tuple[Optional[ConversationType], Optional[List[int]], Optional[ConversationType]]:
        initial_messages = chat_history[:keep_initial]
        tail_messages = chat_history[len(chat_history) - keep_last :] if keep_last > 0 else []
        history_to_summarize = chat_history[keep_initial : len(chat_history) - keep_last]
        if not history_to_summarize:
            return None, None, None, None

        summary_instruction = {"role": "user", "content": self.summary_prompt}
        trimmed_history = list(history_to_summarize)

        while True:
            summary_request = initial_messages + trimmed_history + [summary_instruction]
            summary_prompt_ids = self.tokenizer.apply_chat_template(
                summary_request,
                add_generation_prompt=True,
                tokenize=True,
                **self.chat_template_kwargs,
            )
            if len(summary_prompt_ids) <= max_input_length:
                return summary_request, summary_prompt_ids, tail_messages
            if not trimmed_history:
                break
            trimmed_history = trimmed_history[1:]

        return None, None, None

    def _build_summary_sampling_params(self) -> Optional[Dict[str, Any]]:
        summary_cfg = OmegaConf.create({})
        if self.base_sampling_params is not None:
            summary_cfg = OmegaConf.merge(summary_cfg, self.base_sampling_params)
        if self.summary_sampling_params is not None:
            summary_cfg = OmegaConf.merge(summary_cfg, self.summary_sampling_params)
        if self.summary_max_tokens is not None:
            summary_cfg = OmegaConf.merge(summary_cfg, {"max_generate_length": int(self.summary_max_tokens)})
        if len(summary_cfg) == 0:
            return None
        return get_sampling_params_for_backend(self.backend, summary_cfg)

    def _extract_summary(self, summary_text: str) -> str:
        match = self.summary_pattern.search(summary_text)
        if match:
            return match.group(1).strip()
        return summary_text.strip()

    def _render_summary_prefix(self, summary_text: str) -> str:
        if "{summary}" in self.summary_prefix:
            return self.summary_prefix.format(summary=summary_text)
        return f"{self.summary_prefix}{summary_text}"
