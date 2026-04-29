"""``OpenRouterInferenceEngine``: an ``InferenceEngineInterface`` that routes
generation to the OpenRouter chat-completions API.

Used by ``RLMGymGenerator`` when ``generator.frozen_openrouter_model`` is set —
in-REPL ``llm_query`` calls are routed here so they hit a frozen external
model (e.g. ``openai/gpt-5.4-nano``) instead of the policy.

Implemented as a thin subclass of ``RemoteInferenceEngine`` to inherit
abstract-method satisfaction (``wake_up``/``sleep``/weight-sync, etc.) — none
of those are called for an external API engine, but the abstract base
requires they exist. Only ``__init__`` and ``generate`` are overridden:
OpenRouter speaks chat-completions with a JSON message list, not vLLM's
token-id completions.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger
from transformers import PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import (
    RemoteInferenceEngine,
)


class OpenRouterInferenceEngine(RemoteInferenceEngine):
    """OpenRouter-backed inference engine.

    Subclasses ``RemoteInferenceEngine`` purely to inherit the
    ``InferenceEngineInterface`` abstract-method stubs. The methods we don't
    override (``wake_up``, ``sleep``, ``init_weight_update_communicator``,
    weight-sync, etc.) are never called for an external API engine, so the
    inherited implementations are dead code paths.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        api_key: Optional[str] = None,
    ):
        # Skip RemoteInferenceEngine.__init__ deliberately: it prepends "http://" to
        # the URL and assumes a vLLM/SGLang server with weight-sync hooks. We have a
        # full https URL and no weight loader.
        self.url = self.BASE_URL
        self.model_name = model
        self.engine_backend = "openrouter"
        self.tokenizer = tokenizer
        self._tp_size = self._pp_size = self._dp_size = self._ep_size = 1
        self._weight_loader = None  # OpenRouter has no weights to sync

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable must be set when using OpenRouterInferenceEngine"
            )

        # Per-call usage counters for observability.
        self.usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "requests": 0}

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Send a batched chat-completions request to OpenRouter and return decoded outputs.

        OpenRouter is OpenAI-compatible but does not accept token-id completions, so we
        decode each prompt back to text via the tokenizer, build a single-message chat,
        POST it, and re-tokenize the response so downstream loss-mask / step-wise
        bookkeeping still works.
        """
        prompts = input_batch.get("prompts")
        prompt_token_ids: Optional[List[List[int]]] = input_batch.get("prompt_token_ids")
        sampling_params: Dict[str, Any] = input_batch.get("sampling_params") or {}

        # Accept either prompts (list of message lists) or prompt_token_ids.
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided.")
        if prompts is not None and prompt_token_ids is not None:
            raise ValueError("Provide only one of `prompts` / `prompt_token_ids`.")

        # Build per-prompt message lists. For token IDs we decode to a single user message;
        # for prompts we use them as-is (already in OpenAI message format).
        if prompts is not None:
            message_lists: List[List[Dict[str, str]]] = list(prompts)
        else:
            message_lists = [
                [{"role": "user", "content": self.tokenizer.decode(ids, skip_special_tokens=True)}]
                for ids in prompt_token_ids
            ]

        body_template: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_tokens": sampling_params.get("max_generate_length", 1024),
            "reasoning": {"effort": "none"},
        }
        if sampling_params.get("additional_kwargs"):
            body_template.update(sampling_params["additional_kwargs"])

        async def _post_one(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            body = dict(body_template)
            body["messages"] = messages
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                        async with session.post(f"{self.url}/chat/completions", json=body, headers=headers) as resp:
                            resp.raise_for_status()
                            return await resp.json()
                except Exception as e:
                    last_exc = e
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
            raise RuntimeError(f"OpenRouter API call failed after 3 attempts: {last_exc}") from last_exc

        responses_data = await asyncio.gather(*(_post_one(msgs) for msgs in message_lists))

        responses: List[str] = []
        response_ids: List[List[int]] = []
        stop_reasons: List[str] = []
        for data in responses_data:
            usage = data.get("usage", {}) or {}
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.usage["requests"] += 1
            self.usage["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "") or ""
            responses.append(text)
            response_ids.append(self.tokenizer.encode(text, add_special_tokens=False))
            stop_reasons.append(choice.get("finish_reason") or "stop")

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=None,
            rollout_expert_indices=None,
        )
