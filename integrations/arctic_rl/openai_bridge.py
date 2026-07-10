"""OpenAI HTTP shim in front of ``arctic_client.generate()``.

Harbor's terminus-2 agent inside the sandbox talks LiteLLM
``hosted_vllm/<name>`` → ``POST /v1/chat/completions``. Arctic's sampling
side is a ``ReplicaPool`` reached via ``arctic_client.generate(prompts,
sampling_params)``. This adapter duck-types the fields SkyRL's existing
FastAPI shim reads (``model_name``, ``chat_completion``, ``completion``)
so ``inference_engine_client_http_endpoint.serve()`` runs verbatim.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Fields that the arctic sampling worker forwards to ``vllm.SamplingParams``
# (see ``arctic_inference.server.worker.InferenceWorker.generate``). Anything
# not in this set is dropped from the payload so we don't hit vLLM's strict
# kwargs check when LiteLLM adds a new knob.
_VLLM_SAMPLING_KEYS = frozenset({
    "n",
    "best_of",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "min_tokens",
    "seed",
    "stop",
    "stop_token_ids",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logprobs",
    "prompt_logprobs",
    "logit_bias",
    "include_stop_str_in_output",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "ignore_eos",
})


def _map_finish_reason(reason: Optional[str]) -> str:
    # vLLM emits stop|length|abort; OpenAI accepts stop|length|tool_calls|...
    # Fold abort -> length (token-budget cutoff), unknown -> stop.
    if reason == "length" or reason == "abort":
        return "length"
    return "stop"


def _sanitize_assistant_text(text: str) -> str:
    """Drop leading whitespace on the raw model completion.

    Covers the residual case where a plain (non-thinking) response starts
    with ``\\n\\n``: concatenated with the chat template's trailing ``\\n``
    after ``<|im_start|>assistant`` this becomes three consecutive newlines,
    which Qwen3's BPE fuses into token 1406 (``\\n\\n\\n``) rather than the
    expected ``[198, ...]``. That trips SkyRL's assertion in
    ``get_response_ids_and_loss_mask_from_messages`` that each assistant
    message starts with the generation prompt tokens.

    For thinking-mode responses (``<think>...</think>\\n\\n{content}``) the
    ``reasoning_content=None`` field we emit below prevents LiteLLM from
    stripping the think block off ``content``; without that, LiteLLM's
    ``_parse_content_for_reasoning`` would leave a bare ``\\n\\n{content}``
    downstream and re-trigger the same fusion.
    """
    return text.lstrip()


def _build_sampling_params_from_openai(
    body: Dict[str, Any],
    *,
    default_max_tokens: int,
) -> Dict[str, Any]:
    """Extract vLLM-compatible sampling params from an OpenAI request body."""
    out: Dict[str, Any] = {k: v for k, v in body.items() if k in _VLLM_SAMPLING_KEYS}

    # ``max_completion_tokens`` is OpenAI's newer name for ``max_tokens``.
    if "max_completion_tokens" in body and "max_tokens" not in out:
        out["max_tokens"] = body["max_completion_tokens"]
    out.setdefault("max_tokens", int(default_max_tokens))

    # LiteLLM's /chat/completions ``logprobs`` is a bool; vLLM wants an int
    # (top-k). Harbor doesn't consume logprobs, so drop the bool form.
    if isinstance(out.get("logprobs"), bool):
        out["logprobs"] = 1 if out["logprobs"] else None

    return {k: v for k, v in out.items() if v is not None}


def _openai_text_response(
    *,
    model_name: str,
    texts: Sequence[str],
    finish_reasons: Sequence[str],
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    return {
        "id": "cmpl-" + uuid.uuid4().hex,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {"index": i, "text": text, "finish_reason": fr, "logprobs": None}
            for i, (text, fr) in enumerate(zip(texts, finish_reasons))
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        },
    }


def _bad_request(message: str, code: int = 400) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "BadRequest", "code": code}}


def _server_error(message: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "InternalServerError", "code": 500}}


class ArcticInferenceEngineAdapter:
    """Adapts ``arctic_client`` to the ``InferenceEngineClient`` surface that
    SkyRL's OpenAI HTTP shim depends on.

    The shim (``inference_engine_client_http_endpoint``) only reads
    ``.model_name`` and calls ``.chat_completion(payload)`` /
    ``.completion(payload)`` on whatever object it's handed. This class
    supplies exactly those, plus the HTTP-endpoint config fields the
    shim's startup helper reads (host, port, enable flag) and a picklable
    ``__getstate__`` so Ray serialization ignores the uvicorn thread.
    """

    def __init__(
        self,
        *,
        arctic_client: Any,
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
        inference_engine_cfg: Any,
        default_max_tokens: int = 4096,
    ) -> None:
        self._arctic_client = arctic_client
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.backend = "vllm"  # matches HarborGenerator's expected value
        self.inference_engine_cfg = inference_engine_cfg
        self._default_max_tokens = int(default_max_tokens)

        self.enable_http_endpoint: bool = bool(
            getattr(inference_engine_cfg, "enable_http_endpoint", False)
        )
        self.http_endpoint_host: str = str(
            getattr(inference_engine_cfg, "http_endpoint_host", "127.0.0.1")
        )
        self.http_endpoint_port: int = int(
            getattr(inference_engine_cfg, "http_endpoint_port", 8000)
        )

        # Match HarborGenerator's chat_template lookup so training-time
        # re-tokenization sees the same prompt bytes we sent to sampling.
        engine_init_kwargs = getattr(inference_engine_cfg, "engine_init_kwargs", None) or {}
        template_path = (
            engine_init_kwargs.get("chat_template", None)
            if hasattr(engine_init_kwargs, "get") else None
        )
        self._custom_chat_template: Optional[str] = None
        if template_path:
            template_path = str(template_path)
            if not os.path.isabs(template_path):
                logger.warning(
                    "chat_template=%r is not absolute; CWD-relative resolution "
                    "may differ from HarborGenerator's", template_path,
                )
            with open(template_path, "r") as f:
                self._custom_chat_template = f.read()

        self._server_thread: Optional[threading.Thread] = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_server_thread"] = None
        return state

    def spin_up_http_endpoint(self) -> None:
        """Run SkyRL's OpenAI-compatible shim in a daemon thread."""
        if not self.enable_http_endpoint:
            return
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        # Deferred import: shim pulls in fastapi + uvicorn.
        from skyrl.backends.skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            serve,
            wait_for_server_ready,
        )

        self._server_thread = threading.Thread(
            target=serve,
            args=(self,),
            kwargs={
                "host": self.http_endpoint_host,
                "port": self.http_endpoint_port,
                "log_level": "warning",
            },
            daemon=True,
        )
        self._server_thread.start()
        wait_for_server_ready(
            host=self.http_endpoint_host,
            port=self.http_endpoint_port,
            max_wait_seconds=60,
        )
        logger.info(
            "OpenAI shim ready at http://%s:%s/v1 (model=%s)",
            self.http_endpoint_host, self.http_endpoint_port, self.model_name,
        )

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=self._custom_chat_template,
        )

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return 0

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload["json"]
        messages = body.get("messages", [])
        if not messages:
            return _bad_request("empty messages in /v1/chat/completions")

        prompt_text = self._messages_to_prompt(messages)
        sampling_params = _build_sampling_params_from_openai(
            body, default_max_tokens=self._default_max_tokens,
        )
        # ``InferenceWorker.generate`` returns choice.outputs[0] only, so n>1
        # is expressed by duplicating the prompt and stitching results back
        # into ``choices``.
        n = int(sampling_params.pop("n", 1) or 1)
        prompts = [prompt_text] * n

        try:
            results = await self._arctic_client.generate(
                prompts=prompts, sampling_params=sampling_params,
            )
        except Exception as exc:
            logger.error("chat_completion arctic_client.generate failed: %s", exc)
            return _server_error(f"arctic_client.generate failed: {exc}")

        if not results:
            return _server_error("arctic_client.generate returned no results")

        prompt_tokens = int(results[0].get("prompt_len") or self._count_tokens(prompt_text))
        completion_tokens = int(sum(r.get("generation_len", 0) or 0 for r in results))
        # ``reasoning_content: None`` short-circuits LiteLLM's
        # ``_extract_reasoning_content`` regex which would otherwise split a
        # thinking-mode response like ``<think>x</think>\n\n{JSON}`` into
        # (reasoning=``x``, content=``\n\n{JSON}``) and leave the ``\n\n``
        # prefix behind, breaking re-tokenization on the training side.
        choices = [
            {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": _sanitize_assistant_text(r.get("text", "")),
                    "reasoning_content": None,
                },
                "finish_reason": _map_finish_reason(r.get("finish_reason")),
                "logprobs": None,
            }
            for i, r in enumerate(results)
        ]
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload["json"]
        raw_prompt = body.get("prompt")

        prompts: List[str]
        if isinstance(raw_prompt, str):
            prompts = [raw_prompt]
        elif isinstance(raw_prompt, list):
            if not raw_prompt:
                return _bad_request("empty prompt in /v1/completions")
            if isinstance(raw_prompt[0], str):
                prompts = list(raw_prompt)
            elif isinstance(raw_prompt[0], int):
                # Arctic wire only speaks text; decode token-id prompts.
                prompts = [self.tokenizer.decode(raw_prompt, skip_special_tokens=False)]
            elif isinstance(raw_prompt[0], list):
                prompts = [
                    self.tokenizer.decode(p, skip_special_tokens=False)
                    for p in raw_prompt
                ]
            else:
                return _bad_request(f"unsupported prompt type {type(raw_prompt[0]).__name__}")
        else:
            return _bad_request("missing `prompt` in /v1/completions")

        sampling_params = _build_sampling_params_from_openai(
            body, default_max_tokens=self._default_max_tokens,
        )
        # SkyRL's shim rejects n>1 for /completions; strip defensively.
        sampling_params.pop("n", None)

        try:
            results = await self._arctic_client.generate(
                prompts=prompts, sampling_params=sampling_params,
            )
        except Exception as exc:
            logger.error("completion arctic_client.generate failed: %s", exc)
            return _server_error(f"arctic_client.generate failed: {exc}")

        texts = [_sanitize_assistant_text(r.get("text", "")) for r in results]
        finish_reasons = [_map_finish_reason(r.get("finish_reason")) for r in results]
        prompt_tokens = int(sum(r.get("prompt_len", 0) or 0 for r in results)) or sum(
            self._count_tokens(p) for p in prompts
        )
        completion_tokens = int(sum(r.get("generation_len", 0) or 0 for r in results))
        return _openai_text_response(
            model_name=self.model_name,
            texts=texts,
            finish_reasons=finish_reasons,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
