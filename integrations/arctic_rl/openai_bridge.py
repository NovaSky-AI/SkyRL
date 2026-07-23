"""OpenAI HTTP shim in front of ``arctic_client.generate()``.

Harbor's terminus-2 agent lives inside a sandbox VM and only reaches the model
over HTTP (LiteLLM's ``hosted_vllm/<name>`` → ``POST /v1/chat/completions``).
Arctic's sampling side is a ``ReplicaPool`` reached in-process via
``arctic_client.generate(prompts, sampling_params)``. This module bridges the
two by:

  1. Building a small FastAPI app that exposes ``/v1/chat/completions`` and
     ``/v1/completions``.
  2. Serving it in a daemon thread via ``uvicorn.Server`` on a port we own.
  3. Implementing SkyRL's ``InferenceEngineInterface`` so ``HarborGenerator``
     can call ``get_endpoint_url()`` / ``finish_session()`` on the same object
     with no wrapper.

Only the OpenAI paths call into Arctic; every other interface method is a
no-op or forwards to the underlying ``arctic_client`` (weight-sync goes
through the Arctic RL server directly, not through this shim).
"""

import asyncio
import logging
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)

logger = logging.getLogger(__name__)


# Fields the arctic sampling worker forwards to ``vllm.SamplingParams``
# (see ``arctic_inference.server.worker.InferenceWorker.generate``). Anything
# not in this set is dropped from the payload so we don't hit vLLM's strict
# kwargs check when LiteLLM adds a new knob.
_VLLM_SAMPLING_KEYS = frozenset(
    {
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
    }
)


def _map_finish_reason(reason: Optional[str]) -> str:
    # vLLM emits stop|length|abort; OpenAI accepts stop|length|tool_calls|...
    # Fold abort -> length (token-budget cutoff), unknown -> stop.
    if reason == "length" or reason == "abort":
        return "length"
    return "stop"


def _build_sampling_params_from_openai(
    body: Dict[str, Any],
    *,
    default_max_tokens: int,
) -> Dict[str, Any]:
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


def _openai_logprobs_from_arctic(
    arctic_result: Dict[str, Any],
    token_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[Dict[str, Any]]:
    """Translate ``arctic_client.generate()``'s logprob payload to OpenAI
    ``chat.logprobs``. Assumes vLLM's canonical
    ``list[dict[token_id, {"logprob": float, ...}]]`` shape from
    ``SamplingOutput.logprobs``; anything else falls through to zeros.
    """
    if not token_ids:
        return None

    raw = arctic_result.get("logprobs") or []
    content: List[Dict[str, Any]] = []
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        lp = 0.0
        if i < len(raw) and isinstance(raw[i], dict) and raw[i]:
            entry = raw[i].get(tid) or next(iter(raw[i].values()))
            if isinstance(entry, dict):
                lp = float(entry.get("logprob", 0.0))
        content.append(
            {
                "token": tok,
                "logprob": lp,
                "bytes": list(tok.encode("utf-8", errors="ignore")),
                "top_logprobs": [],
            }
        )
    return {"content": content}


def _bad_request(message: str, code: int = 400) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "BadRequest", "code": code}}


def _server_error(message: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "InternalServerError", "code": 500}}


def _pick_port(start_port: int, host: str = "0.0.0.0") -> Tuple[int, socket.socket]:
    """Reserve a free TCP port at or above ``start_port`` (bind-and-listen)."""
    end = start_port + 128
    port = start_port
    while port < end:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(1)
            return port, sock
        except OSError:
            port += 1
    raise RuntimeError(f"No available port in [{start_port}, {end}) on {host}")


class ArcticInferenceEngineAdapter(InferenceEngineInterface):
    """Adapts ``arctic_client`` to SkyRL's ``InferenceEngineInterface``.

    Serves an OpenAI-compatible HTTP endpoint (FastAPI + uvicorn) that
    Harbor's LiteLLM agent (inside the Daytona/Modal sandbox) can reach.
    Weight-sync and other server-side operations bypass this class —
    ``arctic_client`` owns those paths and the vLLM engine the shim forwards to.
    """

    def __init__(
        self,
        *,
        arctic_client: Any,
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
        default_max_tokens: int = 4096,
        host: str = "0.0.0.0",
        port: int = 8000,
        chat_template: Optional[str] = None,
    ) -> None:
        self._arctic_client = arctic_client
        self.tokenizer = tokenizer
        self._model_name = model_name
        self._default_max_tokens = int(default_max_tokens)
        self._host = host
        self._preferred_port = int(port)
        self._custom_chat_template = chat_template

        self._server: Optional[Any] = None  # uvicorn.Server
        self._server_thread: Optional[threading.Thread] = None
        self._reserved_sock: Optional[socket.socket] = None
        self._bound_port: Optional[int] = None

    # ------------------------------------------------------------------ #
    # InferenceEngineInterface — required properties                     #
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_endpoint_url(self) -> str:
        if self._bound_port is None:
            raise RuntimeError(
                "OpenAI shim has not been started; call spin_up_http_endpoint() first."
            )
        # Harbor's sandbox reaches this via the host's routable IP; bind on
        # 0.0.0.0 (host arg) but advertise 127.0.0.1 by default so single-host
        # runs work without egress reconfig. Override with `.set_advertised_host()`
        # if the sandbox is on a different machine.
        advertised_host = self._advertised_host()
        return f"http://{advertised_host}:{self._bound_port}"

    def _advertised_host(self) -> str:
        if self._host in ("0.0.0.0", "::"):
            # Prefer explicit override; else localhost (sandbox → host loopback
            # via port forwarding, or localhost if driver + sandbox share a node).
            import os

            return os.environ.get("ARCTIC_HARBOR_SHIM_ADVERTISED_HOST", "127.0.0.1")
        return self._host

    # ------------------------------------------------------------------ #
    # InferenceEngineInterface — OpenAI paths                            #
    # ------------------------------------------------------------------ #

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
        body = request_payload["json"] if "json" in request_payload else request_payload
        messages = body.get("messages", [])
        if not messages:
            return _bad_request("empty messages in /v1/chat/completions")

        prompt_text = self._messages_to_prompt(messages)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        sampling_params = _build_sampling_params_from_openai(
            body, default_max_tokens=self._default_max_tokens
        )
        # Arctic's InferenceWorker.generate returns choice.outputs[0] only, so
        # n>1 is expressed by duplicating the prompt and stitching results
        # back into ``choices``.
        n = int(sampling_params.pop("n", 1) or 1)
        prompts = [prompt_text] * n

        want_logprobs = bool(body.get("logprobs")) or ("logprobs" in sampling_params)
        if want_logprobs:
            # Arctic's ``generate`` accepts vLLM sampling kwargs; asking for the
            # top-1 logprob populates per-token ``logprobs`` in the result dict.
            sampling_params.setdefault("logprobs", 1)

        try:
            results = await self._arctic_client.generate(
                prompts=prompts, sampling_params=sampling_params
            )
        except Exception as exc:
            logger.error("chat_completion arctic_client.generate failed: %s", exc)
            return _server_error(f"arctic_client.generate failed: {exc}")

        if not results:
            return _server_error("arctic_client.generate returned no results")

        prompt_tokens = int(results[0].get("prompt_len") or len(prompt_ids))
        completion_tokens = int(
            sum(r.get("generation_len", 0) or 0 for r in results)
        )
        choices = []
        for i, r in enumerate(results):
            text = r.get("text", "")
            token_ids = list(r.get("token_ids") or [])
            if not token_ids and text:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # ``token_ids`` sits directly on the choice — matches vLLM's
            # ``ChatCompletionResponseChoice``. LiteLLM's ``convert_dict_to_response``
            # folds non-``_CHOICES_FIELDS`` keys into ``provider_specific_fields``,
            # which is where Harbor's ``_extract_token_ids`` reads them.
            choice: Dict[str, Any] = {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _map_finish_reason(r.get("finish_reason")),
                "logprobs": _openai_logprobs_from_arctic(
                    r, token_ids, self.tokenizer
                )
                if want_logprobs
                else None,
                "token_ids": token_ids,
            }
            choices.append(choice)

        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            # vLLM emits the full prompt token IDs at the top level; Harbor's
            # LiteLLM reads them via ``response.prompt_token_ids`` (attribute
            # lookup on the pydantic ModelResponse — unknown JSON keys become
            # attributes via ``__pydantic_extra__``).
            "prompt_token_ids": prompt_ids,
        }

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload["json"] if "json" in request_payload else request_payload
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
                prompts = [self.tokenizer.decode(raw_prompt, skip_special_tokens=False)]
            elif isinstance(raw_prompt[0], list):
                prompts = [
                    self.tokenizer.decode(p, skip_special_tokens=False)
                    for p in raw_prompt
                ]
            else:
                return _bad_request(
                    f"unsupported prompt type {type(raw_prompt[0]).__name__}"
                )
        else:
            return _bad_request("missing `prompt` in /v1/completions")

        sampling_params = _build_sampling_params_from_openai(
            body, default_max_tokens=self._default_max_tokens
        )
        sampling_params.pop("n", None)

        try:
            results = await self._arctic_client.generate(
                prompts=prompts, sampling_params=sampling_params
            )
        except Exception as exc:
            logger.error("completion arctic_client.generate failed: %s", exc)
            return _server_error(f"arctic_client.generate failed: {exc}")

        texts = [r.get("text", "") for r in results]
        finish_reasons = [_map_finish_reason(r.get("finish_reason")) for r in results]
        prompt_tokens = int(sum(r.get("prompt_len", 0) or 0 for r in results)) or sum(
            self._count_tokens(p) for p in prompts
        )
        completion_tokens = int(
            sum(r.get("generation_len", 0) or 0 for r in results)
        )
        return {
            "id": "cmpl-" + uuid.uuid4().hex,
            "object": "text_completion",
            "created": int(time.time()),
            "model": self._model_name,
            "choices": [
                {"index": i, "text": t, "finish_reason": f, "logprobs": None}
                for i, (t, f) in enumerate(zip(texts, finish_reasons))
            ],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            },
        }

    async def render_chat_completion(
        self, request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply the chat template + tokenize without generating.

        Only used on HarborGenerator's multimodal path; mirrors vLLM's
        renderer output shape.
        """
        body = request_payload["json"] if "json" in request_payload else request_payload
        messages = body.get("messages", [])
        prompt_text = self._messages_to_prompt(messages)
        token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return {"prompt": prompt_text, "prompt_token_ids": token_ids}

    # ------------------------------------------------------------------ #
    # InferenceEngineInterface — SkyRL trainer paths (unused for Harbor) #
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: Optional[str] = None,
    ) -> InferenceEngineOutput:
        """Interface method — unused on the Harbor path (rollouts go over
        HTTP), implemented so eval / dry-run callers still work.
        """
        prompts = input_batch.get("prompts") or []
        prompt_texts: List[str] = []
        for prompt in prompts:
            if isinstance(prompt, list):
                prompt_texts.append(
                    self.tokenizer.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False,
                        chat_template=self._custom_chat_template,
                    )
                )
            else:
                prompt_texts.append(str(prompt))

        sampling_params = dict(input_batch.get("sampling_params") or {})
        sampling_params.setdefault("max_tokens", self._default_max_tokens)

        results = await self._arctic_client.generate(
            prompts=prompt_texts, sampling_params=sampling_params
        )

        responses: List[str] = []
        response_ids: List[List[int]] = []
        stop_reasons: List[str] = []
        for r in results:
            text = r.get("text", "")
            token_ids = r.get("token_ids") or self.tokenizer.encode(
                text, add_special_tokens=False
            )
            responses.append(text)
            response_ids.append(list(token_ids))
            stop_reasons.append(
                "completed"
                if _map_finish_reason(r.get("finish_reason")) == "stop"
                else "length"
            )

        return InferenceEngineOutput(  # type: ignore[typeddict-item]
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=None,
            prompt_logprobs=None,
            rollout_expert_indices=None,
        )

    # Weight sync flows through the Arctic RL server (arctic_client owns the
    # DeepSpeed <-> vLLM handshake); the SkyRL trainer never calls these on
    # this adapter — we return no-ops for interface completeness.
    async def init_weight_update_communicator(self, init_info):  # type: ignore[override]
        return None

    async def update_named_weights(self, request):  # type: ignore[override]
        return None

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        # Best-effort: arctic_client may expose this; if not, silently skip.
        fn = getattr(self._arctic_client, "reset_prefix_cache", None)
        if fn is None:
            return None
        try:
            return await fn(reset_running_requests=reset_running_requests)
        except TypeError:
            return await fn()

    async def sleep(self, *args: Any, **kwargs: Any):
        fn = getattr(self._arctic_client, "sleep_inference", None)
        if fn is None:
            return None
        level = kwargs.get("level", 2)
        return await fn(level=level)

    async def wake_up(self, *args: Any, **kwargs: Any):
        fn = getattr(self._arctic_client, "wake_inference", None)
        if fn is None:
            return None
        tags = kwargs.get("tags")
        try:
            return await fn(tags=tags)
        except TypeError:
            return await fn()

    async def pause_generation(self) -> None:
        return None

    async def resume_generation(self) -> None:
        return None

    async def finish_session(self, session_id: str) -> None:
        # Arctic has no session concept; best-effort no-op.
        return None

    async def get_world_size(self) -> Tuple[int, int]:
        # Not consumed on the Harbor path (used by weight-sync setup only).
        # Return (1, 1) so any accidental consumer gets a sane placeholder.
        return (1, 1)

    async def teardown(self) -> None:
        srv = self._server
        if srv is not None:
            try:
                srv.should_exit = True
            except Exception:
                pass
        thread = self._server_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    # FastAPI + uvicorn lifecycle                                        #
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # uvicorn / socket / thread aren't picklable; strip before Ray serdes.
        state["_server"] = None
        state["_server_thread"] = None
        state["_reserved_sock"] = None
        return state

    def _build_app(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="arctic-harbor-openai-shim")

        @app.get("/v1/models")
        async def _models() -> Dict[str, Any]:
            return {
                "object": "list",
                "data": [
                    {
                        "id": self._model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "arctic-rl",
                    }
                ],
            }

        @app.post("/v1/chat/completions")
        async def _chat(request: Request) -> Any:
            body = await request.json()
            result = await self.chat_completion({"json": body})
            status = 400 if isinstance(result, dict) and "error" in result else 200
            return JSONResponse(result, status_code=status)

        @app.post("/v1/completions")
        async def _completions(request: Request) -> Any:
            body = await request.json()
            result = await self.completion({"json": body})
            status = 400 if isinstance(result, dict) and "error" in result else 200
            return JSONResponse(result, status_code=status)

        @app.get("/health")
        async def _health() -> Dict[str, str]:
            return {"status": "ok"}

        return app

    def spin_up_http_endpoint(self) -> None:
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        # Reserve a port up front so we know the bound port before starting
        # uvicorn (needed for get_endpoint_url() ordering).
        port, sock = _pick_port(self._preferred_port, host=self._host)
        # uvicorn creates its own socket; free ours before it binds.
        sock.close()
        self._bound_port = port

        import uvicorn

        app = self._build_app()
        config = uvicorn.Config(
            app,
            host=self._host,
            port=port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            asyncio.run(self._server.serve())

        self._server_thread = threading.Thread(
            target=_run, name="arctic-harbor-openai-shim", daemon=True
        )
        self._server_thread.start()
        self._wait_ready(port)
        logger.info(
            "arctic-harbor OpenAI shim ready at http://%s:%s/v1 (model=%s)",
            self._advertised_host(),
            port,
            self._model_name,
        )

    def _wait_ready(self, port: int, timeout_s: float = 60.0) -> None:
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError, ConnectionError) as exc:
                last_err = exc
                time.sleep(0.2)
        raise RuntimeError(
            f"arctic-harbor OpenAI shim on :{port} failed to become ready in {timeout_s}s "
            f"(last error: {last_err!r})"
        )


__all__ = ["ArcticInferenceEngineAdapter"]
