"""Teacher log-probability clients for on-policy distillation.

On-policy distillation needs exactly one thing from the teacher: for a sequence the student
produced, the teacher's log-probability of every response token given everything before it.
That is a prefill (a forward pass over ``prompt + response`` with nothing sampled), so the teacher
belongs on an inference engine, not on a training worker.

``TeacherLogprobClient`` is the abstract base class. It owns everything every backend needs
identically -- the concurrency limit, the empty-response case, the length and finiteness
invariants -- and a backend implements one method, ``_compute_logprobs``.
Callers only ever use ``compute_logprobs``.

Backends:

- ``SkyRLTeacherClient``: a vLLM deployment this job launched (``trainer.teacher.backend="skyrl"``),
  driven through its ``RemoteInferenceClient`` exactly as the student's engines are; scoring goes
  through the client's ``sample()`` with ``include_prompt_logprobs``.
- ``VLLMTeacherClient``: one vLLM endpoint you run per teacher model (a stock ``vllm serve``, or a
  router in front of several servers such as the one SkyRL's ``serve`` entrypoint starts), through the
  OpenAI-compatible ``/v1/completions`` with vLLM's ``prompt_logprobs`` parameter.

``VLLMTeacherClient`` is over ``aiohttp`` so the package adds no dependency. Neither is an
``InferenceEngineInterface`` implementation, deliberately: a frozen teacher never sleeps, wakes,
syncs weights or pauses, and a class of eleven stubs would only hide that.
"""

from __future__ import annotations

import abc
import asyncio
import math
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp
from loguru import logger

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.setup import InferenceServerSetup

_HTTP_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class TeacherLogprobClient(abc.ABC):
    """A frozen model that can return its log-probability of each token of a given sequence."""

    def __init__(self, max_concurrency: int = 32):
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self._max_concurrency = max_concurrency
        # Created lazily per event loop (as RemoteInferenceClient._get_semaphores does): the
        # generator's agent-loop tasks all share one client inside the trainer's loop, and a
        # semaphore is bound to the loop that created it.
        self._sem: Optional[asyncio.Semaphore] = None
        self._sem_loop: Optional[asyncio.AbstractEventLoop] = None

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self._max_concurrency)
            self._sem_loop = loop
        return self._sem

    async def compute_logprobs(self, prompt_ids: List[int], response_ids: List[int]) -> List[float]:
        """``log p_teacher(response_ids[t] | prompt_ids + response_ids[:t])`` for every ``t``.

        Returns one float per response token, in order. Empty responses return ``[]`` without a
        request. At most ``max_concurrency`` requests are in flight per client. A wrong number of
        values or a non-finite value raises: a NaN or infinity would poison the advantages.
        """
        if not response_ids:
            return []
        async with self._semaphore():
            logprobs = await self._compute_logprobs(list(prompt_ids), list(response_ids))
        if len(logprobs) != len(response_ids):
            raise RuntimeError(f"teacher returned {len(logprobs)} logprobs for {len(response_ids)} response tokens")
        if not all(math.isfinite(value) for value in logprobs):
            raise RuntimeError(
                "teacher returned a non-finite logprob (NaN or infinity); its forward pass is numerically broken "
                "or a logit processor masked a token"
            )
        return logprobs

    @abc.abstractmethod
    async def _compute_logprobs(self, prompt_ids: List[int], response_ids: List[int]) -> List[float]:
        """Backend-specific request. Called by ``compute_logprobs``; do not call directly.

        Must return the teacher's log-probability of each response token, in order, as floats.
        ``response_ids`` is non-empty.
        """

    async def aclose(self) -> None:
        """Release network resources. The default holds none."""
        return None


def _vllm_prompt_logprob(entry: Any, token_id: int) -> float:
    """The logprob of ``token_id`` in one vLLM ``prompt_logprobs`` entry.

    An entry is ``{token_id: {"logprob": float, "rank": int, "decoded_token": str}, ...}`` (keys are
    strings once serialized to JSON) covering the prompt token at that position plus any top-k
    alternatives requested, or ``None`` at position 0, which has no context to be scored under. The
    lookup by id works for any k, so a server configured to return more entries is fine.
    """
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"vLLM has no prompt logprob for token id {token_id} (entry {entry!r}); position 0 of a sequence "
            "cannot be scored, so the prompt must be non-empty"
        )
    value = entry.get(str(token_id), entry.get(token_id))
    if value is None:
        raise RuntimeError(
            f"vLLM returned no logprob for token id {token_id}; the teacher's tokenizer does not match the student's"
        )
    logprob = value.get("logprob") if isinstance(value, dict) else value
    if logprob is None:
        raise RuntimeError(f"vLLM returned a null logprob for token id {token_id}")
    return float(logprob)


def normalize_server_url(server_url: Optional[str]) -> str:
    """Validate a vLLM teacher's server root and drop a trailing slash.

    The client appends ``/v1/completions``, so a URL that already ends in ``/v1`` (the OpenAI SDK's
    ``base_url`` convention) would request ``/v1/v1/completions``; it is refused instead.
    """
    url = (server_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("vLLM teacher needs a server url, e.g. http://host:8000")
    if url.endswith("/v1"):
        raise ValueError(
            "server_url is the server root and the client appends /v1/completions; drop the trailing /v1 from "
            f"{server_url!r}"
        )
    return url


class VLLMTeacherClient(TeacherLogprobClient):
    """Teacher behind one vLLM endpoint you run, scored through the OpenAI-compatible ``/v1/completions``.

    ``server_url`` is this teacher model's one endpoint: a single ``vllm serve``, a data-parallel one, or
    a router in front of several servers (SkyRL's ``serve`` entrypoint starts one and logs it as
    ``proxy_url``). Spreading requests over replicas is the endpoint's job; the client neither
    round-robins nor fails over.

    Request: integer prompt ``prompt_ids + response_ids``, ``max_tokens=1`` (vLLM refuses 0; the
    generated token is discarded), ``temperature=1.0`` and vLLM's ``prompt_logprobs=0``: zero top-k
    alternatives, so each position carries only the prompt token's own entry (vLLM always includes
    it), which is the value SkyRL's own Tinker-compatible sampling path sends for the same purpose.
    The choice's ``prompt_logprobs`` has one entry per prompt token; the last ``len(response_ids)``
    entries are looked up by the token id that was sent, so a tokenizer mismatch surfaces as a
    missing id rather than a wrong number. ``model_name`` is the served model name.

    Transport: one ``aiohttp`` session per event loop (a session is bound to the loop that created
    it); timeouts, connection errors and retryable statuses are retried on the same endpoint with
    exponential backoff.
    """

    def __init__(
        self,
        model_name: str,
        *,
        server_url: str,
        max_concurrency: int = 32,
        request_timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__(max_concurrency=max_concurrency)
        if not model_name:
            raise ValueError("vLLM teacher needs the served model name, e.g. Qwen/Qwen3-32B")
        self._model_name = model_name
        self._server_url = normalize_server_url(server_url)
        self._completions_url = f"{self._server_url}/v1/completions"
        self._headers = {"Content-Type": "application/json"}
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._max_retries = max(0, max_retries)
        self._sessions: Dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def server_url(self) -> str:
        return self._server_url

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        session = self._sessions.get(loop)
        if session is None or session.closed:
            session = aiohttp.ClientSession(timeout=self._timeout, headers=self._headers)
            self._sessions[loop] = session
        return session

    async def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``body`` to the endpoint, retrying transient failures on it with exponential backoff."""
        url = self._completions_url
        session = await self._get_session()
        for attempt in range(self._max_retries + 1):
            try:
                async with session.post(url, json=body) as resp:
                    if resp.status in _HTTP_RETRY_STATUSES:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status, message=await resp.text()
                        )
                    if resp.status >= 400:
                        raise RuntimeError(f"teacher HTTP {resp.status} from {url}: {(await resp.text())[:500]}")
                    return await resp.json()
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientResponseError) as exc:
                if attempt >= self._max_retries:
                    raise RuntimeError(
                        f"teacher request to {url} failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                delay = min(8.0, 0.5 * (2**attempt)) * (1.0 + 0.25 * random.random())
                logger.warning(f"teacher request to {url} failed ({exc}); retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _compute_logprobs(self, prompt_ids: List[int], response_ids: List[int]) -> List[float]:
        body = {
            "model": self._model_name,
            "prompt": prompt_ids + response_ids,
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 0,
        }
        response = await self._post(body)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {response!r}")
        prompt_logprobs = choices[0].get("prompt_logprobs")
        if prompt_logprobs is None:
            raise RuntimeError(
                "vLLM returned no prompt_logprobs; the server must accept the prompt_logprobs completions parameter"
            )
        expected = len(prompt_ids) + len(response_ids)
        if len(prompt_logprobs) != expected:
            raise RuntimeError(f"vLLM returned {len(prompt_logprobs)} prompt logprobs for {expected} tokens")
        tail = list(prompt_logprobs)[-len(response_ids) :]
        return [_vllm_prompt_logprob(entry, token_id) for entry, token_id in zip(tail, response_ids)]

    async def aclose(self) -> None:
        sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            if not session.closed:
                await session.close()


class SkyRLTeacherClient(TeacherLogprobClient):
    """Teacher on a vLLM deployment this job launched, driven through its ``RemoteInferenceClient``.

    Scores through the client's ``sample()`` (SkyRL's Tinker-shaped route, ``/inference/v1/generate``)
    with ``include_prompt_logprobs=True``, which the client maps to vLLM's ``prompt_logprobs=0``: each
    position carries the prompt token's own logprob and nothing else. ``max_tokens=1`` because vLLM
    refuses 0; the generated token is discarded. The client answers with one entry per sent token,
    looked up by the token id that was sent, so a tokenizer mismatch surfaces as a missing entry
    (``None``) rather than a plausible-looking number. Routing through the deployment's router, the
    per-engine concurrency cap and the retry policy are the client's, the same ones the student's
    rollouts get; ``model`` is the client's ``model_name``, the name the servers were launched under.

    The client also owns the deployment it drives: ``server_setup``, the ``InferenceServerSetup`` the
    launch returned (``skyrl.train.opd.teacher_launch.launch_teacher``), is held for the client's
    lifetime because Ray terminates a non-detached actor once every handle to it is gone, and the
    setup's server groups hold the only handles to the teacher's server actors.
    """

    def __init__(
        self,
        client: "RemoteInferenceClient",
        *,
        server_setup: Optional["InferenceServerSetup"] = None,
        max_concurrency: int = 32,
    ):
        super().__init__(max_concurrency=max_concurrency)
        self._client = client
        # Keeps the teacher's server actors alive for as long as this client exists; None when the
        # caller owns the deployment.
        self._server_setup = server_setup

    @property
    def client(self) -> "RemoteInferenceClient":
        return self._client

    @property
    def server_setup(self) -> Optional["InferenceServerSetup"]:
        return self._server_setup

    async def _compute_logprobs(self, prompt_ids: List[int], response_ids: List[int]) -> List[float]:
        token_ids = prompt_ids + response_ids
        response = await self._client.sample(
            {
                "json": {
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": token_ids}]},
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 1, "temperature": 1.0},
                    "include_prompt_logprobs": True,
                }
            }
        )
        prompt_logprobs = response.get("prompt_logprobs")
        if prompt_logprobs is None:
            raise RuntimeError("the teacher deployment returned no prompt_logprobs")
        if len(prompt_logprobs) != len(token_ids):
            raise RuntimeError(
                f"the teacher deployment returned {len(prompt_logprobs)} prompt logprobs for {len(token_ids)} tokens"
            )
        tail = list(prompt_logprobs)[-len(response_ids) :]
        missing = [i for i, value in enumerate(tail) if value is None]
        if missing:
            shown = missing[:5]
            raise RuntimeError(
                f"the teacher deployment returned no logprob for response position(s) {shown} "
                f"(token ids {[response_ids[i] for i in shown]}); the teacher's vocabulary may differ from the student's"
            )
        return [float(value) for value in tail]

    async def aclose(self) -> None:
        # TODO (kyuds): consider gracefully shutting down the deployment in server_setup here (router,
        # server groups, placement group) when the run ends. Today it goes down with the Ray job, like
        # the student's engines; an explicit stop would matter for a long-lived driver that runs several
        # experiments in one job, where placement groups outlive the run.
        await self._client.teardown()
