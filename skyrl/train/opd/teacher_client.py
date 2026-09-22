"""Teacher log-probability clients for on-policy distillation.

On-policy distillation needs exactly one thing from the teacher: for a sequence the student
produced, the teacher's log-probability of every response token given everything before it.
That is a prefill (a forward pass over ``prompt + response`` with nothing sampled), so the teacher
belongs on an inference engine, not on a training worker.

``TeacherLogprobClient`` is the abstract base class. It owns everything every backend needs
identically -- the concurrency limit, the empty-response case, the length invariant and the
startup determinism check -- and a backend implements one method, ``_compute_logprobs``.
Callers only ever use ``compute_logprobs``.
"""

from __future__ import annotations

import abc
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai"
_FIREWORKS_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


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
        request. At most ``max_concurrency`` requests are in flight per client.
        """
        if not response_ids:
            return []
        async with self._semaphore():
            logprobs = await self._compute_logprobs(list(prompt_ids), list(response_ids))
        if len(logprobs) != len(response_ids):
            raise RuntimeError(f"teacher returned {len(logprobs)} logprobs for {len(response_ids)} response tokens")
        return logprobs

    @abc.abstractmethod
    async def _compute_logprobs(self, prompt_ids: List[int], response_ids: List[int]) -> List[float]:
        """Backend-specific request. Called by ``compute_logprobs``; do not call directly.

        Must return the teacher's log-probability of each response token, in order, as floats.
        ``response_ids`` is non-empty.
        """

    async def self_test(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        n: int = 8,
        max_abs_diff: float = 0.05,
    ) -> float:
        """Score one sequence ``n`` times concurrently and require the answers to agree.

        Returns the worst absolute difference (nats) between any two answers, and raises if it
        exceeds ``max_abs_diff``. A teacher whose logprobs depend on which replica answered is not
        usable for a reverse-KL advantage whose signal is a few hundredths of a nat.
        """
        if n < 2:
            raise ValueError(f"self_test needs n >= 2 identical requests, got {n}")
        answers = await asyncio.gather(*(self.compute_logprobs(prompt_ids, response_ids) for _ in range(n)))
        worst = 0.0
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                worst = max(worst, max(abs(a - b) for a, b in zip(answers[i], answers[j])))
        if worst > max_abs_diff:
            raise RuntimeError(
                f"teacher logprobs are not reproducible: max |diff| = {worst:.4f} nats across {n} identical "
                f"requests (limit {max_abs_diff}). A serverless endpoint served by mixed replicas does this; "
                "use a dedicated deployment."
            )
        return worst

    async def aclose(self) -> None:
        """Release network resources. The default holds none."""
        return None


def _extract_echoed_logprobs(choice: Dict[str, Any]) -> Tuple[List[Optional[int]], List[float]]:
    """Read ``(token_ids, logprobs)`` off a Fireworks completions choice, in either response shape.

    ``logprobs: true`` yields ``logprobs.content[]`` items with ``token_id`` and ``logprob``; the
    integer form yields the legacy ``logprobs.token_logprobs`` / ``logprobs.token_ids`` lists.
    Entries are echoed prompt tokens first, then generated tokens (none when ``max_tokens=0``).
    """
    logprobs_obj = choice.get("logprobs") or {}
    content = logprobs_obj.get("content")
    if content is not None:
        ids: List[Optional[int]] = [item.get("token_id") for item in content]
        raw = [item.get("logprob") for item in content]
    elif logprobs_obj.get("token_logprobs") is not None:
        raw = list(logprobs_obj["token_logprobs"])
        ids = list(logprobs_obj.get("token_ids") or choice.get("token_ids") or [None] * len(raw))
    else:
        raise RuntimeError(f"Fireworks response carries no logprobs: {choice!r}")
    if any(value is None for value in raw):
        raise RuntimeError("Fireworks returned a null logprob for an echoed token")
    return ids, [float(value) for value in raw]


class FireworksTeacherClient(TeacherLogprobClient):
    """Teacher served by Fireworks: a serverless model id or a dedicated deployment id.

    Request (verified live): integer prompt ``prompt_ids + response_ids``, ``max_tokens=0``,
    ``echo_last=len(response_ids)``, ``logprobs=true``, ``return_token_ids=true``. The response
    carries exactly the echoed response tokens with their raw model logprob (never
    ``sampling_logprob``), and the echoed ids are checked against what was sent.
    """

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        max_concurrency: int = 32,
        request_timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__(max_concurrency=max_concurrency)
        if not model_name:
            raise ValueError("Fireworks teacher needs a model id, e.g. accounts/fireworks/models/gpt-oss-120b")
        if not api_key:
            raise ValueError("Fireworks teacher needs an API key")
        base_url = (base_url or DEFAULT_FIREWORKS_BASE_URL).rstrip("/")
        if base_url.endswith("/v1"):
            raise ValueError(
                "base_url is the server root; the completions path is appended by the client. "
                f"Drop the trailing /v1 from {base_url!r}."
            )
        self._model_name = model_name
        self._url = f"{base_url}/inference/v1/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._max_retries = max(0, max_retries)
        self._sessions: Dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        session = self._sessions.get(loop)
        if session is None or session.closed:
            session = aiohttp.ClientSession(timeout=self._timeout, headers=self._headers)
            self._sessions[loop] = session
        return session

    async def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST with exponential backoff on timeouts, connection errors and retryable statuses."""
        session = await self._get_session()
        for attempt in range(self._max_retries + 1):
            try:
                async with session.post(self._url, json=body) as resp:
                    if resp.status in _FIREWORKS_RETRY_STATUSES:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status, message=await resp.text()
                        )
                    if resp.status >= 400:
                        raise RuntimeError(f"Fireworks HTTP {resp.status}: {(await resp.text())[:500]}")
                    return await resp.json()
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientResponseError) as exc:
                if attempt >= self._max_retries:
                    raise RuntimeError(f"Fireworks request failed after {attempt + 1} attempt(s): {exc}") from exc
                delay = min(8.0, 0.5 * (2**attempt)) * (1.0 + 0.25 * random.random())
                logger.warning(f"Fireworks teacher request failed ({exc}); retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _compute_logprobs(self, prompt_ids: List[int], response_ids: List[int]) -> List[float]:
        body = {
            "model": self._model_name,
            "prompt": prompt_ids + response_ids,
            "max_tokens": 0,
            "temperature": 1.0,
            "logprobs": True,
            "echo_last": len(response_ids),
            "return_token_ids": True,
        }
        response = await self._post(body)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Fireworks returned no choices: {response!r}")
        ids, logprobs = _extract_echoed_logprobs(choices[0])
        n = len(response_ids)
        # Echoed tokens come first; with max_tokens=0 they are the whole list.
        echoed_ids, echoed_logprobs = ids[:n], logprobs[:n]
        if len(echoed_logprobs) != n:
            raise RuntimeError(f"Fireworks echoed {len(echoed_logprobs)} tokens, expected {n}")
        if any(tid is not None for tid in echoed_ids) and echoed_ids != response_ids:
            raise RuntimeError(
                "Fireworks echoed different token ids than were sent; the teacher's tokenizer does not "
                "match the student's, or the echo is misaligned"
            )
        return echoed_logprobs

    async def aclose(self) -> None:
        sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            if not session.closed:
                await session.close()
