"""Teacher log-probability clients for on-policy distillation.

During on-policy distillation, the teacher only scores the student's trajectories: one prefill over
prompt + response gives its log-probability of chosen token ids at each response position. Both backends
send the same request to vLLM's ``/inference/v1/generate``: a ``[tokens, width]`` id matrix in
``prompt_logprob_token_ids``, windowed to the response by ``prompt_logprob_start``. Column 0 is the sampled
token, so ``width=1`` is sampled-token OPD, and further columns are the student's top-K candidates. The
answer is one float32 matrix, ``prompt_token_id_logprobs``. The same request can also ask for the teacher's own
top ``topk`` tokens at every position (vLLM's ``prompt_logprobs``); vLLM cannot window that to the response and
answers it as JSON over the whole sequence, so callers ask for it on a few rows.

The ``TeacherLogprobClient`` abstract base class builds and checks that request and answer, caps
concurrency, and probes a server before training (``preflight``). Creating a new teacher backend
consists of implementing ``_post_generate``. Callers use ``compute_logprobs``.

Backends:

- ``SkyRLTeacherClient``: a vLLM deployment this job launched (``trainer.teacher.backend="skyrl"``),
  driven through its ``RemoteInferenceClient``.
- ``VLLMTeacherClient``: one vLLM endpoint you run (``trainer.teacher.backend="vllm"``).

The request needs vLLM >= 0.32 with the V2 model runner. SkyRL pins 0.32.0, so only a server you run can
differ, which ``preflight`` catches.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import io
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp
import numpy as np
from loguru import logger

from skyrl.backends.utils import convert_vllm_prompt_logprobs

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.setup import InferenceServerSetup

_HTTP_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CandidateScores:
    """One trajectory's scores: ``logprobs[t, j]`` is log q(candidates[t, j] | prompt + response[:t])."""

    logprobs: np.ndarray
    """float32 ``[tokens, width]``, ``-inf`` where the candidate is padding (``-1``)."""
    topk_ids: Optional[np.ndarray] = None
    """int32 ``[tokens, topk]``: the teacher's own top tokens at each response position, best first. Only when
    requested."""
    topk_logprobs: Optional[np.ndarray] = None
    """float32 ``[tokens, topk]``, the teacher's logprobs of ``topk_ids``."""


def scoring_request(prompt_ids: List[int], candidates: np.ndarray, topk: int = 0) -> Dict[str, Any]:
    """One prefill over prompt + response. ``max_tokens=1`` because vLLM refuses 0; the token is discarded.

    Row ``i`` of ``prompt_logprob_token_ids`` scores sequence token ``prompt_logprob_start + i + 1``, so
    starting at the last prompt token gives exactly one row per response token. ``topk > 0`` also asks for the
    teacher's own top ``topk`` at every position of the sequence.
    """
    sampling_params: Dict[str, Any] = {
        "max_tokens": 1,
        "temperature": 1.0,
        "detokenize": False,
        "prompt_logprob_token_ids": candidates.tolist(),
        "prompt_logprob_start": len(prompt_ids) - 1,
    }
    if topk:
        sampling_params["prompt_logprobs"] = topk
    return {"token_ids": prompt_ids + candidates[:, 0].tolist(), "sampling_params": sampling_params}


def _parse_teacher_topk(response: Dict[str, Any], token_ids: List[int], num_tokens: int, topk: int):
    """The teacher's top ``topk`` at the last ``num_tokens`` positions of ``token_ids``, as id and logprob arrays.

    vLLM's ``prompt_logprobs`` has one entry per sequence token, and entry ``i`` is the distribution that predicts
    token ``i``, so the last ``num_tokens`` entries are the response's positions. An entry holds ``topk`` or
    ``topk + 1`` tokens (the scored token when it is outside the top ``topk``); ``convert_vllm_prompt_logprobs``
    keeps the best ``topk``.
    """
    raw = response.get("prompt_logprobs")
    if raw is None:
        raise RuntimeError("no prompt_logprobs: the server did not return the teacher's requested top-k")
    if len(raw) != len(token_ids):
        raise RuntimeError(f"expected prompt_logprobs for {len(token_ids)} tokens, got {len(raw)}")
    _, rows = convert_vllm_prompt_logprobs(token_ids, raw, topk=topk)
    tail = rows[len(rows) - num_tokens :]
    if any(row is None or len(row) != topk for row in tail):
        raise RuntimeError(f"the teacher's prompt_logprobs do not hold {topk} tokens at every response position")
    ids = np.array([[token for token, _ in row] for row in tail], dtype=np.int32)
    logprobs = np.array([[value for _, value in row] for row in tail], dtype=np.float32)
    if not np.isfinite(logprobs).all():
        raise RuntimeError("teacher returned a non-finite top-k logprob; its forward pass is numerically broken")
    return ids, logprobs


def parse_scoring_response(
    response: Dict[str, Any], prompt_ids: List[int], candidates: np.ndarray, topk: int = 0
) -> CandidateScores:
    """Decode and check vLLM's answer to ``scoring_request``: a base64 ``.npy`` float32 matrix, and with ``topk``
    the teacher's own top tokens."""
    encoded = response.get("prompt_token_id_logprobs")
    if encoded is None:
        raise RuntimeError("no prompt_token_id_logprobs: the server needs vLLM >= 0.32 with the V2 model runner")
    logprobs = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
    if logprobs.shape != candidates.shape:
        raise RuntimeError(f"expected {candidates.shape} scores, got {logprobs.shape}")
    if not np.isfinite(logprobs[candidates >= 0]).all():
        raise RuntimeError(
            "teacher returned a non-finite logprob (NaN or infinity); its forward pass is numerically broken"
        )
    logprobs = logprobs.astype(np.float32, copy=False)
    if not topk:
        return CandidateScores(logprobs)
    token_ids = list(prompt_ids) + candidates[:, 0].tolist()
    topk_ids, topk_logprobs = _parse_teacher_topk(response, token_ids, len(candidates), topk)
    return CandidateScores(logprobs, topk_ids=topk_ids, topk_logprobs=topk_logprobs)


class TeacherLogprobClient(abc.ABC):
    """A frozen model that scores given token ids at each position of a sequence."""

    def __init__(self, max_concurrency: int = 32):
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self._max_concurrency = max_concurrency
        # Created lazily per event loop: the generator's agent-loop tasks all share one client
        # inside the trainer's loop, and a semaphore is bound to the loop that created it.
        self._sem: Optional[asyncio.Semaphore] = None
        self._sem_loop: Optional[asyncio.AbstractEventLoop] = None

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self._max_concurrency)
            self._sem_loop = loop
        return self._sem

    async def compute_logprobs(
        self, prompt_ids: List[int], candidates: np.ndarray, *, topk: int = 0
    ) -> CandidateScores:
        """Score ``candidates``, a ``[tokens, width]`` int32 array whose column 0 is each response token.

        ``-1`` pads a short row and scores ``-inf``. ``topk > 0`` also returns the teacher's own top ``topk``
        tokens at each response position. Empty responses return without a request. At most
        ``max_concurrency`` requests are in flight per client. A wrong shape, a missing top-k or a non-finite
        score for a real candidate raises: a NaN or infinity would poison the advantages. An id the teacher's
        vocabulary does not have is vLLM's error.
        """
        if candidates.ndim != 2:
            raise ValueError(f"candidates must be a [tokens, width] array, got shape {candidates.shape}")
        if not len(candidates):
            empty = np.empty((0, topk), dtype=np.float32)
            return CandidateScores(
                np.empty(candidates.shape, dtype=np.float32),
                topk_ids=empty.astype(np.int32) if topk else None,
                topk_logprobs=empty if topk else None,
            )
        request = scoring_request(list(prompt_ids), candidates, topk)
        async with self._semaphore():
            response = await self._post_generate(request)
        return parse_scoring_response(response, list(prompt_ids), candidates, topk)

    async def preflight(self, width: int, topk: int = 0) -> None:
        """Score one token at ``width``, with the teacher's top ``topk`` when set, so a server that cannot run
        this flow fails before any model loads."""
        try:
            await self.compute_logprobs([0], np.zeros((1, width), dtype=np.int32), topk=topk)
        except Exception as error:
            raise RuntimeError(
                f"teacher preflight failed: {error}. OPD scores through /inference/v1/generate and needs vLLM >= 0.32 "
                f"(SkyRL pins 0.32.0), the V2 model runner and --max-logprobs >= {max(width, topk)}"
            ) from error

    @abc.abstractmethod
    async def _post_generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send ``request`` to the backend's ``/inference/v1/generate`` and return the JSON answer.

        Called by ``compute_logprobs``; do not call directly.
        """

    async def aclose(self) -> None:
        """Release network resources. The client must stay usable afterwards: ``preflight`` runs on a
        short-lived event loop and closes it. The default holds none."""
        return None


def normalize_server_url(server_url: Optional[str]) -> str:
    """Validate a vLLM teacher's server root and drop a trailing slash.

    The client appends ``/inference/v1/generate``, so a URL that ends in ``/v1`` (the OpenAI SDK's
    ``base_url`` convention) is refused instead of requesting a path that does not exist.
    """
    url = (server_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("vLLM teacher needs a server url, e.g. http://host:8000")
    if url.endswith("/v1"):
        raise ValueError(
            "server_url is the server root and the client appends /inference/v1/generate; drop the trailing /v1 "
            f"from {server_url!r}"
        )
    return url


class VLLMTeacherClient(TeacherLogprobClient):
    """Teacher behind one vLLM endpoint you run, scored through its ``/inference/v1/generate``.

    ``server_url`` is this teacher model's one endpoint: a single ``vllm serve``, a data-parallel one, or
    a router in front of several servers (SkyRL's ``serve`` entrypoint starts one and logs it as
    ``proxy_url``). Spreading requests over replicas is the endpoint's job; the client neither
    round-robins nor fails over. ``model_name`` is the served model name.

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
        self._generate_url = f"{self._server_url}/inference/v1/generate"
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
        url = self._generate_url
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

    async def _post_generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return await self._post({**request, "model": self._model_name})

    async def aclose(self) -> None:
        sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            if not session.closed:
                await session.close()


class SkyRLTeacherClient(TeacherLogprobClient):
    """Teacher on a vLLM deployment this job launched, driven through its ``RemoteInferenceClient``.

    Scores through the client's ``score_tokens``, so routing through the deployment's router, the
    per-engine concurrency cap and the retry policy are the client's, the same ones the student's
    rollouts get. The model is the client's ``model_name``, the name the servers were launched under.

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

    async def _post_generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return await self._client.score_tokens(request)

    async def aclose(self) -> None:
        # TODO (kyuds): consider gracefully shutting down the deployment in server_setup here (router,
        # server groups, placement group) when the run ends. Today it goes down with the Ray job, like
        # the student's engines; an explicit stop would matter for a long-lived driver that runs several
        # experiments in one job, where placement groups outlive the run.
        await self._client.teardown()
