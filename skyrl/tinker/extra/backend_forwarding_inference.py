"""Forwards EXTERNAL sample requests to the backend-managed vLLM.

This client is intentionally thin — its job is schema translation, not
request scheduling. Concurrency is bounded only by the httpx connection
pool; vllm-router + each vLLM server's ``max_num_seqs`` are the real
backpressure points. Don't add an asyncio.Semaphore on top: that would
just serialize work above what vLLM already manages.

What the client does:
  - Resolve the vLLM proxy URL lazily from ``EngineStateDB`` (written by
    the backend after ``_create_new_inference_client``); refresh-on-error.
  - Translate Tinker ``SamplingParams`` to a vLLM ``/v1/completions``
    payload (max_tokens, temperature, top_p, top_k, stop, stop_token_ids,
    logprobs, seed).
  - Translate vLLM's completion response back into ``SampleOutput`` —
    normalize ``finish_reason`` ({"stop", "stop_token"} -> "stop", else
    "length"), zero-fill missing logprobs (RL workloads need them), and
    set ``prompt_logprobs=None``.
  - Write the result into ``FutureDB`` so ``/api/v1/retrieve_future``
    sees a completed row.

For LoRA tenants: uses ``model=<model_id>`` since the backend's
``save_weights_for_sampler`` already registered the adapter under that
name on vLLM via ``RemoteInferenceClient.load_lora_adapter``. For
base-model sampling, uses ``model=<base_model>`` directly.

Failure modes:
  - Proxy URL not yet published -> fail the future with a clear message.
  - Connection error -> refresh cached URL once and retry; if still
    failing, fail the future.
  - vLLM 4xx/5xx -> fail the future with the upstream body verbatim.
"""

import asyncio
from datetime import datetime, timezone

import httpx
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineStateDB, FutureDB, RequestStatus
from skyrl.utils.log import logger


class BackendForwardingInferenceClient:
    """Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.engine_config = engine_config
        self.db_engine = db_engine
        self._cached_proxy_url: str | None = None
        self._cache_lock = asyncio.Lock()
        # Persistent client with a high connection pool ceiling. Backpressure
        # is layered: this httpx pool -> vllm-router (load balances across
        # vLLM servers) -> each vLLM server's max_num_seqs (final cap).
        # We intentionally do NOT impose an additional asyncio.Semaphore in
        # the API process — that would just create an extra serializing
        # bottleneck above what vLLM already enforces. Closed via aclose()
        # in api.py lifespan shutdown.
        max_conn = engine_config.forwarding_inference_max_connections
        self._http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=max(max_conn // 4, 32),
            ),
        )

    async def aclose(self) -> None:
        """Close the persistent httpx client. Called from api.py lifespan shutdown."""
        await self._http_client.aclose()

    async def _read_proxy_url_from_db(self) -> str | None:
        """Read the singleton EngineStateDB row.

        Returns the published proxy URL, or None when no row exists yet
        (e.g. before the first ``create_model``) or when the backend last
        published ``proxy_url=None`` (post-teardown).
        """
        async with AsyncSession(self.db_engine) as session:
            row = await session.get(EngineStateDB, 1)
            if row is None or row.inference_proxy_url is None:
                return None
            return row.inference_proxy_url

    async def _resolve_proxy_url(self, *, force_refresh: bool = False) -> str:
        async with self._cache_lock:
            if force_refresh or self._cached_proxy_url is None:
                url = await self._read_proxy_url_from_db()
                if url is None:
                    raise RuntimeError(
                        "inference engine not ready: no proxy URL has been "
                        "published to EngineStateDB. Either no create_model has "
                        "been issued yet, or the engine just tore down vLLM."
                    )
                self._cached_proxy_url = url
            return self._cached_proxy_url

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Forward a sample request to vLLM and write the result to FutureDB."""
        try:
            result = await self._forward_with_retry(sample_req, model_id, base_model=base_model)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("Backend-forwarded sample failed (request_id=%s)", request_id)
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_with_retry(self, sample_req, model_id: str, *, base_model: str | None) -> types.SampleOutput:
        """Forward once; on connection error, refresh the cached URL and retry once."""
        try:
            proxy_url = await self._resolve_proxy_url()
            return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
            logger.warning("Connection error to %s: %s — refreshing proxy URL", self._cached_proxy_url, e)
            proxy_url = await self._resolve_proxy_url(force_refresh=True)
            return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)

    async def _forward(
        self, proxy_url: str, sample_req, model_id: str, *, base_model: str | None
    ) -> types.SampleOutput:
        """POST {proxy_url}/v1/completions and parse into SampleOutput."""
        # vLLM identifies the LoRA adapter by the name passed to load_lora_adapter,
        # which was set to model_id in save_weights_for_sampler. For base-model
        # sampling we point at the underlying HF model name directly.
        model_name = base_model if base_model else model_id

        model_input = sample_req.prompt.to_types()
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        sp = sample_req.sampling_params
        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": sample_req.num_samples,
            "seed": sp.seed,
            "max_tokens": sp.max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            # vLLM (and the upstream vllm-router) expects an integer for the
            # OpenAI-compatible /v1/completions endpoint — the number of top
            # tokens to return logprobs for. 1 gives the chosen token's
            # logprob, which is what the Tinker SampleOutput requires.
            "logprobs": 1,
            "stream": False,
            "return_token_ids": True,
        }
        # API-level SamplingParams.stop is polymorphic: list[str] -> stop
        # strings, list[int] -> stop token ids. Mirrors the dispatch in
        # api.py SamplingParams.to_types().
        stop = getattr(sp, "stop", None)
        if stop:
            if all(isinstance(s, int) for s in stop):
                payload["stop_token_ids"] = list(stop)
            elif all(isinstance(s, str) for s in stop):
                payload["stop"] = list(stop)

        url = f"{proxy_url}/v1/completions"
        response = await self._http_client.post(url, json=payload)
        if response.status_code >= 400:
            # Surface vLLM's body verbatim (e.g. 404 for unknown LoRA name).
            raise RuntimeError(f"vLLM /v1/completions returned {response.status_code}: {response.text}")
        result = response.json()

        sequences = []
        for choice in result.get("choices", []):
            tokens = choice.get("token_ids", [])
            lp = choice.get("logprobs") or {}
            logprobs = lp.get("token_logprobs") or []
            # vLLM occasionally returns logprobs=None even when requested
            # (upstream issue under load). RL workloads compute advantages
            # against these, so zero-fill rather than return a ragged shape
            # — matches the legacy in-process sample path's behavior.
            if not logprobs and tokens:
                logger.warning("No logprobs returned from vLLM — filling with zeros")
                logprobs = [0.0] * len(tokens)
            # Map vLLM's finish_reason ({"stop", "stop_token", "length", ...})
            # to Tinker's Literal["stop", "length"]. Mirrors the
            # normalization in skyrl_train_backend._sample_with_remote_client.
            finish_reason = choice.get("finish_reason")
            stop_reason = "stop" if finish_reason in ("stop", "stop_token") else "length"
            sequences.append(
                types.GeneratedSequence(
                    tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=stop_reason,
                )
            )

        return types.SampleOutput(sequences=sequences, prompt_logprobs=None)
