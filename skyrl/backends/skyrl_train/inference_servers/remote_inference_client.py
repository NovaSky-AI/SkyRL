"""
RemoteInferenceClient - Serializable HTTP client for inference.

This is a lightweight, fully serializable HTTP client that wraps the inference
server HTTP API. It replaces the old InferenceEngineInterface for HTTP-based
inference servers.

Architecture:
-------------
This client is responsible for BOTH data plane and control plane operations:

1. Data Plane (routed through proxy_url):
   - generate, chat_completion, completion, tokenize, detokenize
   - Uses proxy_url which points to a router (vllm-router, sglang-router, InferenceRouter)
   - Router handles load balancing and session-aware routing

2. Control Plane (fan-out to all server_urls):
   - pause, resume, sleep, wake_up, reset_prefix_cache
   - init_weight_transfer, update_weights_skyrl
   - Fans out directly to all backend servers (bypassing router)
   - This allows using external routers that only handle data plane

The router (proxy_url) is expected to be a data-plane-only router. Control plane
operations are always fanned out to all backends by this client directly.

Key features:
- Serializable: Can be pickled and passed between processes
- Two URL types:
  - proxy_url: Single URL for data plane operations (routed requests)
  - server_urls: List of backend URLs for control plane operations (fan-out)
- Lazy world_size fetching from /get_server_info
- Keep-mode pause: in-flight requests are frozen by the vLLM scheduler and
  resume where they left off after /resume. No client-side retry needed.

Usage:
    client = RemoteInferenceClient(
        proxy_url="http://router:8080",  # Data plane (router)
        server_urls=["http://backend1:8000", "http://backend2:8000"],  # Control plane
    )

Comparison with existing code:
- Replaces: InferenceEngineClient + RemoteInferenceEngine (for remote-only usage)
- Key difference: Talks directly to router via HTTP, no Ray actor wrapping
- The router handles session-aware routing; this client handles control plane fan-out
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import aiohttp

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl.env_vars import (
    SKYRL_GENERATE_CONCURRENCY_PER_ENGINE,
    SKYRL_HTTP_CONNECTION_LIMIT,
)

_DATA_PLANE_RETRIES = 30

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
        WeightSyncInitInfo,
    )


logger = logging.getLogger(__name__)


class PauseMode(Enum):
    """
    Pause mode for inference servers.

    Maps to the ``mode`` query parameter on vLLM's ``/pause`` endpoint.

    Modes:
        ABORT: Abort in-flight requests immediately. Clients receive partial
            tokens with ``finish_reason="abort"`` and must retry.
        KEEP: Freeze in-flight requests in the scheduler. They resume
            exactly where they left off when ``/resume`` is called.
            No retry needed. KV cache is preserved.
        WAIT: Wait for in-flight requests to complete before pausing.
            New requests are blocked. No retry needed.
    """

    ABORT = "abort"
    KEEP = "keep"
    WAIT = "wait"


@dataclass
class RemoteInferenceClient:
    """
    Serializable HTTP client for inference. Replaces InferenceEngineInterface.

    This class maintains two URL types:
    - proxy_url: Single URL for data plane operations (routed requests)
    - server_urls: List of backend URLs for control plane operations (fan-out)

    The router (proxy_url) is expected to be a data-plane-only router (like
    vllm-router, sglang-router, or InferenceRouter). Control plane operations
    are always fanned out to all backends directly by this client.

    Usage:
        client = RemoteInferenceClient(
            proxy_url="http://router:8080",  # Data plane (router)
            server_urls=["http://backend1:8000", "http://backend2:8000"],  # Control plane
        )
    """

    proxy_url: str
    """Data plane URL (single endpoint - router or direct server)."""

    server_urls: List[str]
    """Control plane URLs (list of backend servers for fan-out)."""

    model_name: str = "default"
    """Model name for OpenAI-compatible API calls."""

    active_lora_name: Optional[str] = None
    """Name of the active LoRA adapter. If set, generation requests use this adapter instead of the base model."""

    tokenizer: Optional[Any] = None
    """Optional HF tokenizer for local tokenize/detokenize (avoids HTTP round-trips)."""

    # Private fields excluded from repr for cleaner output
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)
    _world_size: Optional[Tuple[int, int]] = field(default=None, repr=False)
    _gen_sem: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _detok_sem: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _sem_loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)

    # ---------------------------
    # Session Management
    # ---------------------------

    def _get_semaphores(self) -> Tuple[Optional[asyncio.Semaphore], Optional[asyncio.Semaphore]]:
        """Get or create the shared generate/detokenize semaphores for this client.

        Semaphores are event-loop-bound (Python 3.10+). If the running loop has
        changed since they were created, recreate them.

        All concurrent generate() calls on the same client instance share these
        semaphores, capping total in-flight requests at
        SKYRL_GENERATE_CONCURRENCY_PER_ENGINE × num_engines.
        """
        current_loop = asyncio.get_running_loop()
        if self._sem_loop is not current_loop:
            if SKYRL_GENERATE_CONCURRENCY_PER_ENGINE > 0:
                concurrency = SKYRL_GENERATE_CONCURRENCY_PER_ENGINE * len(self.server_urls)
                self._gen_sem = asyncio.Semaphore(concurrency)
                self._detok_sem = asyncio.Semaphore(concurrency)
            else:
                self._gen_sem = None
                self._detok_sem = None
            self._sem_loop = current_loop
        return self._gen_sem, self._detok_sem

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        # Re-use the existing session object if it is not closed.
        # Note that we also create a new session object if the event loop has changed, since
        # aiohttp.ClientSession is tied to the event loop.
        current_loop = asyncio.get_running_loop()
        if self._session is not None and not self._session.closed and self._session.loop != current_loop:
            # Event loop changed - the old session is unusable (bound to a dead loop).
            # Force-close the connector to release socket FDs immediately.
            _force_close_connector(self._session.connector)
            self._session = None
        if self._session is None or self._session.closed:
            # keepalive_timeout must be shorter than the server's timeout_keep_alive
            # (uvicorn default: 5s). Otherwise aiohttp reuses connections the server
            # has already closed, causing ECONNRESET under high concurrency.
            connector = aiohttp.TCPConnector(
                limit=SKYRL_HTTP_CONNECTION_LIMIT,
                keepalive_timeout=2,
            )
            self._session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None))
        return self._session

    async def _post(self, url: str, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        """POST with retry + backoff on transient connection errors.

        Between generate bursts the pool's keep-alive connections go stale
        (server closes them after ``timeout_keep_alive``).  An immediate
        retry would grab another stale connection from the same pool, so we
        sleep briefly to let the connector detect and purge dead sockets
        before the next attempt.
        """
        session = await self._get_session()
        last_exc: Optional[Exception] = None
        for attempt in range(_DATA_PLANE_RETRIES):
            try:
                async with session.post(url, json=json, headers=headers) as resp:
                    body = await resp.json()
                    raise_for_status(resp, body)
                    return body
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
                last_exc = e
                logger.warning(f"POST retry {attempt + 1}/{_DATA_PLANE_RETRIES} for {url=}: {e}")
                await asyncio.sleep(1)
                continue
        raise last_exc  # type: ignore[misc]

    # ---------------------------
    # Data Plane
    # ---------------------------

    async def generate(
        self,
        input_batch: InferenceEngineInput,
    ) -> InferenceEngineOutput:
        """
        Generate completions via /v1/completions.

        This is the interface for token-in-token-out workflows. Input will have
        token ids, and the output is token ids as well.

        Each prompt is sent as a separate request to allow the router to route
        based on session_id. All requests are made in parallel.

        With keep-mode pause, in-flight requests are frozen and resume
        transparently after /resume -- no client-side retry needed.

        Args:
            input_batch: Contains prompt_token_ids, sampling_params, and optional session_ids.

        Returns:
            InferenceEngineOutput with responses, response_ids, and stop_reasons.
        """

        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError("RemoteInferenceClient only accepts `prompt_token_ids`, not `prompts`.")

        sampling_params = input_batch.get("sampling_params") or {}
        if sampling_params.get("n", 1) > 1:
            raise ValueError("n > 1 is not supported. Use `config.generator.n_samples_per_prompt` instead.")

        session_ids = input_batch.get("session_ids")
        get_logprobs = sampling_params.get("logprobs") is not None

        # Two semaphores decouple the generate and detokenize stages:
        #   gen_sem:   limits concurrent in-flight generate requests so we don't
        #              overwhelm the router/vLLM scheduler.  Released as soon as
        #              generation finishes, so the GPU slot is freed immediately.
        #   detok_sem: limits concurrent detokenize calls independently.  Uses the
        #              same concurrency limit so detokenize never starves generate.
        # Semaphores are shared across all concurrent generate() calls on this client
        # instance, so total in-flight requests are capped at
        # SKYRL_GENERATE_CONCURRENCY_PER_ENGINE × num_engines regardless of how many
        # callers invoke generate() simultaneously.
        # TODO (sumanthrh) (RemoteInferenceClient data-plane-deprecation): We should move this outside of the client to a runner abstraction that will also parallelize client requests across processes.
        gen_sem, detok_sem = self._get_semaphores()
        batch_size = len(prompt_token_ids)
        concurrency = (
            SKYRL_GENERATE_CONCURRENCY_PER_ENGINE * len(self.server_urls) if self._gen_sem is not None else "unlimited"
        )
        logger.info(
            f"generate: batch_size={batch_size}, concurrency_limit={concurrency} "
            f"(shared across all concurrent generate() calls)"
        )

        async def _throttled_generate(idx: int) -> Dict[str, Any]:
            if gen_sem is None:
                return await self._generate_single(
                    prompt_token_ids=prompt_token_ids[idx],
                    sampling_params=sampling_params,
                    session_id=session_ids[idx] if session_ids and idx < len(session_ids) else None,
                )
            async with gen_sem:
                return await self._generate_single(
                    prompt_token_ids=prompt_token_ids[idx],
                    sampling_params=sampling_params,
                    session_id=session_ids[idx] if session_ids and idx < len(session_ids) else None,
                )

        async def _throttled_detokenize(token_ids: List[int]) -> str:
            if detok_sem is None:
                return (await self.detokenize([token_ids]))[0]
            async with detok_sem:
                return (await self.detokenize([token_ids]))[0]

        raw_results = await asyncio.gather(*[_throttled_generate(idx) for idx in range(batch_size)])
        responses = await asyncio.gather(*[_throttled_detokenize(r["response_ids"]) for r in raw_results])

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=[r["stop_reason"] for r in raw_results],
            response_ids=[r["response_ids"] for r in raw_results],
            response_logprobs=[r["response_logprobs"] for r in raw_results] if get_logprobs else None,
        )

    async def _generate_single(
        self,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
        session_id: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Generate completion for a single prompt.

        With keep-mode pause, in-flight requests are frozen by the vLLM
        scheduler and resume where they left off after /resume. No retry
        logic is needed.

        Returns:
            Dict with keys: stop_reason, response_ids, response_logprobs
        """
        url = f"{self.proxy_url}/inference/v1/generate"

        # Use LoRA adapter name if one is active, otherwise use base model name
        effective_model = self.active_lora_name if self.active_lora_name else self.model_name

        payload = {
            "sampling_params": sampling_params,
            "model": effective_model,
            "token_ids": prompt_token_ids,
        }

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        response = await self._post(url, json=payload, headers=headers)

        choice = response["choices"][0]
        token_ids = choice["token_ids"]
        stop_reason = choice["finish_reason"]

        response_logprobs: Optional[List[float]] = None
        logprobs = choice.get("logprobs")
        if logprobs is not None:
            logprobs_content = logprobs.get("content", [])
            if logprobs_content:
                response_logprobs = [logprob_info["logprob"] for logprob_info in logprobs_content]

        return {
            "stop_reason": stop_reason,
            "response_ids": token_ids,
            "response_logprobs": response_logprobs,
        }

    async def chat_completion(
        self,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Chat completion via /v1/chat/completions.

        Args:
            request_payload: Dict with {"json": <request-body>, "headers": <headers-dict>}.
                The request body should be OpenAI-compatible chat completion request.
                session_id can be included in json for consistent routing.

        Returns:
            OpenAI-compatible chat completion response.
        """
        body = request_payload.get("json", {})

        # Extract session_id for routing (same as InferenceEngineClient)
        session_id = body.pop("session_id", None)

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        url = f"{self.proxy_url}/v1/chat/completions"
        return await self._post(url, json=body, headers=headers)

    async def completion(
        self,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Completion via /v1/completions.

        Args:
            request_payload: Dict with {"json": <request-body>, "headers": <headers-dict>}.
                The request body should be OpenAI-compatible completion request.
                session_id can be included in json for consistent routing.

        Returns:
            OpenAI-compatible completion response.
        """
        body = request_payload.get("json", {})

        # Extract session_id for routing (same as InferenceEngineClient)
        session_id = body.pop("session_id", None)

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        url = f"{self.proxy_url}/v1/completions"
        return await self._post(url, json=body, headers=headers)

    async def tokenize(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Tokenize texts.

        Uses the local tokenizer if available, otherwise falls back to HTTP /tokenize.

        Args:
            texts: List of texts to tokenize.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token ID lists.
        """
        if self.tokenizer is not None:
            return self.tokenizer(texts, add_special_tokens=add_special_tokens)["input_ids"]

        url = f"{self.proxy_url}/tokenize"

        # vLLM /tokenize expects individual requests, batch them
        results = []
        for text in texts:
            payload = {
                "model": self.model_name,
                "prompt": text,
                "add_special_tokens": add_special_tokens,
            }
            result = await self._post(url, json=payload)
            results.append(result.get("tokens", []))

        return results

    async def detokenize(
        self,
        token_ids: List[List[int]],
    ) -> List[str]:
        """
        Detokenize token IDs.

        Uses the local tokenizer if available, otherwise falls back to HTTP /detokenize.

        Args:
            token_ids: List of token ID lists.

        Returns:
            List of decoded texts.
        """
        if self.tokenizer is not None:
            return self.tokenizer.batch_decode(token_ids)

        url = f"{self.proxy_url}/detokenize"

        # vLLM /detokenize expects individual requests, batch them
        results = []
        for ids in token_ids:
            payload = {
                "model": self.model_name,
                "tokens": ids,
            }
            result = await self._post(url, json=payload)
            results.append(result.get("prompt", ""))

        return results

    # ---------------------------
    # Control Plane (fan-out to all server_urls)
    # ---------------------------

    async def _call_server(
        self,
        server_url: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call endpoint on a single server.

        Args:
            server_url: Base URL of the server.
            endpoint: Endpoint path (e.g., "/pause").
            json: JSON payload to send as request body.
            method: HTTP method (default: POST).
            params: URL query parameters (e.g., for FastAPI Query() params).

        Returns:
            Tuple of (server_url, {"status": <int>, "body": <response>}).
        """
        session = await self._get_session()
        url = f"{server_url}{endpoint}"
        async with session.request(method, url, json=json, params=params) as resp:
            body = await resp.json() if resp.content_length else None
            raise_for_status(resp, body)
            return server_url, {"status": resp.status, "body": body}

    async def _call_all_servers(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call endpoint on all server_urls concurrently.

        Args:
            endpoint: Endpoint path (e.g., "/pause").
            json: JSON payload to send as request body.
            method: HTTP method (default: POST).
            params: URL query parameters (e.g., for FastAPI Query() params).

        Returns:
            Dict mapping server_url to response.
        """
        results = await asyncio.gather(
            *[self._call_server(url, endpoint, json, method, params) for url in self.server_urls]
        )
        return {url: resp for url, resp in results}

    async def pause(self, mode: Union[PauseMode, str] = PauseMode.KEEP, clear_cache: bool = False) -> Dict[str, Any]:
        """
        Pause generation on all backends.

        Args:
            mode: Pause mode determining how in-flight requests are handled.
                Can be a PauseMode enum or string ("abort", "keep", "wait").
                - KEEP / "keep": Freeze in-flight requests in the scheduler.
                    They resume where they left off on /resume. KV cache is
                    preserved. No retry needed. (default)
                - ABORT / "abort": Abort in-flight requests immediately. Clients
                    receive partial tokens and must retry with accumulated context.
                - WAIT / "wait": Wait for in-flight requests to complete before
                    pausing. New requests are blocked. No retry needed.
            clear_cache: Whether to clear the KV cache on pause. Defaults to False.

        Returns:
            Dict mapping server_url to response.
        """
        if isinstance(mode, str):
            mode = PauseMode(mode.lower())

        params: Dict[str, Any] = {"mode": mode.value, "clear_cache": str(clear_cache).lower()}

        return await self._call_all_servers("/pause", params=params)

    async def resume(self) -> Dict[str, Any]:
        """Resume generation on all backends."""
        return await self._call_all_servers("/resume")

    async def pause_generation(self, clear_cache: bool = False) -> Dict[str, Any]:
        """Pause using keep mode - compatibility with InferenceEngineClient interface."""
        return await self.pause(mode=PauseMode.KEEP, clear_cache=clear_cache)

    async def resume_generation(self) -> Dict[str, Any]:
        """Resume after pause - compatibility with InferenceEngineClient interface."""
        return await self.resume()

    async def sleep(self, level: int = 2, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Put all backends to sleep (offload weights to CPU).

        Args:
            level: Sleep level (1 or 2). Level 2 offloads more aggressively.
            tags: Optional list of tags to sleep specific resources.
                Common tags: ["weights"], ["kv_cache"], or None for all.

        Returns:
            Dict mapping server_url to response.
        """
        body = {"level": level}
        if tags:
            body["tags"] = tags
        return await self._call_all_servers("/sleep", body)

    async def wake_up(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Wake up all backends (load weights back to GPU).

        Args:
            tags: Optional list of tags to wake up specific resources.
                Common tags: ["weights"], ["kv_cache"], or None for all.
        """
        body = {"tags": tags} if tags else {}
        return await self._call_all_servers("/wake_up", body)

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
    ) -> Dict[str, Any]:
        """
        Reset KV cache on all backends.

        Args:
            reset_running_requests: Whether to reset running requests.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers("/reset_prefix_cache", {"reset_running_requests": reset_running_requests})

    # ---------------------------
    # Weight Sync (control plane - fan-out)
    # ---------------------------

    async def init_weight_update_communicator(
        self,
        init_info: "WeightSyncInitInfo",
    ) -> Dict[str, Any]:
        """
        Initialize weight sync via vLLM native /init_weight_transfer_engine.

        Fetches per-server world sizes, expands init_info into per-server
        payloads (with correct NCCL rank offsets), and fans out to all servers.

        Args:
            init_info: A WeightSyncInitInfo (e.g. BroadcastInitInfo) that supports
                for_servers() and to_api_payload().

        Returns:
            Dict mapping server_url to response.
        """
        _, world_size_per_server = await self.get_world_size()
        num_servers = len(self.server_urls)
        server_infos = init_info.for_servers(world_size_per_server, num_servers)
        payloads = [{"init_info": x.to_api_payload()} for x in server_infos]
        results = await asyncio.gather(
            *[
                self._call_server(url, "/init_weight_transfer_engine", payload)
                for url, payload in zip(self.server_urls, payloads)
            ]
        )
        return {url: resp for url, resp in results}

    async def update_named_weights(
        self,
        update_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update model weights via vLLM native /update_weights. Used for full parameter fine-tuning.

        For LoRA weight sync, use update_lora_from_disk() instead.

        Args:
            update_info: Dict with keys expected by vLLM (names, dtype_names, shapes, packed, etc.)

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/update_weights",
            {"update_info": update_info},
        )

    async def update_lora_from_disk(
        self,
        lora_path: str,
    ) -> Dict[str, Any]:
        """
        Update LoRA adapter weights by loading from disk on all backend servers via /v1/load_lora_adapter.

        Always loads under self.active_lora_name so the same slot is reused across
        weight syncs.

        After loading, generation requests will automatically use the LoRA adapter
        by setting the model name to the LoRA adapter name.

        Args:
            lora_path: Path to the LoRA adapter on disk (must be accessible from servers).

        Returns:
            Dict mapping server_url to response.
        """
        if self.active_lora_name is None:
            raise ValueError("active_lora_name must be set on RemoteInferenceClient before loading a LoRA adapter.")

        lora_name = self.active_lora_name
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
            "load_inplace": True,
        }

        # Call /v1/load_lora_adapter on all servers directly.
        # This endpoint returns a plain text response (not JSON), so we use a
        # custom call instead of _call_all_servers which expects JSON.
        session = await self._get_session()

        async def _load_on_server(server_url: str):
            url = f"{server_url}/v1/load_lora_adapter"
            async with session.post(url, json=payload) as resp:
                # vLLM returns 200 with text body on success, or JSON ErrorResponse on failure
                if resp.status >= 400:
                    body = await resp.json()
                    raise_for_status(resp, body)
                return server_url, {"status": resp.status, "body": await resp.text()}

        results = await asyncio.gather(*[_load_on_server(url) for url in self.server_urls])

        logger.info(f"Loaded LoRA adapter '{lora_name}' from {lora_path}")

        return {url: resp for url, resp in results}

    # ---------------------------
    # Info
    # ---------------------------

    async def get_world_size(self) -> Tuple[int, int]:
        """
        Get total and per-server world size across all inference workers.

        Fetches from vLLM's /get_world_size endpoint on each server.
        All servers are expected to have the same world size.
        Result is cached after first call.

        Returns:
            Tuple of (total_world_size, world_size_per_server).
        """
        if self._world_size is not None:
            return self._world_size

        results = await self._call_all_servers("/get_world_size", {}, method="GET")

        per_server = []
        for server_url in self.server_urls:
            resp = results.get(server_url)
            if resp is None:
                raise RuntimeError(f"No response for server {server_url}")
            body = resp.get("body", {})
            world_size = body.get("world_size")
            if world_size is None:
                raise RuntimeError(f"Missing world_size in response from {server_url}")
            per_server.append(world_size)

        assert all(
            ws == per_server[0] for ws in per_server
        ), f"All servers must have the same world_size, got {per_server}"

        self._world_size = (per_server[0] * len(self.server_urls), per_server[0])
        return self._world_size

    # ---------------------------
    # Lifecycle
    # ---------------------------

    async def teardown(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "RemoteInferenceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.teardown()

    # ---------------------------
    # Serialization
    # ---------------------------

    def __getstate__(self) -> dict:
        """Exclude non-serializable fields from pickle."""
        state = self.__dict__.copy()
        state["_session"] = None
        state["_gen_sem"] = None
        state["_detok_sem"] = None
        state["_sem_loop"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._session = None
        self._gen_sem = None
        self._detok_sem = None
        self._sem_loop = None


def _force_close_connector(connector: Optional[aiohttp.TCPConnector]) -> None:
    """Release socket FDs held by a connector whose event loop is already closed.

    asyncio's normal transport.close() schedules a connection_lost() callback
    on the loop, which raises RuntimeError when the loop is closed. We instead
    close the raw file descriptors via os.close(fileno()) and then manually
    clear aiohttp's internal connection pools.

    Note: this uses aiohttp internals (_conns, _acquired, _closed). These are
    stable across aiohttp 3.x but may change in 4.x.
    """
    if connector is None or connector.closed:
        return

    def _release(proto: aiohttp.client_proto.ResponseHandler) -> None:
        if proto.transport is not None:
            tsock = proto.transport.get_extra_info("socket")
            if tsock is not None:
                fd = tsock.fileno()
                if fd != -1:
                    try:
                        tsock.close()
                    except OSError:
                        pass

    for proto_list in connector._conns.values():
        for proto, _ in proto_list:
            _release(proto)
    for proto in list(connector._acquired):
        _release(proto)

    connector._conns.clear()
    connector._acquired.clear()
    connector._closed = True


def raise_for_status(resp: aiohttp.ClientResponse, body: Optional[Any] = None) -> None:
    """Modified version of resp.raise_for_status() that reads the body for the error message.

    Raises aiohttp.ClientResponseError with the error message from the body if there is an error

    The standard `raise_for_status()` only uses the HTTP reason phrase (e.g. "Bad Request"), which is often unhelpful. APIs typically put more descriptive error details in the response body. This function bridges that gap by surfacing the body's error message in the exception.
    """
    if resp.status >= 400 and body is not None:
        error_detail = body.get("error", {})
        detail_msg = error_detail.get("message", resp.reason) if isinstance(error_detail, dict) else resp.reason
        raise aiohttp.ClientResponseError(
            resp.request_info,
            resp.history,
            status=resp.status,
            message=detail_msg,
            headers=resp.headers,
        )
    resp.raise_for_status()
