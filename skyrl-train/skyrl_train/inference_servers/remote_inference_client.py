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
   - init_weight_transfer, update_weights, finalize_weight_update
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
- Built-in retry on abort for in-flight weight updates (temporary)

Usage:
    client = RemoteInferenceClient(
        proxy_url="http://router:8080",  # Data plane (router)
        server_urls=["http://backend1:8000", "http://backend2:8000"],  # Control plane
    )

Comparison with existing code:
- Replaces: InferenceEngineClient + RemoteInferenceEngine (for remote-only usage)
- Key difference: Talks directly to router via HTTP, no Ray actor wrapping
- The router handles session-aware routing; this client handles control plane fan-out

TODO: Data Plane Operations - Future Deprecation
------------------------------------------------
All data plane operations (generate, chat_completion, completion, tokenize, detokenize)
and the retry-on-abort logic will eventually be removed from this client.

When vLLM RFC #32103 lands with PauseMode.KEEP:
- The retry logic in generate() will be deleted
- pause() will use mode="keep" which preserves KV cache and scheduler state
- Requests resume seamlessly after unpause with zero client changes

The generator code will transition to:
1. OpenAI-compatible endpoints (/v1/chat/completions) for text-based interaction
2. Tinker sample API for token-in-token-out workflows:
   - Input: ModelInput.from_ints(tokens=input_ids)
   - Output: sequences[0].tokens, sequences[0].logprobs
   - Internally maps to /v1/completions with token-in-token-out
   - May become a native vLLM API in the future

This client will then primarily handle control plane operations only.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import asdict


import aiohttp

from skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput

if TYPE_CHECKING:
    from skyrl_train.weight_sync import BroadcastInitInfo, WeightUpdateRequest

logger = logging.getLogger(__name__)


class PauseMode(Enum):
    """
    Pause mode for inference servers.

    This enum mirrors the pause modes that will be available in vLLM RFC #32103.
    For now, we map these to the existing `wait_for_inflight_request` parameter.

    Modes:
        ABORT: Abort in-flight requests immediately. Clients receive partial
            tokens and must retry with accumulated context.
            Maps to: wait_for_inflight_request=False

        FINISH: Wait for in-flight requests to complete before pausing.
            New requests are blocked. No retry needed.
            Maps to: wait_for_inflight_request=True
    """

    ABORT = "abort"
    FINISH = "finish"


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

    # Private fields excluded from repr for cleaner output
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)
    _world_size: Optional[int] = field(default=None, repr=False)

    # ---------------------------
    # Session Management
    # ---------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        # Re-use the existing session object if it is not closed.
        # Note that we also create a new session object if the event loop has changed, since
        # aiohttp.ClientSession is tied to the event loop.
        current_loop = asyncio.get_running_loop()
        if self._session is None or self._session.closed or self._session.loop != current_loop:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        return self._session

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

        # Create parallel tasks for all prompts
        # Each task handles its own retry on abort
        tasks = [
            self._generate_single(
                prompt_token_ids=prompt_token_ids[idx],
                sampling_params=sampling_params,
                session_id=session_ids[idx] if session_ids and idx < len(session_ids) else None,
            )
            for idx in range(len(prompt_token_ids))
        ]

        # Run all in parallel - retries happen within each task
        results = await asyncio.gather(*tasks)

        return InferenceEngineOutput(
            responses=[r["response"] for r in results],
            stop_reasons=[r["stop_reason"] for r in results],
            response_ids=[r["response_ids"] for r in results],
            response_logprobs=None,
        )

    # TODO: Delete retry logic when vLLM RFC #32103 lands with PauseMode.KEEP
    async def _generate_single(
        self,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
        session_id: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Generate completion for a single prompt with built-in retry on abort.

        When pause(mode=ABORT) is called, running requests return partial tokens
        with stop_reason="abort". This method retries with accumulated tokens
        until generation completes with a non-abort stop reason.

        TODO: Retry logic will be deleted when vLLM RFC #32103 lands.
        With PauseMode.KEEP, requests resume seamlessly after unpause.

        Returns:
            Dict with keys: response, stop_reason, response_ids
        """
        session = await self._get_session()
        url = f"{self.proxy_url}/v1/completions"

        # Determine max_tokens key and original value
        max_key = None
        if "max_tokens" in sampling_params:
            max_key = "max_tokens"
        elif "max_completion_tokens" in sampling_params:
            max_key = "max_completion_tokens"
        original_max_tokens = sampling_params.get(max_key) if max_key else None

        # Accumulate across retries
        accum_text = ""
        accum_token_ids: List[int] = []
        stop_reason = "abort"

        while stop_reason == "abort":
            # Build payload with accumulated context
            cur_params = sampling_params.copy()
            if original_max_tokens is not None and max_key:
                remaining = original_max_tokens - len(accum_token_ids)
                if remaining <= 0:
                    break
                cur_params[max_key] = remaining

            # New prompt = original + accumulated tokens
            new_prompt = prompt_token_ids + accum_token_ids

            payload = cur_params.copy()
            payload["model"] = self.model_name
            payload["prompt"] = new_prompt

            headers = {"Content-Type": "application/json"}
            if session_id:
                headers["X-Session-ID"] = str(session_id)

            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                response = await resp.json()

            choice = response["choices"][0]
            new_text = choice["text"]
            stop_reason = choice["finish_reason"]

            # Accumulate text
            accum_text += new_text
            # Tokenize the new text to get token IDs for next iteration
            if stop_reason == "abort" and new_text:
                new_token_ids = (await self.tokenize([new_text], add_special_tokens=False))[0]
                accum_token_ids.extend(new_token_ids)

        # Final response
        # Tokenize full accumulated text for response_ids
        final_token_ids = (await self.tokenize([accum_text], add_special_tokens=False))[0] if accum_text else []

        return {
            "response": accum_text,
            "stop_reason": stop_reason,
            "response_ids": final_token_ids,
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

        session = await self._get_session()
        url = f"{self.proxy_url}/v1/chat/completions"

        async with session.post(url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

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

        session = await self._get_session()
        url = f"{self.proxy_url}/v1/completions"

        async with session.post(url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def tokenize(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Tokenize texts via /tokenize.

        Args:
            texts: List of texts to tokenize.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token ID lists.
        """
        session = await self._get_session()
        url = f"{self.proxy_url}/tokenize"

        # vLLM /tokenize expects individual requests, batch them
        results = []
        for text in texts:
            payload = {
                "model": self.model_name,
                "prompt": text,
                "add_special_tokens": add_special_tokens,
            }
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                results.append(result.get("tokens", []))

        return results

    async def detokenize(
        self,
        token_ids: List[List[int]],
    ) -> List[str]:
        """
        Detokenize token IDs via /detokenize.

        Args:
            token_ids: List of token ID lists.

        Returns:
            List of decoded texts.
        """
        session = await self._get_session()
        url = f"{self.proxy_url}/detokenize"

        # vLLM /detokenize expects individual requests, batch them
        results = []
        for ids in token_ids:
            payload = {
                "model": self.model_name,
                "tokens": ids,
            }
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                results.append(result.get("prompt", ""))

        return results

    # ---------------------------
    # Control Plane (fan-out to all server_urls)
    # ---------------------------

    async def _call_all_servers(
        self,
        endpoint: str,
        json: Dict[str, Any],
        method: str = "POST",
    ) -> Dict[str, Any]:
        """
        Call endpoint on all server_urls concurrently.

        Args:
            endpoint: Endpoint path (e.g., "/pause").
            json: JSON payload to send.
            method: HTTP method (default: POST).

        Returns:
            Dict mapping server_url to response.
        """
        session = await self._get_session()

        async def call_server(server_url: str) -> tuple:
            url = f"{server_url}{endpoint}"
            async with session.request(method, url, json=json) as resp:
                resp.raise_for_status()
                body = await resp.json() if resp.content_length else None
                return server_url, {"status": resp.status, "body": body}

        results = await asyncio.gather(*[call_server(url) for url in self.server_urls])
        return {url: resp for url, resp in results}

    async def pause(self, mode: Union[PauseMode, str] = PauseMode.ABORT) -> Dict[str, Any]:
        """
        Pause generation on all backends.

        Args:
            mode: Pause mode determining how in-flight requests are handled.
                Can be a PauseMode enum or string ("abort", "finish").
                - ABORT / "abort": Abort in-flight requests immediately. Clients
                    receive partial tokens and must retry with accumulated context.
                    New requests are blocked.
                - FINISH / "finish": Wait for in-flight requests to complete before
                    pausing. New requests are blocked. No retry needed.

        Returns:
            Dict mapping server_url to response.

        TODO:
            When vLLM RFC #32103 lands, we'll use the native mode parameter.
            For now, we map modes to wait_for_inflight_request:
            - ABORT → wait_for_inflight_request=False
            - FINISH → wait_for_inflight_request=True
        """
        # Convert string to PauseMode if needed
        if isinstance(mode, str):
            mode = PauseMode(mode.lower())

        wait_for_inflight_request = mode == PauseMode.FINISH

        return await self._call_all_servers("/pause", {"wait_for_inflight_request": wait_for_inflight_request})

    async def resume(self) -> Dict[str, Any]:
        """Resume generation on all backends."""
        return await self._call_all_servers("/resume", {})

    # TODO (Kourosh): Compatibility aliases for InferenceEngineClient interface, delete this when we deprecate the old interface
    async def pause_generation(self) -> Dict[str, Any]:
        """Alias for pause() - compatibility with InferenceEngineClient interface."""
        return await self.pause(mode=PauseMode.ABORT)

    async def resume_generation(self) -> Dict[str, Any]:
        """Alias for resume() - compatibility with InferenceEngineClient interface."""
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
        init_info: "BroadcastInitInfo",
    ) -> Dict[str, Any]:
        """
        Initialize weight sync process group on all backends.

        Args:
            init_info: BroadcastInitInfo containing all args for weight sync setup.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers("/init_weight_transfer", asdict(init_info))

    async def update_named_weights(
        self,
        request: "WeightUpdateRequest",
    ) -> Dict[str, Any]:
        """
        Update weights on all backends.

        Args:
            request: BroadcastWeightUpdateRequest containing weight metadata.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers("/update_weights", request.to_json_dict())

    async def finalize_weight_update(self) -> Dict[str, Any]:
        """
        Finalize weight update on all backends.

        Called after all update_weights() calls are complete.
        Reserved for any post-processing steps that may be needed:
        - Cache invalidation
        - State synchronization
        - Future vLLM requirements

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers("/finalize_weight_update", {})

    # ---------------------------
    # Info
    # ---------------------------

    async def get_world_size(self) -> int:
        """
        Get total world size across all inference workers.

        Fetches from /get_server_info on each server and sums the world_size values.
        Result is cached after first call.
        """
        if self._world_size is not None:
            return self._world_size

        results = await self._call_all_servers("/get_server_info", {}, method="GET")

        total_world_size = 0
        for server_url, resp in results.items():
            if resp.get("status") != 200:
                error = resp.get("error", resp.get("body"))
                raise RuntimeError(f"Failed to fetch server info from {server_url}: {error}")
            body = resp.get("body", {})
            world_size = body.get("world_size")
            if world_size is None:
                raise RuntimeError(f"Failed to fetch server info from {server_url}: world_size is missing")
            total_world_size += world_size

        self._world_size = total_world_size
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
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._session = None
