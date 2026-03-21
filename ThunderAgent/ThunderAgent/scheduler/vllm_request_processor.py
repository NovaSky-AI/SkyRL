"""vLLM request processing utilities.

Handles HTTP communication with vLLM backends, including:
- Streaming and non-streaming request forwarding
- SSE (Server-Sent Events) parsing
- Usage info extraction from responses
"""
import json
import math
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import httpx
from fastapi.responses import Response, StreamingResponse


def extract_usage_info(payload: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract usage info from a vLLM response.
    
    Returns:
        (total_tokens, prompt_tokens, cached_tokens)
        - cached_tokens is 0 if prompt_tokens_details is null or missing
        - All None if usage is not available
    """
    if not isinstance(payload, dict):
        return None, None, None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    
    total_tokens = None
    prompt_tokens = None
    cached_tokens = 0  # Default to 0 if not available
    
    if "total_tokens" in usage:
        val = usage.get("total_tokens")
        if isinstance(val, (int, float)) and math.isfinite(val):
            total_tokens = int(val)
    
    if "prompt_tokens" in usage:
        val = usage.get("prompt_tokens")
        if isinstance(val, (int, float)) and math.isfinite(val):
            prompt_tokens = int(val)
    
    # Extract cached_tokens from prompt_tokens_details
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        ct = prompt_details.get("cached_tokens")
        if isinstance(ct, (int, float)) and math.isfinite(ct):
            cached_tokens = int(ct)
    
    return total_tokens, prompt_tokens, cached_tokens


def filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
    """Filter out hop-by-hop headers that shouldn't be forwarded."""
    hop_by_hop = {"content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


def remove_program_id(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove program_id from payload before forwarding to vLLM.
    
    vLLM doesn't recognize program_id, so we need to strip it.
    """
    payload = payload.copy()
    payload.pop("program_id", None)
    if "extra_body" in payload and isinstance(payload["extra_body"], dict):
        payload["extra_body"] = payload["extra_body"].copy()
        payload["extra_body"].pop("program_id", None)
        if not payload["extra_body"]:
            del payload["extra_body"]
    return payload


# Default interval for token progress updates during streaming
DEFAULT_TOKEN_PROGRESS_INTERVAL = 20


async def forward_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    *,
    on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
    on_first_token: Callable[[], None] | None = None,
    on_token: Callable[[], None] | None = None,
    on_token_progress: Callable[[int], None] | None = None,
    token_progress_interval: int = DEFAULT_TOKEN_PROGRESS_INTERVAL,
) -> StreamingResponse:
    """Forward a streaming request to vLLM and return a StreamingResponse.
    
    This function handles:
    - SSE (Server-Sent Events) parsing
    - Usage info extraction from the stream
    - Token timing callbacks (on_first_token, on_token)
    - Token progress updates at regular intervals (on_token_progress)
    - Calling on_usage when stream ends
    
    Args:
        client: httpx AsyncClient to use
        url: vLLM endpoint URL
        payload: Request payload (program_id will be removed)
        on_usage: Called with (total_tokens, prompt_tokens, cached_tokens) when stream ends
        on_first_token: Called when first token is received
        on_token: Called for each token received
        on_token_progress: Called with cumulative token count at regular intervals
        token_progress_interval: How often to call on_token_progress (default: 20 tokens)
    
    Returns:
        FastAPI StreamingResponse that forwards the vLLM stream to client
    """
    # Remove program_id before forwarding
    payload = remove_program_id(payload)
    
    # Add stream_options to get usage info in streaming response
    if on_usage is not None:
        stream_options = payload.get("stream_options")
        if stream_options is None:
            payload["stream_options"] = {"include_usage": True}
        elif isinstance(stream_options, dict):
            stream_options.setdefault("include_usage", True)

    resp_cm = client.stream("POST", url, json=payload)
    resp = await resp_cm.__aenter__()
    headers = filtered_headers(resp.headers)
    status = resp.status_code
    media_type = resp.headers.get("content-type")

    async def iterator():
        buffer = b""
        usage_extracted = False
        total_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        cached_tokens: int = 0
        first_token_seen = False
        token_count = 0  # Track cumulative generated tokens
        last_reported_count = 0  # Last count reported via on_token_progress
        try:
            async for chunk in resp.aiter_raw():
                buffer += chunk
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        
                        # Token timing callbacks
                        if not first_token_seen:
                            first_token_seen = True
                            if on_first_token is not None:
                                on_first_token()
                        if on_token is not None:
                            on_token()
                        
                        # Track token count and report progress at intervals
                        token_count += 1
                        if on_token_progress is not None:
                            if token_count - last_reported_count >= token_progress_interval:
                                # Report the delta (new tokens since last report)
                                delta = token_count - last_reported_count
                                on_token_progress(delta)
                                last_reported_count = token_count
                        
                        # Extract usage info (only once)
                        if usage_extracted:
                            continue
                        try:
                            payload_obj = json.loads(data)
                        except Exception:
                            continue
                        tt, pt, ct = extract_usage_info(payload_obj)
                        if tt is not None:
                            total_tokens = tt
                            prompt_tokens = pt
                            cached_tokens = ct
                            usage_extracted = True
                yield chunk
        finally:
            await resp_cm.__aexit__(None, None, None)
            if total_tokens is not None and on_usage is not None:
                await on_usage(total_tokens, prompt_tokens or 0, cached_tokens)

    return StreamingResponse(
        iterator(),
        status_code=status,
        headers=headers,
        media_type=media_type,
    )


async def forward_non_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    *,
    on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
) -> Response:
    """Forward a non-streaming request to vLLM and return a Response.
    
    Args:
        client: httpx AsyncClient to use
        url: vLLM endpoint URL
        payload: Request payload (program_id will be removed)
        on_usage: Called with (total_tokens, prompt_tokens, cached_tokens) after response
    
    Returns:
        FastAPI Response with vLLM response content
    """
    # Remove program_id before forwarding
    payload = remove_program_id(payload)
    
    resp = await client.post(url, json=payload)
    
    # Extract usage info
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cached_tokens: int = 0
    try:
        payload_obj = resp.json()
    except Exception:
        payload_obj = None
    tt, pt, ct = extract_usage_info(payload_obj)
    if tt is not None:
        total_tokens = tt
        prompt_tokens = pt
        cached_tokens = ct
    
    # Call usage callback
    if total_tokens is not None and on_usage is not None:
        await on_usage(total_tokens, prompt_tokens or 0, cached_tokens)
    
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=filtered_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def forward_get_request(client: httpx.AsyncClient, url: str) -> Response:
    """Forward a GET request to vLLM backend.
    
    Args:
        client: httpx AsyncClient to use
        url: Full URL to request
    
    Returns:
        FastAPI Response with vLLM response content
    """
    resp = await client.get(url)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=filtered_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )

