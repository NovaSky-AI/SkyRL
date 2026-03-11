"""Pure helper functions for the /sample endpoint.

Kept in a separate module (no vLLM import) so they can be unit-tested on CPU
without a GPU environment.
"""

import aiohttp


async def _render_image(session: aiohttp.ClientSession, base_url: str, b64_data: str, fmt: str, model: str) -> list:
    """Call /v1/chat/completions/render with a single bare image message."""
    data_uri = f"data:image/{fmt};base64,{b64_data}"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]}]
    async with session.post(
        f"{base_url}/v1/chat/completions/render",
        json={"model": model, "messages": messages},
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _render_image_url(session: aiohttp.ClientSession, base_url: str, location: str, model: str) -> list:
    """Call /v1/chat/completions/render with an image URL."""
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": location}}]}]
    async with session.post(
        f"{base_url}/v1/chat/completions/render",
        json={"model": model, "messages": messages},
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def _extract_render_info(render_resp: list) -> tuple[list[int], str]:
    """Extract placeholder token IDs and mm_hash from a render response.

    The render endpoint returns ``[conversation, engine_prompts]``.  Each
    engine prompt carries a ``features`` list of GenerateMultiModalFeature
    dicts with ``offset``, ``length``, and ``mm_hash``.

    Returns:
        (placeholder_tokens, mm_hash) where placeholder_tokens are the token IDs
        for the image placeholder region in the rendered sequence.
    """
    ep = render_resp[1][0]
    token_ids = ep["prompt_token_ids"]
    feature = ep["features"][0]
    offset, length = feature["offset"], feature["length"]
    placeholder_tokens = token_ids[offset : offset + length]
    mm_hash = feature["mm_hash"]
    return placeholder_tokens, mm_hash


async def _assemble_tokens_from_chunks(
    chunks: list[dict],
    base_url: str,
    model: str,
) -> tuple[list[int], list[dict] | None]:
    """Assemble a flat token_ids list and optional features list from ModelInput chunks.

    Handles EncodedTextChunk (plain tokens), ImageChunk (render via data URI),
    and ImageAssetPointerChunk (render via URL).

    Returns:
        (assembled_tokens, features) where features is None if no images were present,
        or a list of GenerateMultiModalFeature-shaped dicts otherwise.
    """
    has_images = any(c["type"] in ("image", "image_asset_pointer") for c in chunks)

    if not has_images:
        token_ids = [t for c in chunks if c["type"] == "encoded_text" for t in c["tokens"]]
        return token_ids, None

    assembled_tokens: list[int] = []
    features: list[dict] = []

    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            if chunk["type"] == "encoded_text":
                assembled_tokens.extend(chunk["tokens"])

            elif chunk["type"] == "image":
                b64 = chunk["data"]  # already a base64 string in JSON
                fmt = chunk["format"]
                render_resp = await _render_image(session, base_url, b64, fmt, model)
                placeholder_tokens, mm_hash = _extract_render_info(render_resp)

                expected = chunk.get("expected_tokens")
                if expected is not None and expected != len(placeholder_tokens):
                    raise ValueError(
                        f"ImageChunk.expected_tokens={expected} but render returned "
                        f"{len(placeholder_tokens)} placeholder tokens (mm_hash={mm_hash})"
                    )

                features.append(
                    {
                        "modality": "image",
                        "mm_hash": mm_hash,
                        "offset": len(assembled_tokens),
                        "length": len(placeholder_tokens),
                        "kwargs_data": None,
                    }
                )
                assembled_tokens.extend(placeholder_tokens)

            elif chunk["type"] == "image_asset_pointer":
                location = chunk["location"]
                render_resp = await _render_image_url(session, base_url, location, model)
                placeholder_tokens, mm_hash = _extract_render_info(render_resp)

                expected = chunk.get("expected_tokens")
                if expected is not None and expected != len(placeholder_tokens):
                    raise ValueError(
                        f"ImageAssetPointerChunk.expected_tokens={expected} but render returned "
                        f"{len(placeholder_tokens)} placeholder tokens (mm_hash={mm_hash})"
                    )

                features.append(
                    {
                        "modality": "image",
                        "mm_hash": mm_hash,
                        "offset": len(assembled_tokens),
                        "length": len(placeholder_tokens),
                        "kwargs_data": None,
                    }
                )
                assembled_tokens.extend(placeholder_tokens)

    return assembled_tokens, features if features else None
