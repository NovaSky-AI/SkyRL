from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, NamedTuple, Union

import torch

from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
    MultiModalKwargs,
    MultiModalPlaceholder,
    RenderedModelInput,
)

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )


def render_model_input(model_inputs: list[ModelInput]) -> list[RenderedModelInput]:
    """Text-only renderer. Concatenates token chunks."""
    return [
        RenderedModelInput(
            prompt_ids=[tok for chunk in mi.chunks for tok in (chunk.tokens if hasattr(chunk, "tokens") else [])]
        )
        for mi in model_inputs
    ]


def decode_mm_kwargs(mm_kwargs: dict[str, list[str]] | None) -> MultiModalKwargs:
    """Decode raw base64-encoded multimodal kwargs from vLLM into vision arrays.

    Args:
        mm_kwargs: Raw kwargs dict from vLLM's render endpoint, e.g. {"image": [b64_str, ...]}.

    Returns:
        MultiModalKwargs with decoded and concatenated pixel_values and image_grid_thw.
        Values are None when no vision data is present.
    """
    if not mm_kwargs or "image" not in mm_kwargs:
        return MultiModalKwargs(pixel_values=None, image_grid_thw=None)

    from vllm.entrypoints.serve.disagg.mm_serde import (
        decode_mm_kwargs_item as _vllm_decode,
    )

    pv_parts: list[torch.Tensor] = []
    thw_parts: list[torch.Tensor] = []
    for b64_str in mm_kwargs["image"]:
        item = _vllm_decode(b64_str)
        data = item.get_data()
        if "pixel_values" in data and isinstance(data["pixel_values"], torch.Tensor):
            pv_parts.append(data["pixel_values"])
        if "image_grid_thw" in data and isinstance(data["image_grid_thw"], torch.Tensor):
            thw_parts.append(data["image_grid_thw"])

    pixel_values = torch.cat(pv_parts, dim=0) if pv_parts else None
    thw_parts = [t.reshape(1, -1) if t.dim() == 1 else t for t in thw_parts]
    image_grid_thw = torch.cat(thw_parts, dim=0) if thw_parts else None
    return MultiModalKwargs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)


class RenderedImage(NamedTuple):
    """Per-image result from _render_images."""

    placeholder_tokens: list[int]
    kwargs_data: str | None


class VLLMRenderer:
    """Renders ModelInputs by calling vLLM's /v1/chat/completions/render for image placeholders.

    For text-only inputs, no HTTP call is made.
    For multi-modal inputs, images are sent to the render endpoint to obtain
    placeholder tokens and optional kwargs_data (serialized pixel_values, etc).
    """

    def __init__(self, client: RemoteInferenceClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    async def __call__(self, model_inputs: list[ModelInput]) -> list[RenderedModelInput]:
        return list(await asyncio.gather(*[self._render_single(mi) for mi in model_inputs]))

    # -- internal -------------------------------------------------------------

    async def _render_single(self, model_input: ModelInput) -> RenderedModelInput:
        image_chunks = [c for c in model_input.chunks if isinstance(c, (ImageChunk, ImageAssetPointerChunk))]

        if not image_chunks:
            return render_model_input([model_input])[0]

        rendered_images = await self._render_images(image_chunks)

        # Assemble final token stream: walk chunks in order, splice placeholder tokens
        token_ids: list[int] = []
        placeholders: list[MultiModalPlaceholder] = []
        image_idx = 0
        for chunk in model_input.chunks:
            if isinstance(chunk, EncodedTextChunk):
                token_ids.extend(chunk.tokens)
            elif isinstance(chunk, (ImageChunk, ImageAssetPointerChunk)):
                ri = rendered_images[image_idx]
                offset = len(token_ids)
                token_ids.extend(ri.placeholder_tokens)
                placeholders.append(MultiModalPlaceholder(offset=offset, length=len(ri.placeholder_tokens)))
                image_idx += 1

        kwargs_data_items = [ri.kwargs_data for ri in rendered_images if ri.kwargs_data is not None]
        mm_kwargs_raw = {"image": kwargs_data_items} if kwargs_data_items else None

        return RenderedModelInput(
            prompt_ids=token_ids,
            multi_modal_placeholders=placeholders if placeholders else None,
            multi_modal_kwargs=decode_mm_kwargs(mm_kwargs_raw),
        )

    async def _render_images(
        self,
        image_chunks: list[Union[ImageChunk, ImageAssetPointerChunk]],
    ) -> list[RenderedImage]:
        # Converts image chunks to OpenAI chat-completions content format (image_url parts),
        # sends them to vLLM's /v1/chat/completions/render endpoint, and extracts per-image
        # placeholder tokens and serialized multimodal kwargs from vLLM's response.
        content_parts = []
        for chunk in image_chunks:
            if isinstance(chunk, ImageChunk):
                b64_data = base64.b64encode(chunk.data).decode("ascii")
                url = f"data:image/{chunk.format};base64,{b64_data}"
            else:  # ImageAssetPointerChunk
                url = chunk.location
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        payload = {
            "json": {
                "model": self._model_name,
                "messages": [{"role": "user", "content": content_parts}],
            }
        }

        response = await self._client.render_chat_completion(payload)

        token_ids = response["token_ids"]
        features = response.get("features") or {}
        image_placeholders = features.get("mm_placeholders", {}).get("image", [])
        image_kwargs = (features.get("kwargs_data") or {}).get("image", [])

        if len(image_placeholders) != len(image_chunks):
            raise RuntimeError(f"Expected {len(image_chunks)} image placeholders, got {len(image_placeholders)}")

        rendered: list[RenderedImage] = []
        for i, placeholder in enumerate(image_placeholders):
            offset = placeholder["offset"]
            length = placeholder["length"]
            tokens = token_ids[offset : offset + length]

            chunk = image_chunks[i]
            if chunk.expected_tokens is not None and chunk.expected_tokens != length:
                # Tinker semantics raise an error if expected chunks is incorect.
                raise ValueError(
                    f"Image {i}: expected_tokens={chunk.expected_tokens} but render returned {length} placeholder tokens"
                )

            rendered.append(
                RenderedImage(
                    placeholder_tokens=tokens,
                    kwargs_data=image_kwargs[i] if i < len(image_kwargs) else None,
                )
            )

        return rendered
