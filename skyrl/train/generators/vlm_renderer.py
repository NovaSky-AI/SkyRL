"""Chat rendering through vLLM's ``/v1/chat/completions/render`` for vision-language rollouts.

``SkyRLGymGenerator`` uses ``VLLMChatRenderer`` when ``generator.vision_language_generator=True``.
The number of image placeholder tokens is only known after the model's image processor runs, so
the initial prompt and every observation are rendered by the inference server rather than by the
local tokenizer. The rollout stays token-in-token-out: observations are rendered after a fixed base
conversation and only the suffix is appended, so earlier turns are never re-rendered.
"""

import asyncio
import copy
from typing import Any, Dict, List, NamedTuple, Optional

from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    MultiModalFeatures,
)


class RenderedTokens(NamedTuple):
    token_ids: List[int]
    # Placeholder offsets are relative to ``token_ids``.
    features: Optional[MultiModalFeatures]


def _num_placeholders(features: Optional[MultiModalFeatures]) -> int:
    if not features:
        return 0
    return sum(len(ranges) for ranges in (features.get("mm_placeholders") or {}).values())


def _trim_after_last_eos(token_ids: List[int], eos_token_id: Optional[int]) -> List[int]:
    if eos_token_id is None or eos_token_id not in token_ids:
        return token_ids
    last_eos_index = len(token_ids) - 1 - token_ids[::-1].index(eos_token_id)
    return token_ids[: last_eos_index + 1]


def shift_mm_features(features: MultiModalFeatures, delta: int) -> MultiModalFeatures:
    """Return a copy of ``features`` with every placeholder offset moved by ``delta``."""
    return MultiModalFeatures(
        mm_hashes={modality: list(hashes) for modality, hashes in (features.get("mm_hashes") or {}).items()},
        mm_placeholders={
            modality: [{"offset": r["offset"] + delta, "length": r["length"]} for r in ranges]
            for modality, ranges in (features.get("mm_placeholders") or {}).items()
        },
        kwargs_data=(
            {modality: list(items) for modality, items in features["kwargs_data"].items()}
            if features.get("kwargs_data") is not None
            else None
        ),
    )


def append_mm_features(
    accumulated: Optional[MultiModalFeatures], new: Optional[MultiModalFeatures]
) -> Optional[MultiModalFeatures]:
    """Append ``new``'s items after ``accumulated``'s, per modality, preserving order.

    Both arguments must carry offsets in the same (trajectory-level) coordinate system.
    """
    if not _num_placeholders(new):
        return accumulated
    if accumulated is None:
        return shift_mm_features(new, 0)
    merged = shift_mm_features(accumulated, 0)
    for field in ("mm_hashes", "mm_placeholders"):
        for modality, items in (new.get(field) or {}).items():
            merged[field].setdefault(modality, []).extend(copy.deepcopy(items))
    if new.get("kwargs_data") is not None:
        if merged["kwargs_data"] is None:
            merged["kwargs_data"] = {}
        for modality, items in new["kwargs_data"].items():
            merged["kwargs_data"].setdefault(modality, []).extend(items)
    return merged


def truncate_mm_features(features: Optional[MultiModalFeatures], num_tokens: int) -> Optional[MultiModalFeatures]:
    """Keep the items whose placeholders lie inside the first ``num_tokens`` tokens.

    Raises:
        ValueError: if a placeholder straddles ``num_tokens``.
    """
    if not _num_placeholders(features):
        return None
    kept = MultiModalFeatures(mm_hashes={}, mm_placeholders={}, kwargs_data=None)
    kwargs_data = features.get("kwargs_data")
    for modality, ranges in features["mm_placeholders"].items():
        keep_count = 0
        for r in ranges:
            if r["offset"] + r["length"] <= num_tokens:
                keep_count += 1
            elif r["offset"] < num_tokens:
                raise ValueError(
                    f"{modality} placeholder [{r['offset']}, {r['offset'] + r['length']}) straddles the "
                    f"sequence end {num_tokens}"
                )
        if not keep_count:
            continue
        kept["mm_placeholders"][modality] = copy.deepcopy(ranges[:keep_count])
        kept["mm_hashes"][modality] = list((features.get("mm_hashes") or {}).get(modality, [])[:keep_count])
        if kwargs_data is not None and modality in kwargs_data:
            if kept["kwargs_data"] is None:
                kept["kwargs_data"] = {}
            kept["kwargs_data"][modality] = list(kwargs_data[modality][:keep_count])
    return kept if kept["mm_placeholders"] else None


class VLLMChatRenderer:
    """Renders chat messages into token ids and multi-modal features via the inference server.

    Observations are rendered as ``[*base_conversation, *new_obs]`` and the tokens after the
    rendered base are kept, the same fixed-base approach ``SkyRLGymGenerator.get_obs_ids_from_obs``
    uses with the local tokenizer. Two checks guard the slice on every observation: the render must
    start with the rendered base, and no placeholder may fall inside the base.

    Templates that render an image differently depending on the images before it (for example
    Qwen-VL's ``add_vision_id``, which prefixes "Picture N:") are refused. The first time an
    observation with images is rendered, it is also rendered directly after a copy of itself; if
    the observation's tokens differ between the two positions, a ``ValueError`` is raised.
    """

    def __init__(
        self,
        client: Any,
        base_conversation: ConversationType,
        eos_token_id: Optional[int],
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
        """
        Args:
            client: exposes ``async render_chat_completion({"json": body}) -> {"token_ids", "features"}``.
            base_conversation: fixed conversation that observations are rendered after.
            eos_token_id: tokens after the base's last eos are dropped so the observation slice
                starts at the turn separator, as in ``SkyRLGymGenerator``.
            chat_template_kwargs: forwarded to the render request.
            model_name: forwarded as the request's ``model``; the client resolves ``None``.
        """
        self._client = client
        self._base_conversation = base_conversation
        self._eos_token_id = eos_token_id
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._model_name = model_name
        self._base_token_ids: Optional[List[int]] = None
        self._image_position_checked = False
        self._lock = asyncio.Lock()

    async def _render(self, messages: ConversationType, add_generation_prompt: bool) -> RenderedTokens:
        body: Dict[str, Any] = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        if self._model_name is not None:
            body["model"] = self._model_name
        if self._chat_template_kwargs:
            body["chat_template_kwargs"] = self._chat_template_kwargs
        response = await self._client.render_chat_completion({"json": body})
        features = response.get("features") or None
        return RenderedTokens(token_ids=list(response["token_ids"]), features=features)

    async def _get_base_token_ids(self) -> List[int]:
        if self._base_token_ids is None:
            async with self._lock:
                if self._base_token_ids is None:
                    rendered = await self._render(self._base_conversation, add_generation_prompt=False)
                    self._base_token_ids = _trim_after_last_eos(rendered.token_ids, self._eos_token_id)
        return self._base_token_ids

    async def _render_suffix(
        self,
        prefix: ConversationType,
        prefix_ids: List[int],
        new_messages: ConversationType,
        add_generation_prompt: bool,
        tokens_only: bool = False,
    ) -> RenderedTokens:
        """Render ``prefix + new_messages`` and return the tokens and features after ``prefix_ids``.

        With ``tokens_only``, features are not returned and the prefix may contain images.

        Raises:
            ValueError: if the render does not start with ``prefix_ids``, or, unless ``tokens_only``,
                if a placeholder falls inside the prefix.
        """
        rendered = await self._render([*prefix, *new_messages], add_generation_prompt=add_generation_prompt)
        n = len(prefix_ids)
        if rendered.token_ids[:n] != prefix_ids:
            raise ValueError(
                "Rendered observation does not start with the rendered base conversation, so its tokens "
                "cannot be sliced off. The chat template renders earlier messages differently when later "
                "messages follow."
            )
        if tokens_only or not _num_placeholders(rendered.features):
            return RenderedTokens(token_ids=rendered.token_ids[n:], features=None)
        for modality, ranges in rendered.features["mm_placeholders"].items():
            for r in ranges:
                if r["offset"] < n:
                    raise ValueError(
                        f"{modality} placeholder at offset {r['offset']} falls inside the base conversation "
                        f"({n} tokens)"
                    )
        return RenderedTokens(token_ids=rendered.token_ids[n:], features=shift_mm_features(rendered.features, -n))

    async def _check_image_position_independent(self, new_obs: ConversationType, obs: RenderedTokens) -> None:
        # The probe repeats the observation without an assistant turn in between, since thinking
        # templates render the last assistant turn differently from earlier ones.
        prefix = [*self._base_conversation, *new_obs]
        try:
            prefix_ids = _trim_after_last_eos(
                (await self._render(prefix, add_generation_prompt=False)).token_ids, self._eos_token_id
            )
            repeated = await self._render_suffix(
                prefix, prefix_ids, new_obs, add_generation_prompt=False, tokens_only=True
            )
        except Exception as e:
            raise ValueError(
                "Could not render an image observation after a copy of itself, which the vision-language "
                "generator does once to check that image rendering does not depend on earlier images."
            ) from e
        if repeated.token_ids != obs.token_ids:
            raise ValueError(
                "The chat template renders an image observation differently when an image precedes it "
                "(for example Qwen-VL's `add_vision_id`, which numbers images across the conversation). "
                "The vision-language generator appends each observation's tokens without re-rendering "
                "earlier turns, so it requires image rendering that does not depend on earlier images. "
                "Remove the option from `generator.chat_template_kwargs`."
            )

    async def render_prompt(self, messages: ConversationType) -> RenderedTokens:
        """Render the initial prompt with the generation prompt appended."""
        rendered = await self._render(messages, add_generation_prompt=True)
        features = rendered.features if _num_placeholders(rendered.features) else None
        return RenderedTokens(token_ids=rendered.token_ids, features=features)

    async def render_observation(self, new_obs: ConversationType, is_done: bool) -> RenderedTokens:
        """Render observation messages as they follow an assistant turn.

        Returns the observation tokens (plus the generation prompt unless ``is_done``) and their
        features, with placeholder offsets relative to the returned tokens.
        """
        if not new_obs and is_done:
            return RenderedTokens(token_ids=[], features=None)
        base_ids = await self._get_base_token_ids()
        obs = await self._render_suffix(self._base_conversation, base_ids, new_obs, add_generation_prompt=not is_done)
        if obs.features is not None and not self._image_position_checked:
            without_generation_prompt = (
                obs
                if is_done
                else await self._render_suffix(self._base_conversation, base_ids, new_obs, add_generation_prompt=False)
            )
            await self._check_image_position_independent(new_obs, without_generation_prompt)
            self._image_position_checked = True
        return obs

    async def render_conversation(self, messages: ConversationType, add_generation_prompt: bool) -> RenderedTokens:
        """Render a whole conversation. Used by the opt-in re-render consistency check."""
        return await self._render(messages, add_generation_prompt=add_generation_prompt)
