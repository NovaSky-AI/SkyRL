"""Model-aware rendering.

In tokens mode the proxy -- not the client, and not the inference server -- owns
the conversion from messages to token IDs. That is the whole point: the tokens
stored are the tokens inference saw, because the proxy produced them.

A renderer must be able to do four things, and the third is the one that is
easy to miss:

1. ``render`` a message list into prompt token IDs **with per-token message
   attribution**, so the prompt can be split into one node per message.
2. ``bridge`` from a previous turn's exact prompt+completion IDs, extending
   them with new messages without re-tokenizing what was already sampled.
3. ``parse_response`` exact sampled token IDs back into an assistant message.
4. Report the model's stop token IDs so generation stops the way the model's
   own format expects.

Two implementations ship:

* ``BuiltinRenderer`` -- a dependency-free, deterministic ChatML-style renderer
  with a byte-level tokenizer. It needs no model download, so the whole token capture
  test suite and the offline benchmark run against it.
* ``PrimeRenderer`` -- the real one, over Prime Intellect's ``renderers``
  library. It reports per-token message attribution directly, keeps reasoning
  content across turns, parses tool calls with a status per attempt, and can
  extend a previous turn's exact tokens rather than re-rendering.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from skyrl_capture.tito.types import Message, RenderedPrompt, TokenError, ToolSpec


@runtime_checkable
class TokenRenderer(Protocol):
    def render(self, messages: Sequence[Message], *, tools: Sequence[ToolSpec] | None = None) -> RenderedPrompt: ...

    def bridge(
        self,
        previous_prompt_ids: Sequence[int],
        previous_completion_ids: Sequence[int],
        new_messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> RenderedPrompt | None: ...

    def parse_response(
        self, token_ids: Sequence[int], *, tools: Sequence[ToolSpec] | None = None
    ) -> Message: ...

    def get_stop_token_ids(self) -> Sequence[int]: ...

    def decode_token(self, token_id: int) -> str: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def turn_start_token(self) -> int | None: ...

    @property
    def name(self) -> str: ...


def _tool_call_id(token_ids: Sequence[int], index: int) -> str:
    """Deterministic tool-call ID.

    Derived from the sampled tokens so re-parsing the same completion yields
    the same ID, which keeps message hashes stable across a replay.
    """
    digest = hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()
    return f"call_{digest[:20]}_{index}"


def normalize_tools(tools: Sequence[ToolSpec] | None) -> list[ToolSpec] | None:
    """Flatten OpenAI's ``{"type": "function", "function": {...}}`` wrapper."""
    if tools is None:
        return None
    normalized: list[ToolSpec] = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else None
        normalized.append(dict(function) if isinstance(function, dict) else dict(tool))
    return normalized


# -- built-in ---------------------------------------------------------------
class ByteTokenizer:
    """A tiny reversible tokenizer.

    Bytes map to IDs 0-255; specials live above. It is not a real BPE, but it is
    exactly reversible and deterministic, which is what the token capture invariants
    need to be testable without downloading a model.
    """

    IM_START = 256
    IM_END = 257
    NEWLINE = 258
    EOS = 259
    TOOL_START = 260

    SPECIAL_TEXT = {
        IM_START: "<|im_start|>",
        IM_END: "<|im_end|>",
        NEWLINE: "\n",
        EOS: "<|endoftext|>",
        TOOL_START: "<|tool|>",
    }

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        raw = bytearray()
        parts: list[str] = []
        for token_id in token_ids:
            if token_id < 256:
                raw.append(token_id)
                continue
            if raw:
                parts.append(raw.decode("utf-8", errors="replace"))
                raw = bytearray()
            parts.append(self.SPECIAL_TEXT.get(token_id, ""))
        if raw:
            parts.append(raw.decode("utf-8", errors="replace"))
        return "".join(parts)

    def decode_one(self, token_id: int) -> str:
        if token_id < 256:
            return bytes([token_id]).decode("utf-8", errors="replace")
        return self.SPECIAL_TEXT.get(token_id, "")


class BuiltinRenderer:
    """ChatML-style renderer over :class:`ByteTokenizer`.

    Layout per message::

        <|im_start|> role \\n content <|im_end|> \\n

    The role header and the closing tokens are scaffold (``-1``) except the
    closing marker, which is attributed to the message it closes. The trailing
    ``<|im_start|>assistant\\n`` is the generation prompt and belongs to no
    message.
    """

    def __init__(self, *, model: str = "builtin", tools_in_system: bool = True) -> None:
        self._tokenizer = ByteTokenizer()
        self._model = model
        self._tools_in_system = tools_in_system

    @property
    def name(self) -> str:
        return f"builtin:{self._model}"

    def _message_tokens(self, message: Message) -> tuple[list[int], list[bool]]:
        """Return ``(tokens, is_body)`` where body tokens belong to the message."""
        tokenizer = self._tokenizer
        tokens: list[int] = [tokenizer.IM_START]
        is_body: list[bool] = [False]
        role = str(message.get("role", "user"))
        for token in tokenizer.encode(role):
            tokens.append(token)
            is_body.append(False)
        tokens.append(tokenizer.NEWLINE)
        is_body.append(False)
        for token in tokenizer.encode(_content_text(message)):
            tokens.append(token)
            is_body.append(True)
        tokens.append(tokenizer.IM_END)
        is_body.append(True)
        tokens.append(tokenizer.NEWLINE)
        is_body.append(True)
        return tokens, is_body

    def _tools_prelude(self, tools: Sequence[ToolSpec] | None) -> tuple[list[int], list[int]]:
        """Tool schemas render ahead of the conversation, as scaffold."""
        if not tools or not self._tools_in_system:
            return [], []
        payload = json.dumps(normalize_tools(tools), sort_keys=True, separators=(",", ":"))
        tokens = [self._tokenizer.TOOL_START, *self._tokenizer.encode(payload), self._tokenizer.NEWLINE]
        return tokens, [-1] * len(tokens)

    def render(
        self, messages: Sequence[Message], *, tools: Sequence[ToolSpec] | None = None
    ) -> RenderedPrompt:
        token_ids, indices = self._tools_prelude(tools)
        for index, message in enumerate(messages):
            tokens, is_body = self._message_tokens(message)
            token_ids.extend(tokens)
            indices.extend(index if body else -1 for body in is_body)
        # Generation prompt.
        token_ids.extend([self._tokenizer.IM_START, *self._tokenizer.encode("assistant"), self._tokenizer.NEWLINE])
        indices.extend([-1, *[-1] * len("assistant"), -1])
        return RenderedPrompt(token_ids=tuple(token_ids), message_indices=tuple(indices))

    def bridge(
        self,
        previous_prompt_ids: Sequence[int],
        previous_completion_ids: Sequence[int],
        new_messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> RenderedPrompt | None:
        """Extend the previous exact prompt+completion with new messages.

        The completion ends mid-assistant-message, so the bridge closes that
        message before appending the new ones.

        The prefix is carried over by construction -- it is the caller's own
        token tuples, extended, never re-rendered -- and the indices returned
        describe only the tokens after it, which is the contract
        ``RenderedPrompt`` documents.
        """
        reused = len(previous_prompt_ids) + len(previous_completion_ids)
        token_ids = [*previous_prompt_ids, *previous_completion_ids]
        indices: list[int] = []
        # Close the assistant turn the completion left open.
        closing = [self._tokenizer.IM_END, self._tokenizer.NEWLINE]
        token_ids.extend(closing)
        indices.extend([-1] * len(closing))
        for offset, message in enumerate(new_messages):
            tokens, is_body = self._message_tokens(message)
            token_ids.extend(tokens)
            indices.extend(offset if body else -1 for body in is_body)
        token_ids.extend([self._tokenizer.IM_START, *self._tokenizer.encode("assistant"), self._tokenizer.NEWLINE])
        indices.extend([-1] * (2 + len("assistant")))
        return RenderedPrompt(
            token_ids=tuple(token_ids),
            message_indices=tuple(indices),
            reused_prefix_length=reused,
        )

    def parse_response(
        self, token_ids: Sequence[int], *, tools: Sequence[ToolSpec] | None = None
    ) -> Message:
        text = self._tokenizer.decode(token_ids)
        for marker in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(marker, "")
        message: Message = {"role": "assistant", "content": text}
        calls = _extract_tool_calls(text, token_ids)
        if calls:
            message["tool_calls"] = calls
            # A pure tool call carries no user-visible text.
            message["content"] = None
        return message

    def get_stop_token_ids(self) -> Sequence[int]:
        return (self._tokenizer.IM_END, self._tokenizer.EOS)

    def decode_token(self, token_id: int) -> str:
        return self._tokenizer.decode_one(token_id)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a run of tokens as one string.

        Not a concatenation of `decode_token`: a byte tokenizer splits a
        multi-byte character across tokens, and a real BPE splits a word. Runs
        are decoded whole, which is safe because block boundaries fall on token
        boundaries.
        """
        return self._tokenizer.decode(list(token_ids))

    def turn_start_token(self) -> int | None:
        """The id a chat turn opens with, for cutting blocks on turns."""
        return self._tokenizer.IM_START


def _content_text(message: Message) -> str:
    """Flatten a message's content into the text a template would render."""
    content = message.get("content")
    if content is None:
        calls = message.get("tool_calls")
        if calls:
            return json.dumps(calls, sort_keys=True, separators=(",", ":"))
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("thinking"), str):
                    parts.append(block["thinking"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return json.dumps(content, sort_keys=True, separators=(",", ":"))


def _extract_tool_calls(text: str, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Recognize a ``<|tool|>{json}`` call in a completion."""
    marker = "<|tool|>"
    if marker not in text:
        return []
    calls: list[dict[str, Any]] = []
    for index, chunk in enumerate(text.split(marker)[1:]):
        candidate = chunk.strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
        calls.append(
            {
                "id": payload.get("id") or _tool_call_id(token_ids, index),
                "type": "function",
                "function": {"name": payload["name"], "arguments": arguments},
            }
        )
    return calls


# -- Hugging Face chat template --------------------------------------------
class PrimeRenderer:
    """Renderer backed by Prime Intellect's ``renderers`` library.

    The library exists to handle what a bare ``apply_chat_template`` does not:
    per-model template quirks, reasoning content retained across turns,
    tool-call parsing that reports malformed attempts rather than dropping
    them, and per-token attribution produced directly instead of recovered by
    diffing. It is what SkyRL's token-in/token-out proxy renders through, so
    using it is also the cheapest way to stay aligned with the reference
    implementation.

    Installed by the ``tokens`` extra and used for any tokenizer that is not
    ``builtin``, e.g. ``--tokenizer Qwen/Qwen3-8B``.
    """

    def __init__(
        self,
        tokenizer_name: str,
        *,
        thinking_retention: str = "all",
        chat_template_kwargs: dict[str, Any] | None = None,
        pool_size: int = 8,
    ) -> None:
        try:
            from renderers import AutoRendererConfig, create_renderer_pool
        except ImportError as error:  # pragma: no cover - depends on install
            raise TokenError(
                f"rendering {tokenizer_name!r} needs the renderers library: "
                "uv sync --extra tokens"
            ) from error
        except Exception as error:  # pragma: no cover - depends on install
            # renderers imports 4.x-only transformers paths. On 5.x the failure
            # surfaces as an unrelated ImportError from a half-loaded module,
            # so name the real cause here.
            import transformers

            if transformers.__version__.split(".")[0] not in {"4"}:
                raise TokenError(
                    f"the renderers library needs transformers 4.x, found "
                    f"{transformers.__version__}: pip install 'transformers<5'"
                ) from error
            raise

        self._name = tokenizer_name
        # Resolve the lazy `transformers` attributes before the pool does.
        # `create_renderer_pool` builds its renderers in a ThreadPoolExecutor,
        # and each worker runs `from transformers import AutoTokenizer`.
        # `transformers` is a lazy module, so several threads resolving the
        # same attribute at once can race and one of them sees a partially
        # populated module -- an ImportError for a name that plainly exists.
        # Touching them first makes pool construction deterministic.
        from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast  # noqa: F401

        # `thinking_retention="all"` keeps a reasoning model's thinking tokens
        # in the history. Dropping them would re-render an earlier assistant
        # turn differently from the tokens that were sampled, which the
        # token-space check would then reject as a mismatch -- a correct
        # refusal, and a branch where there should have been prefix reuse.
        self._pool = create_renderer_pool(
            tokenizer_name,
            AutoRendererConfig(thinking_retention=thinking_retention),
            size=pool_size,
            chat_template_kwargs=chat_template_kwargs,
        )
        # Loaded on first use. The pool already holds its own tokenizers; this
        # one only decodes single tokens for an OpenAI-shaped logprobs field,
        # so a target that never asks for logprobs never pays for it.
        self._tokenizer: Any = None
        # Bridges declined because the preserved prefix did not end where we
        # handed it over. Expected to stay at zero; a rising count means
        # trimming is engaging and every one of those turns paid for a full
        # render, so it is worth seeing rather than inferring from latency.
        self.bridge_boundary_rejected = 0

    @property
    def name(self) -> str:
        return f"prime:{self._name}"

    def render(
        self, messages: Sequence[Message], *, tools: Sequence[ToolSpec] | None = None
    ) -> RenderedPrompt:
        rendered = self._pool.render(
            list(messages), tools=normalize_tools(tools), add_generation_prompt=True
        )
        return RenderedPrompt(
            token_ids=tuple(rendered.token_ids),
            message_indices=tuple(rendered.message_indices),
        )

    def bridge(
        self,
        previous_prompt_ids: Sequence[int],
        previous_completion_ids: Sequence[int],
        new_messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> RenderedPrompt | None:
        """Extend an exact previous prompt and completion.

        Unlike the Hugging Face renderer, this one can actually bridge, so a
        continuing trajectory reuses its prefix instead of re-rendering.

        The library's contract is that the returned sequence begins with the
        prefix it was given, and that it returns ``None`` rather than a prefix
        it cannot prove it preserved. That is what lets the trace commit a
        bridged turn without re-reading the prefix: those tokens are the ones
        the proxy already committed, unchanged, by construction.

        The contract has one edge this has to respect. The preserved prefix is
        ``prev_prompt + prev_completion`` only after ``trim_to_turn_close``,
        which keeps the longest prefix ending at a turn-close token -- so a
        completion truncated at ``max_tokens``, or one with tokens after its
        last close, can come back preserved to a *different* length than we
        assumed. The library reports no length, so this checks the boundary and
        declines to bridge when it does not line up. A full render is always
        correct, so falling back costs a slow turn and never correctness.
        """
        with self._pool.checkout() as renderer:
            # The library takes any sequence, so the caller's tuples go
            # straight through. Copying them to lists first cost a full pass
            # over the context on every turn.
            rendered = renderer.bridge_to_next_turn(
                previous_prompt_ids,
                previous_completion_ids,
                new_messages,
                tools=normalize_tools(tools),
            )
        if rendered is None:
            return None
        reused = len(previous_prompt_ids) + len(previous_completion_ids)
        if not self._prefix_survived(rendered.token_ids, previous_completion_ids, reused):
            self.bridge_boundary_rejected += 1
            return None
        return RenderedPrompt(
            token_ids=tuple(rendered.token_ids),
            # The tail alone: attribution for the reused prefix is already on
            # the committed nodes, and re-deriving it cost a pass over the
            # whole context every turn.
            message_indices=tuple(rendered.message_indices[reused:]),
            reused_prefix_length=reused,
        )

    # How much of the reused prefix this checks directly. Checking it in full
    # meant a pass over the whole context every turn, which on a long
    # conversation is the single most expensive thing the proxy does -- and it
    # re-derived a guarantee the library already makes. The boundary is where a
    # bridge actually goes wrong: a renderer that reopens, rewrites, or trims
    # the previous turn disturbs its end, and a trim moves it by construction.
    _PREFIX_PROBE = 64

    def _prefix_survived(
        self,
        token_ids: Sequence[int],
        previous_completion_ids: Sequence[int],
        reused: int,
    ) -> bool:
        """Does the bridge output still carry our prefix, ending where we think?

        False means the boundary moved -- most likely ``trim_to_turn_close``
        kept less than we handed over -- so the caller must re-render rather
        than commit a tail sliced at the wrong offset.
        """
        if len(token_ids) < reused:
            return False
        probe = min(self._PREFIX_PROBE, len(previous_completion_ids))
        if not probe:
            return True
        return tuple(token_ids[reused - probe : reused]) == tuple(
            previous_completion_ids[len(previous_completion_ids) - probe :]
        )

    def parse_response(
        self, token_ids: Sequence[int], *, tools: Sequence[ToolSpec] | None = None
    ) -> Message:
        """Parse sampled tokens into a message, reasoning and tool calls included.

        Only tool calls the renderer parsed cleanly become ``tool_calls``. A
        malformed attempt stays in the text where the model put it, because
        inventing a well-formed call out of a broken one would record
        something the model never emitted.
        """
        from renderers.base import ToolCallParseStatus

        parsed = self._pool.parse_response(list(token_ids), tools=normalize_tools(tools))
        message: Message = {"role": "assistant", "content": parsed.content}
        reasoning = getattr(parsed, "reasoning_content", None)
        if reasoning is not None:
            message["reasoning_content"] = reasoning

        calls: list[dict[str, Any]] = []
        for index, call in enumerate(getattr(parsed, "tool_calls", ()) or ()):
            if call.status != ToolCallParseStatus.OK or not call.name:
                continue
            arguments = call.arguments
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, separators=(",", ":"), ensure_ascii=False)
            calls.append(
                {
                    "id": call.id or _tool_call_id(token_ids, index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            )
        if calls:
            message["tool_calls"] = calls
        return message

    def get_stop_token_ids(self) -> Sequence[int]:
        return tuple(self._pool.get_stop_token_ids())

    def decode_token(self, token_id: int) -> str:
        return self.decode([token_id])

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a run of tokens as one string, special tokens kept.

        Kept, not stripped: every boundary bug found so far lives at one of
        `<|im_start|>`, `<|im_end|>` or `<think>`, and a reader who cannot see
        them cannot see the bug.
        """
        return self._hf().decode(
            list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def turn_start_token(self) -> int | None:
        """The id a chat turn opens with, or None for a template with no such
        marker -- then blocks are cut on the mask alone, which is still correct
        and merely less readable."""
        token = self._hf().convert_tokens_to_ids("<|im_start|>")
        return None if token is None or token == self._hf().unk_token_id else int(token)

    def _hf(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._name)
        return self._tokenizer


def build_renderer(*, tokenizer: str | None, model: str | None, config: dict[str, Any] | None = None) -> TokenRenderer:
    """Choose a renderer for a target.

    ``tokenizer: builtin`` (or an unset tokenizer) selects the offline
    renderer, which exists for the test suite and the benchmark. Any other
    value is a Hugging Face tokenizer name rendered through the ``renderers``
    library -- the only path for a real model, because it is the one that keeps
    reasoning content, parses tool calls, and knows the per-model template
    differences. There is deliberately no bare ``apply_chat_template`` option:
    the library's own fallback renderer is that, with per-token attribution on
    top, so a second thinner path would only be a way to capture worse.
    """
    options = config or {}
    if not tokenizer or tokenizer == "builtin":
        return BuiltinRenderer(model=model or "builtin")
    return PrimeRenderer(
        tokenizer,
        thinking_retention=options.get("thinking_retention", "all"),
        chat_template_kwargs=options.get("chat_template_kwargs"),
    )
