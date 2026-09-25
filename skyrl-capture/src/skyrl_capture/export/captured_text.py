"""Capture-time token text, assembled for a root-to-leaf path.

A `tokens`-mode node carries its own decoded text (`TokenNode.attach_text`),
cut where a path would cut it: at the turn marker and at the sampled boundary.
So every block boundary downstream falls on a segment boundary, and a block's
text is the concatenation of the whole segments it spans.

That is the whole trick, and it is what lets every reader open a TITO capture
with neither `transformers` nor the model's tokenizer installed. IDs remain
authoritative for training; the captured text and offsets are authoritative
for presentation.
"""

from __future__ import annotations

from typing import Any


class CapturedText:
    """The decoded segments of one root-to-leaf path, addressed by token span.

    `span` is deliberately strict. A miss means the cuts moved since the record
    was written, and silently returning a neighbouring segment's text would put
    the wrong words against the right token range -- the one failure this whole
    subsystem exists to make impossible.
    """

    __slots__ = ("_by_span", "_offsets", "_starts", "turn_start_token")

    def __init__(
        self, segments: list[tuple[int, int, str, list[int]]], turn_start_token: int | None
    ) -> None:
        self._by_span = {(begin, end): text for begin, end, text, _ in segments}
        self._offsets = {(begin, end): offsets for begin, end, _, offsets in segments}
        self._starts = sorted(self._by_span)
        self.turn_start_token = turn_start_token

    def token_slices(self, begin: int, end: int) -> list[int] | None:
        """Offsets into the text of the tokens in `[begin, end)`, or `None`.

        One longer than the token count: token `i` of the span is the text
        between `offsets[i]` and `offsets[i+1]`, counted in UTF-16 code units.
        Not code points: these are used to slice a JSON string, and an emoji
        is one code point and two UTF-16 code units, so code points would put
        every token after one out of place.

        The list is non-decreasing rather than strictly increasing: where a
        multi-byte character is split across tokens, the capture groups them
        and gives the character to the first, so the others are empty. That
        keeps every token addressable -- index, id, mask bit, logprob -- and
        confines the arbitrariness to which token inside a shared character
        owns its characters, where there is no non-arbitrary answer.

        `None` only when a segment it covers has no offsets at all, which a
        byte-level BPE should not produce. Approximate offsets are never
        invented: that would put the wrong characters against the right token,
        which is the one thing this subsystem exists to make impossible.
        """
        out = [0]
        cursor, running = begin, 0
        for seg_begin, seg_end in self._starts:
            if seg_begin != cursor:
                continue
            offsets = self._offsets[(seg_begin, seg_end)]
            if offsets is None:
                return None
            out.extend(running + value for value in offsets[1:])
            running += offsets[-1]
            cursor = seg_end
            if cursor >= end:
                break
        if cursor != end:
            raise KeyError(f"no stored offsets covering [{begin}:{end}]")
        return out

    def span(self, begin: int, end: int) -> str:
        """The text of `[begin, end)`, joined from whole segments."""
        if begin == end:
            return ""
        exact = self._by_span.get((begin, end))
        if exact is not None:
            return exact
        out: list[str] = []
        cursor = begin
        for seg_begin, seg_end in self._starts:
            if seg_begin != cursor:
                continue
            out.append(self._by_span[(seg_begin, seg_end)])
            cursor = seg_end
            if cursor >= end:
                break
        if cursor != end:
            raise KeyError(f"no stored text covering [{begin}:{end}]")
        return "".join(out)


def captured_text_for(path: list[Any]) -> CapturedText:
    """Assemble the required capture-time text for one TITO path."""
    segments: list[tuple[int, int, str, list[int] | None]] = []
    turn_start_token: int | None = None
    cursor = 0
    for node in path:
        tokens = (node.payload or {}).get("tokens") or {}
        stored = tokens.get("text_segments")
        if not stored:
            raise ValueError(f"TITO node {node.id!r} has no captured token text")
        if turn_start_token is None:
            turn_start_token = tokens.get("turn_start_token")
        for segment in stored:
            offsets = segment.get("offsets")
            expected = segment["end"] - segment["start"] + 1
            # `None` is a stated absence -- a character spanning tokens -- and
            # is carried through. A list of the wrong length is a corrupt
            # record, and is not.
            if offsets is not None and (
                not isinstance(offsets, list) or len(offsets) != expected
            ):
                raise ValueError(
                    f"TITO node {node.id!r} has invalid captured token offsets"
                )
            segments.append((
                cursor + segment["start"],
                cursor + segment["end"],
                segment["text"],
                offsets,
            ))
        cursor += len(node.token_ids)
    if not segments:
        raise ValueError("TITO path has no captured token text")
    return CapturedText(segments, turn_start_token)
