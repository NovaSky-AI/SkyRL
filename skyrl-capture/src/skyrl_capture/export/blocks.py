"""Put the text back next to the tokens.

A `tokens`-mode trajectory is token IDs all the way down, which is the right
shape for training and the wrong one for reading. The IDs are what the trainer
consumes; they are not what anyone can check. A person looking at a trajectory
is asking a question the numbers cannot answer -- *is this the right text, and
are the right parts of it marked trainable?* -- and the only surface on which
that is answerable is decoded text with the mask laid over it.

So this takes one root-to-leaf path, exactly as `token_samples` exports it, and
returns it as readable blocks. Four things make it readable rather than merely
decoded:

* **Four kinds, not two**. `trainable` is a boolean; the useful
  classification is not. `sampled` is what the model produced and is in the
  loss. `replayed` is assistant text the model did **not** produce -- the case
  tokens mode exists to catch, and indistinguishable from a tool result in the
  counts. `scaffold` is the template's generation prefix inside an assistant
  turn, which is where the `sampled_start` boundary lives. `given` is context.

* **Cut on the mask *and* on the turn boundary**. Either alone is
  unreadable: cutting only where trainability changes renders a multi-turn
  prompt as one slab, and cutting only on messages hides the scaffold, because
  it lives *inside* the assistant turn. Cut on both and the scaffold falls out
  by itself.

* **Every block carries its token range**, indexing `input_ids`,
  `loss_mask` and `rollout_logprobs` identically -- so a range read off a screen
  can be pasted into whatever is being debugged.

* **Special tokens are shown, not stripped**. Every boundary bug found so
  far lives at one of `<|im_start|>`, `<|im_end|>` or `<think>`.

Text and exact token offsets were captured by the renderer that produced the
IDs. Readers only assemble that representation; they never load a tokenizer.
"""

from __future__ import annotations

from typing import Any

from skyrl_capture.export import formats
from skyrl_capture.export.captured_text import captured_text_for
from skyrl_capture.export.view import TrajectoryView

# The kind of a block, most-to-least interesting.
SAMPLED = "sampled"
REPLAYED = "replayed"
SCAFFOLD = "scaffold"
GIVEN = "given"


def path_blocks(
    ids: list[int], mask: list[int], *, captured_text: Any, turn_start_token: int | None
) -> list[dict[str, Any]]:
    """Blocks for one path's tokens and mask.

    ``captured_text`` addresses text and character offsets by token span.
    ``turn_start_token`` is the id the chat template opens a turn with.

    **Without it the classification degrades, not just the readability.** Roles
    are read off the turn marker, so with no marker no block is ever
    ``assistant``, and `replayed` and `scaffold` -- the two kinds worth having
    -- become unreachable: everything untrainable is `given`. The blocking is
    still correct about what is trainable. `path_views` reports which of the
    two it did.
    """
    if not ids:
        return []

    cuts = {0, len(ids)}
    if turn_start_token is not None:
        cuts.update(index for index, token in enumerate(ids) if token == turn_start_token)
    cuts.update(index for index in range(1, len(ids)) if mask[index] != mask[index - 1])

    ordered = sorted(cuts)
    out: list[dict[str, Any]] = []
    for begin, end in zip(ordered, ordered[1:], strict=False):
        text = captured_text.span(begin, end)
        block = {
            "trainable": bool(mask[begin]),
            "role": _role(text),
            "start": begin,
            "end": end,
            "token_count": end - begin,
            "text": text,
        }
        # Where each token starts inside this block's text, in UTF-16 code
        # units, so a reader can point at one without owning a tokenizer. The
        # list is non-decreasing: tokens sharing one multi-byte character give
        # it to the first of them and the rest are empty. Absent only when a
        # span has no offsets at all, which leaves the text exact and renders
        # as one run.
        block["token_ids"] = list(ids[begin:end])
        offsets = captured_text.token_slices(begin, end)
        if offsets is not None:
            block["token_offsets"] = offsets
        out.append(block)

    # Kind is structural, not textual. The generation scaffold is whatever of an
    # assistant turn falls before the first sampled token -- and because the
    # cuts above include every mask change, that is exactly an assistant block
    # whose next block is trainable. No string matching, which matters: SkyRL's
    # template puts `<think>\n` in the scaffold and this one does not, so a test
    # for "opens an assistant turn and contains nothing else" is right for one
    # renderer and wrong for the other.
    for index, block in enumerate(out):
        following = out[index + 1] if index + 1 < len(out) else None
        if block["trainable"]:
            block["kind"] = SAMPLED
        elif block["role"] == "assistant" and following is not None and following["trainable"]:
            block["kind"] = SCAFFOLD
        elif block["role"] == "assistant":
            block["kind"] = REPLAYED
        else:
            block["kind"] = GIVEN
    return out


def _attribute(blocks: list[dict[str, Any]], path: list[Any], total: int) -> None:
    """Name the node each block came from, when that can be known.

    A path's tokens are its nodes' tokens concatenated, so the node boundaries
    are the running totals. That is what lets the UI apply the train-once rule --
    a sampled node reachable from several branches is trainable in exactly one --
    which needs blocks and nodes to be talking about the same thing.

    If the running total does not come to the path's length then something
    bridged or trimmed between the nodes and the row, and every boundary below
    would be off by that much. Attribute nothing rather than something wrong:
    a `node_id` that is quietly one node out is worse than an absent one.
    """
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for node in path:
        width = len(node.token_ids)
        spans.append((node.id, cursor, cursor + width))
        cursor += width
    if cursor != total:
        for block in blocks:
            block["node_id"] = None
        return
    for block in blocks:
        block["node_id"] = next(
            (node_id for node_id, begin, end in spans if begin <= block["start"] < end), None
        )


def _role(text: str) -> str | None:
    """The role this block opens, when it opens one."""
    marker = "<|im_start|>"
    if not text.startswith(marker):
        return None
    return text[len(marker) :].split("\n", 1)[0].strip() or None


def path_views(
    view: TrajectoryView,
    *,
    text: bool = True,
) -> list[dict[str, Any]]:
    """Every root-to-leaf path in this trajectory, ready to read.

    The unit is the path, not the node: a path is what an export row is,
    so what is read on screen and what reaches the trainer are the same object.
    A text-mode trajectory has no tokens, so its paths come back with messages
    and no blocks -- the same shape, minus what does not exist.
    """
    rows = formats.token_sample_records(view)
    by_path = {row["path_id"]: row for row in rows}
    out: list[dict[str, Any]] = []

    for index, path in enumerate(formats._paths(view)):  # noqa: SLF001 - one module over
        if not path:
            continue
        path_id = f"{view.trajectory.id}-p{index:04d}"
        row = by_path.get(path_id)
        entry: dict[str, Any] = {
            "path_id": path_id,
            "node_ids": [node.id for node in path],
            "leaf_node_id": path[-1].id,
            "abandoned": row["abandoned"] if row else False,
            "masked_reason": row.get("masked_reason") if row else None,
            "stop_reason": row.get("stop_reason") if row else None,
            "token_count": 0,
            "trainable_count": 0,
            # "mask+turns" distinguishes all four kinds; "mask" cannot tell
            # `replayed` or `scaffold` from `given`, because roles are read off
            # the turn marker and this tokenizer has none.
            "blocking": None,
            "blocks": [],
            # Text mode has no mask to lay over anything, so the path is its
            # messages. Shown rather than omitted: the tree is the same shape.
            "messages": [
                {"role": node.role, "author": node.author, "chars": len(str(node.message or ""))}
                for node in path
            ],
        }
        if row and row.get("input_ids"):
            captured_text = captured_text_for(path)
            entry["token_count"] = len(row["input_ids"])
            entry["trainable_count"] = row["trainable_count"]
            entry["blocking"] = "mask+turns" if captured_text.turn_start_token is not None else "mask"
            entry["blocks"] = path_blocks(
                row["input_ids"],
                row["loss_mask"],
                captured_text=captured_text,
                turn_start_token=captured_text.turn_start_token,
            )
            _attribute(entry["blocks"], path, entry["token_count"])
            # Whether every block can be addressed token by token. `partial`
            # means some span has no offsets at all, so its text is exact and
            # its internal boundaries are not known. A character shared
            # between tokens does not cause this: those tokens are grouped and
            # the span stays addressable.
            entry["token_offsets"] = (
                "exact"
                if all("token_offsets" in block for block in entry["blocks"])
                else "partial"
            )
            # Always the count, only sometimes the array. A caller that dropped
            # the text still has to tell "no logprobs" from "not asked for",
            # and those are one flag apart on the table.
            entry["logprob_count"] = len(row.get("rollout_logprobs") or ())
            if text:
                entry["logprobs"] = row.get("rollout_logprobs")
            else:
                # Kinds and roles are read off the text, so it is decoded and
                # then dropped: the caller wanted the shape, not the words.
                for block in entry["blocks"]:
                    block.pop("text", None)
                    block.pop("token_ids", None)
                    block.pop("token_offsets", None)
        out.append(entry)
    return out
