"""The on-disk record: round trips, the file split, and when a trajectory is written."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import openai
import pytest
import zstandard

from skycap import record
from skycap.graph import CallInfo, NodeTokens
from skycap.trajectory import Failure, Trajectory
from tests.conftest import Stack, openai_client, running_stack


def _client(base_url: str) -> openai.AsyncOpenAI:
    return openai_client(base_url)


PROMPT_TEXT = "<s>user\nqé\n"
REPLY_TEXT = "<s>assistant\n🙂ok"


def _offsets(pieces: list[str]) -> list[int]:
    """Each piece's UTF-8 byte offset in their concatenation."""
    out, cursor = [], 0
    for piece in pieces:
        out.append(cursor)
        cursor += len(piece.encode())
    return out


def _token_trajectory() -> Trajectory:
    """A prompt node and a sampled node, with every array kind and multi-byte text."""
    trajectory = Trajectory(id="tr_tokens", meta={"step": 3})
    graph = trajectory.graph
    prompt, _ = graph.add(
        None,
        role="user",
        author="client",
        message={"role": "user", "content": "qé"},
        match_hash="m0",
        delta_hash="d0",
        created_at=1.0,
        tokens=NodeTokens(
            token_ids=[1, 2, 3],
            routed_experts=np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2),
            text=PROMPT_TEXT,
            text_offsets=_offsets(["<s>", "user\n", "qé\n"]),
        ),
    )
    reply, _ = graph.add(
        prompt.id,
        role="assistant",
        author="model",
        message={"role": "assistant", "content": "🙂ok"},
        match_hash="m1",
        delta_hash="d1",
        created_at=2.0,
        tokens=NodeTokens(
            token_ids=[4, 5, 6, 7],
            sampled_start=1,
            logprobs=[0.0, -0.5, -0.25, -1.0],
            routed_experts=np.full((4, 2, 2), 9, dtype=np.uint8),
            sampling_mask=[[5, 8], [6], [7, 1, 2]],
            text=REPLY_TEXT,
            # The emoji takes two tokens; it belongs to the second, which
            # completes it, and the first has an empty span.
            text_offsets=_offsets(["<s>assistant\n", "", "🙂", "ok"]),
        ),
    )
    reply.calls.append(CallInfo(t_start=1.5, t_end=2.0, model="policy", sampling={"top_p": 0.9}))
    trajectory.failures.append(Failure(t=1.2, status=500, error="boom"))
    trajectory.seal("finished", {"reward": 1.0})
    return trajectory


def test_a_token_trajectory_round_trips(tmp_path: Path) -> None:
    original = _token_trajectory()
    record.write(tmp_path, original)
    loaded = record.load(tmp_path, original.id)

    assert loaded.document() == original.document()
    for before, after in zip(original.graph, loaded.graph, strict=True):
        assert before.tokens is not None and after.tokens is not None
        assert after.tokens.token_ids == before.tokens.token_ids
        assert after.tokens.sampled_start == before.tokens.sampled_start
        assert after.tokens.logprobs == before.tokens.logprobs
        assert after.tokens.sampling_mask == before.tokens.sampling_mask
        assert after.tokens.text == before.tokens.text
        assert after.tokens.text_offsets == before.tokens.text_offsets
        np.testing.assert_array_equal(after.tokens.routed_experts, before.tokens.routed_experts)


def test_the_files_are_split_by_who_reads_them(tmp_path: Path) -> None:
    record.write(tmp_path, _token_trajectory())

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "tr_tokens.experts.zst",
        "tr_tokens.json.zst",
        "tr_tokens.sampling_mask.zst",
        "tr_tokens.tokens.zst",
    ]
    for kind in ("tokens", "experts", "sampling_mask"):
        record.sidecar_path(tmp_path, "tr_tokens", kind).unlink()
    document = record.read_document(tmp_path, "tr_tokens")
    assert document["format_version"] == 1
    assert document["nodes"][1]["tokens"] == {
        "offset": 3,
        "length": 4,
        "sampled_start": 1,
        "has_logprobs": True,
        "text_offset": len(PROMPT_TEXT.encode()),
        "text_bytes": len(REPLY_TEXT.encode()),
        "experts_offset": 3,
        "experts_rows": 4,
        "mask_offset": 0,
        "mask_rows": 3,
    }
    assert list(record.list_ids(tmp_path)) == ["tr_tokens"]


def test_a_sidecar_is_raw_aligned_arrays_a_manifest_describes(tmp_path: Path) -> None:
    """What a reader in any language does: decompress, then view each array at its offset."""
    record.write(tmp_path, _token_trajectory())
    document = record.read_document(tmp_path, "tr_tokens")
    tokens = document["sidecars"]["tokens"]
    raw = zstandard.ZstdDecompressor().decompressobj().decompress((tmp_path / tokens["file"]).read_bytes())
    arrays = tokens["arrays"]

    assert {name: spec["dtype"] for name, spec in arrays.items()} == {
        "token_ids": "int32",
        "logprobs": "float64",
        "text_offsets": "int32",
        "text": "uint8",
    }
    assert all(spec["offset"] % 8 == 0 for spec in arrays.values())
    ids = struct.unpack_from("<7i", raw, arrays["token_ids"]["offset"])
    assert ids == (1, 2, 3, 4, 5, 6, 7)
    text = raw[arrays["text"]["offset"] :][: arrays["text"]["shape"][0]]
    assert text.decode() == PROMPT_TEXT + REPLY_TEXT
    experts = document["sidecars"]["experts"]["arrays"]["routed_experts"]
    assert (experts["dtype"], experts["shape"]) == ("uint8", [7, 2, 2])


def test_each_token_maps_to_whole_characters_of_the_text(tmp_path: Path) -> None:
    record.write(tmp_path, _token_trajectory())
    reply = record.load(tmp_path, "tr_tokens").graph.nodes[1].tokens
    assert reply is not None and reply.text is not None and reply.text_offsets is not None

    data = reply.text.encode()
    bounds = [*reply.text_offsets, len(data)]
    spans = [data[a:b] for a, b in zip(bounds, bounds[1:])]
    assert spans[0] == b"<s>assistant\n"
    assert spans[1:] == [b"", "🙂".encode(), b"ok"]
    assert b"".join(spans) == data


def test_bad_text_offsets_are_refused_at_write(tmp_path: Path) -> None:
    trajectory = Trajectory(id="tr_bad")
    trajectory.graph.add(
        None,
        role="user",
        author="client",
        message={"role": "user", "content": "q"},
        match_hash="m",
        delta_hash="d",
        created_at=0.0,
        tokens=NodeTokens(token_ids=[1, 2], text="ab", text_offsets=[1, 0]),
    )
    with pytest.raises(ValueError):
        record.write(tmp_path, trajectory)


def test_a_text_trajectory_has_only_a_document(tmp_path: Path) -> None:
    trajectory = Trajectory(id="tr_text")
    trajectory.graph.commit_text(
        [{"role": "user", "content": "q"}],
        {"role": "assistant", "content": "a"},
        tools=None,
        model="policy",
        call=CallInfo(t_start=0.0, t_end=1.0),
    )
    record.write(tmp_path, trajectory)

    assert [p.name for p in tmp_path.iterdir()] == ["tr_text.json.zst"]
    assert record.read_document(tmp_path, "tr_text")["sidecars"] == {}
    assert record.load(tmp_path, "tr_text").document() == trajectory.document()


async def test_finish_writes_the_record_and_frees_memory(recorded_stack: Stack) -> None:
    created = await recorded_stack.create({"task": "t"})
    await _client(created["base_url"]).chat.completions.create(
        model="policy", messages=[{"role": "user", "content": "q"}]
    )
    finished = await recorded_stack.finish(created["id"], {"reward": 0.5})

    record_dir = recorded_stack.server.record_dir
    assert record_dir is not None
    assert created["id"] not in recorded_stack.server.trajectories
    on_disk = record.read_document(record_dir, created["id"])
    assert on_disk["status"] == "finished"
    assert on_disk["annotations"] == {"reward": 0.5}
    assert await recorded_stack.document(created["id"]) == on_disk
    assert await recorded_stack.finish(created["id"]) == finished


async def test_an_ended_route_answers_410_from_disk(recorded_stack: Stack) -> None:
    created = await recorded_stack.create()
    await recorded_stack.finish(created["id"])

    try:
        await _client(created["base_url"]).chat.completions.create(
            model="policy", messages=[{"role": "user", "content": "q"}]
        )
    except openai.APIStatusError as error:
        assert error.status_code == 410
    else:
        raise AssertionError("expected 410")


async def test_an_idle_trajectory_is_abandoned_after_the_ttl(tmp_path: Path) -> None:
    async with running_stack(record_dir=tmp_path, ttl=0.0) as running:
        idle = await running.create()
        await _client(idle["base_url"]).chat.completions.create(
            model="policy", messages=[{"role": "user", "content": "q"}]
        )
        swept = await running.server.sweep()

        assert swept == [idle["id"]]
        document = record.read_document(tmp_path, idle["id"])
        assert document["status"] == "abandoned"
        assert len(document["nodes"]) == 2


async def test_a_busy_trajectory_is_not_abandoned(tmp_path: Path) -> None:
    async with running_stack(record_dir=tmp_path, ttl=3600.0) as running:
        await running.create()
        assert await running.server.sweep() == []


async def test_graceful_shutdown_writes_open_trajectories(tmp_path: Path) -> None:
    async with running_stack(record_dir=tmp_path) as running:
        created = await running.create()
        await _client(created["base_url"]).chat.completions.create(
            model="policy", messages=[{"role": "user", "content": "q"}]
        )

    document = record.read_document(tmp_path, created["id"])
    assert document["status"] == "open"
    assert len(document["nodes"]) == 2


async def test_a_failed_write_still_finishes(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    async with running_stack(record_dir=blocked) as running:
        created = await running.create()
        finished = await running.finish(created["id"])

        assert finished["status"] == "finished"
        assert created["id"] in running.server.trajectories
