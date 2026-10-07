"""The record mirror: a byte-for-byte copy of each record, made in the background, that fails open."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import fsspec
import pytest
from aiohttp.test_utils import TestServer
from fsspec.implementations.memory import MemoryFileSystem

from skycap import CapturePool, CaptureService, FinishResult, RecordLocation, record
from skycap.cli import build_parser, build_server
from skycap.mirror import RecordMirror
from skycap.server import CaptureServer, node_address
from skycap.text import TextBackend
from skycap.trajectory import Trajectory
from tests.conftest import running_stack
from tests.fake_renderer import FakeRenderer
from tests.mock_engine import MockEngine
from tests.test_tokens import client, converse


class FlakyFS(MemoryFileSystem):
    """The memory filesystem, whose ``put_file`` follows ``plan``: an exception to raise or an event to wait on."""

    protocol = "flaky"
    cachable = False
    plan: list[BaseException | threading.Event] = []
    calls: list[str] = []

    @classmethod
    def _strip_protocol(cls, path):  # type: ignore[no-untyped-def]
        # The same paths as memory://, so a copy can be read back through it.
        if path.startswith("flaky://"):
            path = "memory://" + path.removeprefix("flaky://")
        return super()._strip_protocol(path)

    def put_file(self, lpath, rpath, *args, **kwargs):  # type: ignore[no-untyped-def]
        FlakyFS.calls.append(rpath)
        step = FlakyFS.plan.pop(0) if FlakyFS.plan else None
        if isinstance(step, threading.Event):
            step.wait(30)
        elif step is not None:
            raise step
        return super().put_file(lpath, rpath, *args, **kwargs)


@pytest.fixture
def flaky() -> Iterator[type[FlakyFS]]:
    fsspec.register_implementation("flaky", FlakyFS, clobber=True)
    FlakyFS.plan, FlakyFS.calls = [], []
    yield FlakyFS
    for step in FlakyFS.plan:
        if isinstance(step, threading.Event):
            step.set()


def unique(protocol: str) -> str:
    return f"{protocol}://mirror-{uuid.uuid4().hex[:8]}"


def remote_bytes(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as handle:
        return handle.read()


def written(record_dir: Path, meta: dict | None = None) -> str:
    """A trajectory written to ``record_dir``, as the server writes one. Returns its id."""
    trajectory = Trajectory(id=f"tr_{uuid.uuid4().hex[:8]}", meta=meta or {})
    record.write(record_dir, trajectory)
    return trajectory.id


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


def settled(mirror: RecordMirror) -> Callable[[], bool]:
    return lambda: mirror.stats()["pending"] == 0


async def test_a_record_is_mirrored_byte_for_byte_sidecars_included(tmp_path: Path) -> None:
    url = unique("memory")
    engine = TestServer(MockEngine().app())
    await engine.start_server()
    service = CaptureService(
        str(engine.make_url("")).rstrip("/"),
        mode="tokens",
        renderer=FakeRenderer(),
        record_dir=str(tmp_path),
        record_mirror=url,
        host="127.0.0.1",
    )
    served = await asyncio.to_thread(service.start)
    try:
        async with CapturePool([served]) as pool, pool.trajectory({}) as trajectory:
            await converse(client(trajectory.base_url), "q", "r")
            result = await trajectory.finish({"reward": 1.0})
    finally:
        # Stopping drains the mirror's queue.
        assert await asyncio.to_thread(service.stop)
        await engine.close()

    name = f"{trajectory.id}.json.zst"
    kinds = ("tokens", "experts", "sampling_mask")
    names = (*[f"{trajectory.id}.{kind}.zst" for kind in kinds], name)
    assert result.record == RecordLocation(
        path=str(tmp_path / name), mirror=f"{url}/{name}", files=names, host=service.server.record_host
    )
    assert result.record.uri == f"{url}/{name}"
    files = record.record_files(tmp_path, trajectory.id)
    assert tuple(path.name for path in files) == names
    for path in files:
        assert remote_bytes(f"{url}/{path.name}") == path.read_bytes()
    assert service.server.mirror.stats() == {
        "mirrored": 1,
        "failed": 0,
        "timed_out": 0,
        "dropped": 0,
        "retried": 0,
        "pending": 0,
    }


async def test_finish_says_where_the_record_is(tmp_path: Path) -> None:
    async with running_stack() as stack:
        assert (await stack.finish((await stack.create())["id"]))["record"] is None
    async with running_stack(record_dir=tmp_path) as stack:
        trajectory_id = (await stack.create())["id"]
        name = f"{trajectory_id}.json.zst"
        # Text mode has no sidecars, so the document is the record's only file.
        location = {"host": node_address(), "path": str(tmp_path / name), "mirror": None, "files": [name]}
        assert (await stack.finish(trajectory_id))["record"] == location
        # A repeat finish is answered from the record on disk, with the same location.
        assert (await stack.finish(trajectory_id))["record"] == location
    assert RecordLocation.from_json(location).uri == f"{node_address()}:{location['path']}"
    assert RecordLocation.from_json(location).files == (location["files"][0],)
    # A server from before files existed answers without them.
    assert RecordLocation.from_json({"path": "/r/x.json.zst", "mirror": None}).files == ()


async def test_finish_lists_the_local_files_without_a_mirror(tmp_path: Path) -> None:
    engine = TestServer(MockEngine().app())
    await engine.start_server()
    service = CaptureService(
        str(engine.make_url("")).rstrip("/"),
        mode="tokens",
        renderer=FakeRenderer(),
        record_dir=str(tmp_path),
        host="127.0.0.1",
    )
    served = await asyncio.to_thread(service.start)
    try:
        async with CapturePool([served]) as pool, pool.trajectory({}) as trajectory:
            await converse(client(trajectory.base_url), "q", "r")
            result = await trajectory.finish({"reward": 1.0})
    finally:
        assert await asyncio.to_thread(service.stop)
        await engine.close()
    assert result.record.mirror is None
    # Served on loopback, so the record's host is this machine's own address, and its uri says where it is.
    assert result.record.host == node_address() and result.record.uri == result.record.local
    assert result.record.files == tuple(path.name for path in record.record_files(tmp_path, trajectory.id))
    assert len(result.record.files) == 4
    assert all((tmp_path / name).exists() for name in result.record.files)


def test_a_mirror_names_the_files_it_holds(tmp_path: Path) -> None:
    from tests.test_record import _token_trajectory

    trajectory = _token_trajectory()
    record.write(tmp_path, trajectory)
    on_disk = [path.name for path in record.record_files(tmp_path, trajectory.id)]
    assert RecordMirror(unique("memory")).names(tmp_path, trajectory.id) == on_disk
    excluding = RecordMirror(unique("memory"), exclude=["experts"])
    assert excluding.names(tmp_path, trajectory.id) == [name for name in on_disk if ".experts." not in name]
    # A text-mode record has its document only.
    text_id = written(tmp_path)
    assert excluding.names(tmp_path, text_id) == [f"{text_id}.json.zst"]


async def test_a_store_that_keeps_failing_fails_open(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    flaky.plan = [ConnectionError("reset")] * 3
    mirror = RecordMirror(unique("flaky"), backoff=0.0)
    async with running_stack(record_dir=tmp_path, record_mirror=mirror) as stack:
        trajectory_id = (await stack.create())["id"]
        finished = await stack.finish(trajectory_id)
        await asyncio.to_thread(wait_for, settled(mirror))
        async with stack.http.get(f"{stack.url}/healthz") as response:
            health = await response.json()

    assert finished["status"] == "finished"
    assert finished["record"]["mirror"] == f"{mirror.url}/{trajectory_id}.json.zst"
    assert record.document_path(tmp_path, trajectory_id).exists()
    assert len(flaky.calls) == 3
    assert health["record_mirror"] == {
        "url": mirror.url,
        "mirrored": 0,
        "failed": 1,
        "timed_out": 0,
        "dropped": 0,
        "retried": 2,
        "pending": 0,
    }


def test_an_error_that_may_pass_is_retried(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    flaky.plan = [ConnectionError("reset"), TimeoutError("slow")]
    mirror = RecordMirror(unique("flaky"), backoff=0.0)
    trajectory_id = written(tmp_path)
    assert mirror.submit(tmp_path, trajectory_id)
    assert mirror.close()

    name = f"{trajectory_id}.json.zst"
    assert remote_bytes(mirror.uri(name).replace("flaky://", "memory://")) == (tmp_path / name).read_bytes()
    assert mirror.stats()["mirrored"] == 1 and mirror.stats()["retried"] == 2


def test_an_error_that_will_not_pass_is_not_retried(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    flaky.plan = [PermissionError("denied")]
    mirror = RecordMirror(unique("flaky"), backoff=0.0)
    assert mirror.submit(tmp_path, written(tmp_path))
    assert mirror.close()
    assert len(flaky.calls) == 1
    assert mirror.stats()["failed"] == 1 and mirror.stats()["retried"] == 0


def test_a_sidecar_that_fails_keeps_the_document_out_of_the_mirror(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    from tests.test_record import _token_trajectory

    trajectory = _token_trajectory()
    record.write(tmp_path, trajectory)
    flaky.plan = [PermissionError("denied")]
    mirror = RecordMirror(unique("flaky"))
    assert mirror.submit(tmp_path, trajectory.id)
    assert mirror.close()
    # The first sidecar failed, so nothing after it was tried.
    assert flaky.calls == [f"{mirror._root}/{trajectory.id}.tokens.zst"]
    assert mirror.stats()["failed"] == 1


async def test_a_copy_that_overruns_its_timeout_is_abandoned_not_retried(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    hang = threading.Event()
    flaky.plan = [hang]
    mirror = RecordMirror(unique("flaky"), timeout=0.2, backoff=0.0)
    async with running_stack(record_dir=tmp_path, record_mirror=mirror) as stack:
        # The store never answers, and finish doesn't wait for it.
        finished = await stack.finish((await stack.create())["id"])
        await asyncio.to_thread(wait_for, settled(mirror))
    assert finished["status"] == "finished"
    hang.set()
    time.sleep(0.1)
    # One call: a retry could land a second copy of what the first may still write.
    assert len(flaky.calls) == 1
    assert mirror.stats()["timed_out"] == 1 and mirror.stats()["failed"] == 1


def test_a_record_that_finds_the_queue_full_is_dropped(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    hang = threading.Event()
    flaky.plan = [hang]
    mirror = RecordMirror(unique("flaky"), workers=1, queue_size=1)
    assert mirror.submit(tmp_path, written(tmp_path))
    wait_for(lambda: len(flaky.calls) == 1)  # the worker holds the first; the queue is empty
    assert mirror.submit(tmp_path, written(tmp_path))
    assert not mirror.submit(tmp_path, written(tmp_path))
    assert mirror.stats()["dropped"] == 1
    hang.set()
    assert mirror.close()
    assert mirror.stats()["mirrored"] == 2


def test_shutdown_drops_what_is_left_at_its_deadline(tmp_path: Path, flaky: type[FlakyFS]) -> None:
    hang = threading.Event()
    flaky.plan = [hang]
    mirror = RecordMirror(unique("flaky"), workers=1, timeout=30.0)
    assert mirror.submit(tmp_path, written(tmp_path))
    wait_for(lambda: len(flaky.calls) == 1)
    assert mirror.submit(tmp_path, written(tmp_path))

    started = time.monotonic()
    assert not mirror.close(timeout=0.3)
    assert time.monotonic() - started < 5.0
    assert mirror.stats()["dropped"] == 1 and mirror.stats()["pending"] == 1
    # Closed: a later record is dropped too.
    assert not mirror.submit(tmp_path, written(tmp_path))
    assert mirror.stats()["dropped"] == 2
    hang.set()


def test_a_mirror_needs_a_record_dir() -> None:
    with pytest.raises(ValueError, match="record_dir"):
        CaptureServer(TextBackend("http://upstream/v1"), record_mirror=unique("memory"))
    args = build_parser().parse_args(["serve", "--upstream-url", "http://u/v1", "--record-mirror", "memory://x"])
    with pytest.raises(SystemExit, match="--record-dir"):
        build_server(args)


def test_the_cli_builds_a_mirrored_server(tmp_path: Path) -> None:
    url = unique("memory")
    args = build_parser().parse_args(
        ["serve", "--upstream-url", "http://u/v1", "--record-dir", str(tmp_path), "--record-mirror", url]
    )
    assert build_server(args).mirror.url == url


async def captured(tmp_path: Path, url: str, config: dict | None) -> tuple[str, CaptureService, FinishResult]:
    """One token-mode trajectory, with all three sidecars, written and mirrored. Returns its id and finish."""
    engine = TestServer(MockEngine().app())
    await engine.start_server()
    service = CaptureService(
        str(engine.make_url("")).rstrip("/"),
        mode="tokens",
        renderer=FakeRenderer(),
        record_dir=str(tmp_path / "record"),
        record_mirror=url,
        record_mirror_config=config,
        host="127.0.0.1",
    )
    served = await asyncio.to_thread(service.start)
    try:
        async with CapturePool([served]) as pool, pool.trajectory({}) as trajectory:
            await converse(client(trajectory.base_url), "q", "r")
            result = await trajectory.finish({"reward": 1.0})
    finally:
        assert await asyncio.to_thread(service.stop)
        await engine.close()
    return trajectory.id, service, result


async def test_excluded_sidecars_stay_out_of_the_mirror_and_the_rest_reads_without_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = unique("memory")
    trajectory_id, service, result = await captured(tmp_path, url, {"exclude": ["experts", "sampling_mask"]})
    local = tmp_path / "record"

    fs = fsspec.filesystem("memory")
    names = sorted(path.rsplit("/", 1)[-1] for path in fs.ls(url.removeprefix("memory://"), detail=False))
    assert names == [f"{trajectory_id}.json.zst", f"{trajectory_id}.tokens.zst"]
    # finish lists the files the mirror holds, not the excluded ones the record directory also has.
    assert result.record.files == (f"{trajectory_id}.tokens.zst", f"{trajectory_id}.json.zst")
    assert len(record.record_files(local, trajectory_id)) == 4
    # Everything the mirror holds is a byte-for-byte copy, the document included.
    for name in names:
        assert remote_bytes(f"{url}/{name}") == (local / name).read_bytes()

    # Read back as a record directory, the copy lacks the sidecars its manifest lists: they read as absent.
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in names:
        (copy / name).write_bytes(remote_bytes(f"{url}/{name}"))
    assert set(record.read_document(copy, trajectory_id)["sidecars"]) == {"tokens", "experts", "sampling_mask"}
    with caplog.at_level(logging.WARNING, logger="skycap.record"):
        mirrored = record.load(copy, trajectory_id)
    tokens = [node.tokens for node in mirrored.graph if node.tokens is not None]
    assert tokens and all(t.token_ids and t.routed_experts is None and t.sampling_mask is None for t in tokens)
    assert "experts sidecar is missing" in caplog.text
    assert service.server.mirror.stats()["mirrored"] == 1


def test_a_record_without_its_tokens_sidecar_reads_as_text_only(tmp_path: Path) -> None:
    from tests.test_record import _token_trajectory

    trajectory = _token_trajectory()
    record.write(tmp_path, trajectory)
    record.sidecar_path(tmp_path, trajectory.id, "tokens").unlink()
    loaded = record.load(tmp_path, trajectory.id)
    assert [node.tokens for node in loaded.graph] == [None] * len(loaded.graph)
    assert [node.message for node in loaded.graph] == [node.message for node in trajectory.graph]


def test_exclude_names_known_sidecar_kinds() -> None:
    with pytest.raises(ValueError, match="unknown sidecar kinds \\['routes'\\]"):
        RecordMirror(unique("memory"), exclude=["experts", "routes"])
    with pytest.raises(TypeError, match="not one string"):
        RecordMirror(unique("memory"), exclude="experts")


def test_the_cli_takes_the_whole_mirror_config_as_json(tmp_path: Path) -> None:
    url = unique("memory")
    config = '{"exclude": ["experts", "sampling_mask"], "timeout": 120, "workers": 2}'
    base = ["serve", "--upstream-url", "http://u/v1", "--record-dir", str(tmp_path)]
    mirror = build_server(
        build_parser().parse_args([*base, "--record-mirror", url, "--record-mirror-config", config])
    ).mirror
    assert (mirror.exclude, mirror.timeout, mirror.workers) == (frozenset({"experts", "sampling_mask"}), 120, 2)

    with pytest.raises(SystemExit, match="needs --record-mirror"):
        build_server(build_parser().parse_args([*base, "--record-mirror-config", config]))
    with pytest.raises(SystemExit, match="JSON object"):
        build_server(build_parser().parse_args([*base, "--record-mirror", url, "--record-mirror-config", "[1]"]))
    with pytest.raises(TypeError):
        build_server(
            build_parser().parse_args([*base, "--record-mirror", url, "--record-mirror-config", '{"nope": 1}'])
        )


def test_a_config_goes_with_a_url_not_a_built_mirror(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="record_mirror_config"):
        CaptureServer(
            TextBackend("http://upstream/v1"),
            record_dir=tmp_path,
            record_mirror=RecordMirror(unique("memory")),
            record_mirror_config={"exclude": ["experts"]},
        )


def test_finish_says_which_machine_the_record_is_on(tmp_path: Path) -> None:
    backend = TextBackend("http://upstream/v1")
    assert CaptureServer(backend, record_dir=tmp_path, record_host="10.0.0.5").record_host == "10.0.0.5"
    assert CaptureServer(backend, record_dir=tmp_path).record_host == node_address()
    assert CaptureServer(backend).record_host is None
    # A service reports the address it is reached at, unless that is only loopback.
    assert (
        CaptureService("http://u/v1", record_dir=str(tmp_path), advertise_host="10.0.0.7").server.record_host
        == "10.0.0.7"
    )
    assert CaptureService("http://u/v1", record_dir=str(tmp_path)).server.record_host == node_address()
    args = build_parser().parse_args(
        ["serve", "--upstream-url", "http://u/v1", "--record-dir", str(tmp_path), "--record-host", "10.0.0.9"]
    )
    assert build_server(args).record_host == "10.0.0.9"


def test_a_record_location_names_its_host_the_way_scp_does() -> None:
    assert RecordLocation(path="/r/tr.json.zst", host="10.0.0.5").local == "10.0.0.5:/r/tr.json.zst"
    assert RecordLocation(path="/r/tr.json.zst", host="fe80::1").local == "[fe80::1]:/r/tr.json.zst"
    assert (
        RecordLocation(path="/r/tr.json.zst", host="10.0.0.5", mirror="s3://b/tr.json.zst").uri == "s3://b/tr.json.zst"
    )
    # A server from before `host` sends none.
    old = RecordLocation.from_json({"path": "/r/tr.json.zst", "mirror": None})
    assert old.host is None and old.local == "/r/tr.json.zst" and old.files == ()
