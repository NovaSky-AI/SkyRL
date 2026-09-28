"""The ownership rules, enforced mechanically.

`docs/design/refactor_2026-09-19.md` states them as constraints, not
preferences: `Runtime` is assembled last and injected nowhere; a constructor
returns a working object; a component gets its collaborators and does not reach
through one to find another. Each of those is the kind of rule that holds until
the day somebody is in a hurry, so each is a test.

Two kinds of check. The import-layer checks read the source: nothing outside
composition imports `Runtime`, and the exact shapes the plan lists as "absent"
after Phase 1 are absent. The construction checks build the real components
with fakes for their collaborators -- no runtime, no data plane, no FastAPI, no
disk, no server -- which is only possible if the boundaries are real.
"""

from __future__ import annotations

import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "skyrl_capture"

# Modules that legitimately know what a `Runtime` is: the one that builds it,
# the lifespan that starts it, the process that serves it, and the benchmark
# harness, which is development tooling and reaches into everything.
COMPOSITION = {"runtime.py", "application.py", "service.py", "bench/harness.py"}


def _sources() -> dict[str, str]:
    return {
        str(path.relative_to(SRC)): path.read_text()
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts
    }


# -- import layer ---------------------------------------------------------------
def test_runtime_is_imported_only_by_composition():
    offenders = [
        name
        for name, text in _sources().items()
        if name not in COMPOSITION
        and re.search(r"^\s*(from skyrl_capture\.runtime import|import skyrl_capture\.runtime)", text, re.M)
    ]
    assert not offenders, f"these modules import the runtime: {offenders}"


@pytest.mark.parametrize(
    "shape",
    [
        r"TokenService\(",
        r"ExportRunner\(",
        r"build_control_app\(\s*runtime",
        r"runtime\.tokens\s*=",
        r"runtime\.exports\s*=",
        r"data_plane\._tokens\s*=",
        r"RecordRuntime",
        r"self\._runtime\.data_plane",
        r"self\._runtime\.config",
        # Phase 2/3: the per-resource store modules, the database facade and
        # the ingestion-side graph index are replaced by the domain.
        r"skyrl_capture\.(store|db)\b",
        r"CaptureState\(",
        r"MemoryDatabase\(",
        r"GraphIndex\(",
        r"require_state\(",
        # Phases 4-7: the spool, the ingestion worker, the acceptance ledger,
        # the lease cache and the finished-record writer are replaced by the
        # recorder, the sink and the log.
        r"skyrl_capture\.(spool|ingest)\b",
        r"CaptureIntake\(",
        r"AcceptanceLedger\(",
        r"IngestWorker\(",
        r"SpoolWriter\(",
        r"CaptureRing\(",
        r"RecordWriter\(",
        r"RecordState\(",
        r"LeaseCache\(",
        r"put_compressed\(",
        r"payload_uri",
        # One mode, no leases: a trajectory has no credential, the request
        # path resolves an id, and nothing chooses between two proxies.
        r"LeaseRegistry",
        r"LeaseRecord",
        r"lookup_cached",
        r"lease_for",
        r"active_leases",
        r"lease_by_trajectory",
        r"token_hash",
        r"secret_token",
        r"hash_token",
        r"UpstreamConfig",
        # The trajectory's own mode never decides which code runs.
        r"lease\.mode",
        r"trajectory\.mode\s*==",
        r"\.mode\s*==\s*[\"']tokens[\"']\s*and",
        # One adapter per text protocol: no shared wire object, no base class
        # spanning both modes, and no central parser that has to know every
        # provider before any of them works.
        r"WireFormat",
        r"BaseUpstreamServer",
        r"UpstreamServer\b",
        r"parse_exchange",
        r"capture\.parse\b",
        r"\.wire\b",
        r"token_request\(|token_response\(|token_url\(",
        r"endpoint_kind\s*==\s*[\"']messages[\"']",
        r"registry\.get\((?:envelope|observed)\.",
        # An upstream setting that arrives per trajectory. Deployment and
        # credentials are process configuration; per-inference settings ride
        # in the inference request.
        r"\boverrides\s*[:=]\s*(dict|\{)",
        r"context\.overrides",
        r"upstream_overrides",
        # Retired package names: the package tree itself states the planes,
        # modes, and read/write direction.
        r"skyrl_capture\.(core|read|control|proxy|tokens)\b",
        # Per-trajectory persistence: no global event log, no global reducer,
        # no process-wide recorder, no sink, and nothing that replays a whole
        # record at startup.
        r"\bLiveState\b",
        r"\bRecorder\b",
        r"\bCaptureSink\b",
        r"\bDiskSink\b",
        r"\bDiskReader\b",
        r"\bDiskStateReader\b",
        r"\bLiveStateReader\b",
        r"\bCaptureEvent\b",
        r"skyrl_capture\.domain\.(state|events)\b",
        r"skyrl_capture\.reader\.(live|disk)\b",
        r"segment_bytes|segment_paths|scan_segment",
        r"\bSinkCheckpoint\b",
        r"replay_into\(",
        # Nothing expires a trajectory, and nothing deletes one over the API.
        r"\bexpires_at\b",
        r"TRAJECTORY_TTL_SECONDS",
        r"expire_abandoned|TrajectoryDeleted|TrajectoryFinalized",
        r"def delete_trajectory|annotate_run|RunMetadataUpdated|RunStarted",
        # Memory-only capture is gone: there is no branch for "no record".
        r"if config\.record_dir is not None",
    ],
)
def test_the_shapes_the_plan_lists_as_absent_are_absent(shape):
    pattern = re.compile(shape)
    offenders = [name for name, text in _sources().items() if pattern.search(text)]
    assert not offenders, f"{shape!r} still appears in {offenders}"


# What the domain may import from the rest of the package: identifiers,
# version constants and the route prefix. Nothing that serves, stores, or
# composes -- the domain is what those are built around.
DOMAIN_MAY_IMPORT = {"skyrl_capture.domain", "skyrl_capture.ids", "skyrl_capture.version", "skyrl_capture.routes"}


def test_the_domain_imports_nothing_above_it():
    offenders = []
    for name, text in _sources().items():
        if not name.startswith("domain/"):
            continue
        for module in re.findall(r"^\s*(?:from|import)\s+(skyrl_capture(?:\.\w+)*)", text, re.M):
            if not any(module == allowed or module.startswith(allowed + ".") for allowed in DOMAIN_MAY_IMPORT):
                offenders.append((name, module))
    assert not offenders, f"the domain reaches upward: {offenders}"


def test_the_data_plane_and_modes_do_not_own_each_other():
    """Composition chooses a mode; the shared plane and the modes do not."""
    forbidden = {
        "data_plane/": ("skyrl_capture.text", "skyrl_capture.tito"),
        "text/": ("skyrl_capture.tito",),
        "tito/": ("skyrl_capture.text",),
    }
    offenders = []
    for name, source in _sources().items():
        for prefix, imports in forbidden.items():
            if name.startswith(prefix):
                offenders.extend((name, module) for module in imports if module in source)
    assert not offenders, f"mode ownership leaked across package boundaries: {offenders}"


def test_only_the_coordinator_and_the_lifecycle_append_to_a_journal():
    """Two writers, and they are the two the design names.

    Background appends go through the commit coordinator, which is what keeps
    them ordered per trajectory and bounded. The lifecycle commands append
    directly, because a create, a finish or a metadata edit is not background
    work -- the caller is waiting for it to be durable. A third caller would be
    a third ordering.
    """
    allowed = {"writer/commits.py", "control_plane/commands.py", "persistence/active.py"}
    offenders = [
        name
        for name, text in _sources().items()
        if name not in allowed and re.search(r"\.append\(\s*trajectory_id|_active\.append\(", text)
    ]
    assert not offenders, f"these modules append to a journal themselves: {offenders}"


def test_the_viewer_reads_files_and_not_a_proxys_memory():
    """The product guarantee, as an import rule.

    A viewer that could reach a hot aggregate would show a graph a crash would
    erase. It reads the record directory, so it may hold stores and a reader
    and nothing from the write path.
    """
    forbidden = ("skyrl_capture.writer", "skyrl_capture.text", "skyrl_capture.tito")
    offenders = [
        (name, module)
        for name, text in _sources().items()
        if name in ("reader/records.py", "control_plane/viewer.py")
        for module in forbidden
        if module in text
    ]
    assert not offenders, f"the viewer reaches into the write path: {offenders}"


def test_no_component_takes_a_runtime():
    """A parameter named `runtime` on anything but composition is the cycle
    coming back under its own name."""
    offenders = []
    for name, text in _sources().items():
        if name in COMPOSITION:
            continue
        if re.search(r"def __init__\([^)]*\bruntime\b", text) or re.search(
            r"^def \w+\(\s*runtime\b", text, re.M
        ):
            offenders.append(name)
    assert not offenders, f"these take a runtime: {offenders}"


# -- construction ----------------------------------------------------------------
def _aggregate(trajectory_id: str = "tr_x", *, mode: str = "text"):
    """A hot aggregate, with no credential in it and nothing on disk."""
    from skyrl_capture.domain.records import ActiveTrajectory, TrajectoryHeader

    return ActiveTrajectory.create(
        TrajectoryHeader(
            id=trajectory_id, project="p", run_id=None, task_id=None, step=None,
            mode=mode, upstream={}, labels=(), annotations={}, bodies="full",
            source_metadata={}, created_at=datetime.now(UTC), create_request_hash="h",
        )
    )


class _Journal:
    """An `ActiveStore` that keeps records in a list. No disk, no fsync."""

    def __init__(self) -> None:
        self.records: list = []

    async def create(self, header) -> None:
        self.records.append(header)

    async def append(self, trajectory_id: str, record) -> None:
        self.records.append(record)

    async def recover(self, trajectory_id: str):
        return None

    async def remove(self, trajectory_id: str) -> None:
        return None


class _Traces:
    """A `TraceSource` that has nothing stored: every trace starts empty."""

    async def load(self, trajectory_id: str):
        from skyrl_capture.tito.trace import TokenTrace

        return TokenTrace(trajectory_id)


class _Engine:
    """A token engine whose completion is fixed, so the turn is deterministic."""

    async def generate(self, **kwargs):
        prompt = list(kwargs["prompt_token_ids"])
        return {
            "completion_ids": [7, 8, 9],
            "completion_logprobs": [-0.1, -0.2, -0.3],
            "stop_reason": "stop",
            "routed_experts": None,
            "prompt_token_ids": prompt,
        }


async def test_a_token_turn_runs_with_no_runtime_data_plane_or_server():
    """The token path, end to end, with a fake engine and a journal in a list.

    If this needed a `Runtime`, the cycle would be back; if it needed a disk,
    the store boundary would not be real.
    """
    from skyrl_capture.config import TitoUpstream
    from skyrl_capture.tito.proxy import TitoProxy
    from skyrl_capture.tito.sessions import TokenSessionManager
    from skyrl_capture.writer.commits import CommitCoordinator

    upstream = TitoUpstream(type="tokens", url="http://engine/generate", tokenizer="builtin", model="m")
    journal = _Journal()
    commits = CommitCoordinator(journal, capacity=8)
    active = _aggregate(mode="tokens")
    sessions = TokenSessionManager(upstream=upstream, trace_budget=0, source=_Traces())
    proxy = TitoProxy(
        engine=_Engine(), sessions=sessions, upstream=upstream, commits=commits,
        header_allowlist=("content-type",), clock_epoch="e",
    )
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"method": "POST", "headers": [(b"content-type", b"application/json")], "query_string": b""}
    body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}'
    await proxy.handle(
        scope=scope, send=send, active=active, body=body,
        request_start_wall=0, request_start_mono=0, suffix="/chat/completions",
    )

    assert sent[0]["status"] == 200
    assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    assert proxy.turns_served == 1
    assert len(active.exchanges) == 1, "one exchange reached the aggregate"
    assert len(active.graph) == 2
    trace = await sessions.trace_for("tr_x")
    assert len(trace.nodes()) == 2, "user and assistant nodes were committed"

    # The response could not close until its exchange was durable, and the
    # delivery record is what says the send then completed -- queued after the
    # close, so it lands a moment later.
    await commits.settle(active)
    kinds = [type(record).__name__ for record in journal.records]
    assert kinds == ["ExchangeCommitted", "ExchangeDeliveryConfirmed"], kinds


async def test_the_data_plane_has_one_handler_and_gates_on_the_trajectory():
    """It resolves the id in the route and calls the one proxy it was given.

    No credential is read, no mode is inspected, and there is no second
    handler to choose between -- which is why this constructs with a fake
    proxy and a fake registry and nothing else.
    """
    from skyrl_capture.data_plane.app import DataPlane

    calls: list[str] = []

    class Handler:
        async def handle(self, *, send, active, **kwargs) -> None:
            calls.append(active.id)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        def error_body(self, status: int, message: str) -> bytes:
            return b'{"error": {}}'

    open_one = _aggregate("tr_x")

    class Registry:
        """Hot for one trajectory, holding a committed record for another."""

        def hot(self, trajectory_id):
            return open_one if trajectory_id == "tr_x" else None

        async def resolve(self, trajectory_id):
            return self.hot(trajectory_id)

        async def is_committed(self, trajectory_id):
            return trajectory_id == "tr_done"

    plane = DataPlane(registry=Registry(), proxy=Handler(), max_request_bytes=1024)

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def serve(path: str) -> int:
        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        await plane(
            {"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""},
            receive,
            send,
        )
        return sent[0]["status"]

    assert await serve("/route/tr_x/chat/completions") == 200
    assert await serve("/route/tr_done/chat/completions") == 410, "found, and closed"
    assert await serve("/route/tr_nope/chat/completions") == 404, "never existed"
    assert calls == ["tr_x"], "only the live one reached the proxy"
    assert plane.requests_served == 1
    assert open_one.in_flight == 0, "the turn was counted open and closed"


def test_the_viewer_app_builds_over_an_empty_record_with_no_runtime():
    """Read routes over a record directory nothing has written to.

    No capture process, no proxy, no registry -- which is what makes a
    standalone viewer, and `--disable-viewer` on every other replica, possible.
    """
    from fastapi.testclient import TestClient

    from skyrl_capture.control_plane.viewer import record_viewer_app
    from skyrl_capture.persistence import ensure_record

    root = Path(tempfile.mkdtemp(prefix="viewer-boundary-")) / "record"
    ensure_record(root)
    app = record_viewer_app(str(root))
    with TestClient(app) as client:
        assert client.get("/v1/trajectories").json()["data"] == []
        assert client.get("/healthz").json()["source"] == "record"
        # Lifecycle is not here. Creating a trajectory through the viewer is
        # not a route that exists.
        assert client.post("/v1/trajectories", json={"project": "p"}).status_code in (404, 405)
        assert client.get("/v1/exports/exp_nope").status_code == 404


# -- one mode, chosen once -------------------------------------------------------
async def _runtime_for(upstream):
    """A built runtime, with nothing started. Composition is what is under test."""
    from skyrl_capture.config import Config, ProxyConfig
    from skyrl_capture.runtime import build_runtime

    root = Path(tempfile.mkdtemp(prefix="boundary-record-")) / "record"
    return await build_runtime(
        Config(
            upstream=upstream,
            record_dir=root,
            # Composition is what is under test; a viewer would only add an
            # indexer to tear down.
            viewer=False,
            proxy=ProxyConfig(host="127.0.0.1", port=0),
        )
    )


async def test_a_text_runtime_builds_no_token_machinery():
    """The claim the mode split is for: text mode is a proxy and a recorder.

    Not "the token path is idle" -- absent. A renderer is seconds of startup
    and a `transformers` dependency this install may not have, and a process
    that cannot capture tokens must not pay for either.
    """
    import sys

    from skyrl_capture.config import TextUpstream
    from skyrl_capture.text.proxy import TextProxy

    before = {name for name in sys.modules if "renderer" in name or "transformers" in name}
    runtime = await _runtime_for(TextUpstream(type="openai", url="http://127.0.0.1:1/v1"))
    try:
        assert isinstance(runtime.proxy, TextProxy)
        assert not hasattr(runtime, "decoder")
        assert runtime.proxy_tasks == (), "no token loop-lag sampler"
        assert not hasattr(runtime, "sessions"), "no session manager to hold"
        assert not hasattr(runtime.proxy, "sessions")
        # And nothing it constructed reached for a tokenizer.
        after = {name for name in sys.modules if "renderer" in name or "transformers" in name}
        assert after == before, f"text mode imported {after - before}"
    finally:
        await runtime.stop()


async def test_a_token_runtime_builds_no_text_proxy():
    """The other half: token capture renders every turn itself, so there is no
    text path to fall back to when an endpoint is not chat completions."""
    from skyrl_capture.config import TitoUpstream
    from skyrl_capture.text.proxy import TextProxy

    runtime = await _runtime_for(
        TitoUpstream(url="http://127.0.0.1:1/generate", tokenizer="builtin", model="m")
    )
    try:
        assert not isinstance(runtime.proxy, TextProxy)
        assert not hasattr(runtime, "decoder")
        assert runtime.proxy.sessions is not None
        assert len(runtime.proxy_tasks) == 1, "the loop-lag sampler, and only in this mode"
    finally:
        await runtime.stop()


async def test_the_data_plane_is_given_exactly_one_handler():
    from skyrl_capture.config import TextUpstream

    runtime = await _runtime_for(TextUpstream(type="openai", url="http://127.0.0.1:1/v1"))
    try:
        assert runtime.data_plane._proxy is runtime.proxy  # noqa: SLF001
        assert not hasattr(runtime.data_plane, "_text")
        assert not hasattr(runtime.data_plane, "_tokens")
        assert not hasattr(runtime.data_plane, "_leases")
    finally:
        await runtime.stop()


async def test_a_token_route_refuses_an_endpoint_it_cannot_render():
    """No text fallback: an endpoint this mode does not render is a mistake to
    report, not something to forward blind."""
    from skyrl_capture.config import TitoUpstream

    runtime = await _runtime_for(
        TitoUpstream(url="http://127.0.0.1:1/generate", tokenizer="builtin", model="m")
    )
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    try:
        await runtime.registry.create(_aggregate("tr_x", mode="tokens").header)
        await runtime.data_plane(
            {"type": "http", "method": "POST", "path": "/route/tr_x/embeddings",
             "headers": [], "query_string": b""},
            receive,
            send,
        )
    finally:
        await runtime.stop()
    assert sent[0]["status"] == 404
    assert b"unsupported_endpoint" in sent[1]["body"]


def test_the_forward_path_does_not_import_an_http_client_library():
    """The measurement that made this the only transport, kept true.

    `httpx` is still a dependency -- the public Python SDK is an httpx client --
    but it has no business in the code that runs per inference request. An
    import here is how the 3x comes back.
    """
    offenders = [
        name
        for name, text in _sources().items()
        if name.startswith(("data_plane/", "text/", "tito/", "transport/", "writer/"))
        and re.search(r"^\s*(import httpx|from httpx import)", text, re.M)
    ]
    assert not offenders, f"the forward path imports httpx: {offenders}"


# -- the storage interfaces are the whole contract ---------------------------------
class _MemoryActive:
    """Exactly `ActiveStore`, and deliberately nothing else.

    No `path_for`, no `stats`, no `close`. Anything the write path reaches for
    beyond the five declared operations is an `AttributeError` here rather than
    a surprise for whoever implements this against something that is not a
    filesystem.
    """

    def __init__(self) -> None:
        self.journals: dict[str, list] = {}

    async def create(self, header) -> None:
        from skyrl_capture.persistence import journal

        self.journals[header.id] = [journal.TrajectoryCreated(header)]

    async def append(self, trajectory_id: str, record) -> None:
        self.journals.setdefault(trajectory_id, []).append(record)

    async def recover(self, trajectory_id: str):
        from skyrl_capture.persistence.active import rebuild

        records = self.journals.get(trajectory_id)
        return rebuild(records, recovered=True) if records else None

    async def records(self, trajectory_id: str) -> list:
        return list(self.journals.get(trajectory_id, ()))

    async def remove(self, trajectory_id: str) -> None:
        self.journals.pop(trajectory_id, None)


class _MemoryCommitted:
    """Exactly `CommittedStore`, and deliberately nothing else."""

    def __init__(self) -> None:
        self.records: dict[str, object] = {}

    async def put(self, record) -> None:
        self.records[record.id] = record

    async def get(self, trajectory_id: str):
        return self.records.get(trajectory_id)

    async def exists(self, trajectory_id: str) -> bool:
        return trajectory_id in self.records

    async def update_metadata(self, trajectory_id: str, update):
        record = self.records[trajectory_id]  # KeyError, as the disk store raises
        updated = record.with_metadata(update)
        self.records[trajectory_id] = updated
        return updated


async def test_the_lifecycle_runs_on_the_declared_interfaces_alone():
    """Create, capture, finish and annotate, against two dictionaries.

    The point is not that an in-memory store works -- it is that the write path
    cannot reach past what `ActiveStore` and `CommittedStore` declare. Both
    protocols have been incomplete before: `CommittedStore.exists` was called
    and undeclared, and `finish` read a journal through an `ActiveStore` by
    asking it for a *filesystem path*, which nothing but the disk store could
    ever have given it. A PostgreSQL implementation would have satisfied both
    interfaces and failed at runtime.

    This is what would have caught them, and what will catch the next one.
    """
    from datetime import UTC, datetime

    from test_aggregate import exchange

    from skyrl_capture.config import Config, ProxyConfig, TextUpstream
    from skyrl_capture.control_plane.commands import CaptureCommands
    from skyrl_capture.persistence import journal
    from skyrl_capture.writer.commits import CommitCoordinator
    from skyrl_capture.writer.registry import TrajectoryRegistry

    at = datetime.now(UTC)
    active, committed = _MemoryActive(), _MemoryCommitted()
    registry = TrajectoryRegistry(active=active, committed=committed)
    commits = CommitCoordinator(active)
    commands = CaptureCommands(
        registry=registry,
        commits=commits,
        committed=committed,
        config=Config(
            upstream=TextUpstream(type="openai", url="http://127.0.0.1:1/v1"),
            proxy=ProxyConfig(public_url="http://capture"),
        ),
    )

    created = await commands.create(
        trajectory_id="tr_x", project="p", run_id="r", task_id="t", step=0,
        labels=[], annotations={}, bodies="full", source_metadata={},
        request_hash="create-hash",
    )
    assert created.base_url == "http://capture/route/tr_x/v1"

    # One captured turn, committed the way a proxy commits one.
    hot = registry.hot("tr_x")
    captured, delta = exchange("ex_0", sequence=0, trajectory_id="tr_x")
    hot.add_exchange(captured, delta, at)
    await commits.submit(hot, journal.ExchangeCommitted(captured, delta, at))

    envelope = await commands.finish(
        "tr_x", labels=["done"], annotations={"reward": 1.0}, command_result="success"
    )
    assert envelope.status == "finished"
    assert envelope.records, "and it rendered from what it committed"

    stored = await committed.get("tr_x")
    assert len(stored.exchanges) == 1
    assert stored.trajectory.annotations == {"reward": 1.0}

    # Metadata after finishing, which is the other side of the committed store.
    metadata = await commands.annotate(
        "tr_x", annotations={"grader": "rubric"}, remove_annotations=None,
        labels=None, remove_labels=None,
    )
    assert metadata["annotations"] == {"reward": 1.0, "grader": "rubric"}
    assert (await committed.get("tr_x")).revision == 2


def test_the_disk_stores_satisfy_the_interfaces_they_are_used_through():
    """The other direction: every declared operation is actually implemented."""
    import inspect

    from skyrl_capture.persistence.active import ActiveStore, DiskActiveStore
    from skyrl_capture.persistence.committed import CommittedStore, DiskCommittedStore

    for protocol, implementation in (
        (ActiveStore, DiskActiveStore),
        (CommittedStore, DiskCommittedStore),
    ):
        declared = {
            name
            for name, value in vars(protocol).items()
            if not name.startswith("_") and inspect.isfunction(value)
        }
        assert declared, protocol
        missing = [name for name in declared if not hasattr(implementation, name)]
        assert not missing, f"{implementation.__name__} is missing {missing}"
