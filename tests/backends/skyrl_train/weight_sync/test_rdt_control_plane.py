"""Unit tests for the synchronous sharded-RDT control-plane client.

Covers the wire payloads (per-server replica_rank fan-out + uniform calls), the
body-aware error surfacing, and — most importantly — that the per-call fan-out is
*concurrent*, which is a correctness requirement (serial update_weights would
deadlock the producer's ref-counted group free), not just a perf choice.
"""

import threading

import pytest

from skyrl.backends.skyrl_train.inference_servers.rdt_control_protocol import (
    COLLECTIVE_RPC_ENDPOINT,
    RDT_FINISH_METHOD,
    RDT_INIT_METHOD,
    RDT_START_METHOD,
    RDT_UPDATE_METHOD,
)
from skyrl.backends.skyrl_train.weight_sync.rdt_control_plane import (
    SyncRdtControlPlaneClient,
)


class _FakeResponse:
    def __init__(self, status_code=200, body=None, text="", reason="OK"):
        self.status_code = status_code
        self._body = body
        self.text = text
        self.reason = reason

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _RecordingSession:
    """Stand-in for requests.Session that records every POST.

    ``barrier`` (optional) is entered inside each POST so a test can assert the
    calls actually run concurrently: if the client issued them serially the
    barrier would never fill and time out.
    """

    def __init__(self, barrier=None, status_for=None):
        self.headers = {}
        self.calls = []  # (url, payload)
        self.closed = False
        self._lock = threading.Lock()
        self._barrier = barrier
        self._status_for = status_for or {}

    def post(self, url, json=None, timeout=None):
        # The client POSTs to "{base}/collective_rpc"; record/key on the base url.
        base = url[: -len(COLLECTIVE_RPC_ENDPOINT)] if url.endswith(COLLECTIVE_RPC_ENDPOINT) else url
        if self._barrier is not None:
            self._barrier.wait(timeout=5)
        with self._lock:
            self.calls.append((base, json))
        status = self._status_for.get(base, 200)
        if status >= 400:
            return _FakeResponse(
                status_code=status, body={"error": {"message": "boom on " + base}}, reason="Server Error"
            )
        return _FakeResponse()

    def close(self):
        self.closed = True


def _client(urls, data_parallel_size=1, session=None):
    c = SyncRdtControlPlaneClient(urls, data_parallel_size)
    if session is not None:
        c._session = session  # swap the real requests.Session for the fake
    return c


def _init_infos(session):
    """Extract per-url init_info dicts from recorded init calls."""
    out = {}
    for url, payload in session.calls:
        assert payload["method"] == RDT_INIT_METHOD
        out[url] = payload["kwargs"]["init_info"]
    return out


def test_connection_close_header_set():
    c = _client(["http://a"])
    try:
        assert c._session.headers.get("Connection") == "close"
    finally:
        c.close()


def test_init_fans_out_per_server_replica_rank():
    urls = ["http://a", "http://b", "http://c"]
    sess = _RecordingSession()
    c = _client(urls, data_parallel_size=1, session=sess)
    try:
        c.init_weight_transfer_engine({"num_consumers": 3, "names": ["w"]})
    finally:
        c.close()
    infos = _init_infos(sess)
    assert set(infos) == set(urls)
    # dp=1 -> one replica per server, distinct ranks 0..N-1, num_replicas=N.
    assert sorted(i["replica_rank"] for i in infos.values()) == [0, 1, 2]
    assert all(i["num_replicas"] == 3 for i in infos.values())
    # Shared fields untouched on every server.
    assert all(i["num_consumers"] == 3 and i["names"] == ["w"] for i in infos.values())


def test_init_replica_rank_is_per_deployment_under_dp():
    # 4 servers, dp=2 -> two deployments; the dp servers of a deployment share
    # one replica_rank (server_index // dp), so ranks are [0, 0, 1, 1].
    urls = ["http://a", "http://b", "http://c", "http://d"]
    sess = _RecordingSession()
    c = _client(urls, data_parallel_size=2, session=sess)
    try:
        c.init_weight_transfer_engine({"k": "v"})
    finally:
        c.close()
    infos = _init_infos(sess)
    assert [infos[u]["replica_rank"] for u in urls] == [0, 0, 1, 1]
    assert all(i["num_replicas"] == 2 for i in infos.values())


@pytest.mark.parametrize(
    "call, method, expect_kwargs",
    [
        (lambda c: c.start_weight_update(), RDT_START_METHOD, {"is_checkpoint_format": True}),
        (lambda c: c.update_weights({"names": ["x"]}), RDT_UPDATE_METHOD, {"update_info": {"names": ["x"]}}),
        (lambda c: c.finish_weight_update(), RDT_FINISH_METHOD, None),
    ],
)
def test_uniform_calls_hit_every_server(call, method, expect_kwargs):
    urls = ["http://a", "http://b"]
    sess = _RecordingSession()
    c = _client(urls, session=sess)
    try:
        call(c)
    finally:
        c.close()
    assert {u for u, _ in sess.calls} == set(urls)
    for _, payload in sess.calls:
        assert payload["method"] == method
        if expect_kwargs is None:
            assert "kwargs" not in payload
        else:
            assert payload["kwargs"] == expect_kwargs


def test_fanout_is_concurrent():
    # If the client issued the POSTs serially, the barrier for N servers would
    # never fill and each wait() would time out -> BrokenBarrierError. Reaching
    # the barrier from all workers proves concurrent issue.
    urls = ["http://a", "http://b", "http://c", "http://d"]
    barrier = threading.Barrier(len(urls))
    sess = _RecordingSession(barrier=barrier)
    c = _client(urls, session=sess)
    try:
        c.update_weights({"names": []})  # must not raise (barrier released)
    finally:
        c.close()
    assert len(sess.calls) == len(urls)


def test_error_surfaces_body_message_and_drains_all():
    urls = ["http://a", "http://b", "http://c"]
    sess = _RecordingSession(status_for={"http://b": 500})
    c = _client(urls, session=sess)
    try:
        with pytest.raises(RuntimeError) as ei:
            c.update_weights({"names": []})
    finally:
        c.close()
    # Body error detail is surfaced, not the bare reason phrase.
    assert "boom on http://b" in str(ei.value)
    # All servers were still called (failure drains the whole fan-out).
    assert {u for u, _ in sess.calls} == set(urls)


def test_close_releases_session():
    sess = _RecordingSession()
    c = _client(["http://a"], session=sess)
    c.close()
    assert sess.closed is True


def test_empty_server_urls_rejected():
    with pytest.raises(ValueError):
        SyncRdtControlPlaneClient([], 1)
