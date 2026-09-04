"""Tests for the trainer-side weight-sync control plane.

The per-server init payload rewrites are what matter: both fail
silently-then-fatally when wrong. A bad NCCL ``rank_offset`` mis-maps ranks and
hangs in the rendezvous rather than erroring; a bad RDT ``replica_rank`` collides
two deployments' consumer id blocks.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pytest

from skyrl.backends.skyrl_train.weight_sync.control_plane import (
    FINISH_UPDATE_ENDPOINT,
    INIT_ENGINE_ENDPOINT,
    START_UPDATE_ENDPOINT,
    UPDATE_WEIGHTS_ENDPOINT,
    SkyrlWeightSyncClient,
    nccl_init_payloads,
    rdt_init_payloads,
)


class _NotJsonable:
    """Stands in for the storage type / handle bytes inside reduce_tensor's args."""

    def __eq__(self, other):
        return isinstance(other, _NotJsonable)


class _Resp:
    def __init__(self, status_code: int = 200, body: Any = None, reason: str = "OK") -> None:
        self.status_code = status_code
        self.reason = reason
        self._body = body if body is not None else {"status": "ok"}
        self.text = str(self._body)

    def json(self):
        return self._body


class _FakeSession:
    """Records every POST; optionally fails for one URL."""

    def __init__(self, fail_url: str = None, fail_body: Any = None) -> None:
        self.headers: Dict[str, str] = {}
        self.calls: List[tuple] = []
        self.closed = False
        self._fail_url = fail_url
        self._fail_body = fail_body

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self._fail_url is not None and url.startswith(self._fail_url):
            return _Resp(500, self._fail_body, reason="Internal Server Error")
        return _Resp()

    def close(self):
        self.closed = True


@pytest.fixture
def make_client(monkeypatch):
    """Build a client whose HTTP session is a recording fake."""

    def _make(urls, dp=1, init_payload_fn=None, fail_url=None, fail_body=None):
        client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
        client._urls = list(urls)
        client._dp = max(1, dp)
        client._init_payload_fn = init_payload_fn
        client._session = _FakeSession(fail_url=fail_url, fail_body=fail_body)
        client._pool = ThreadPoolExecutor(max_workers=len(urls))
        return client

    return _make


class TestNcclInitPayloads:
    """``rank_offset`` advances one deployment's worth per deployment, and stays
    put across the DP servers within a deployment."""

    def test_single_deployment_keeps_the_engines_offset(self):
        init = {"master_address": "h", "master_port": 1, "rank_offset": 1, "world_size": 5}
        payloads = nccl_init_payloads(init, ["u0"], 1)
        assert [p["rank_offset"] for p in payloads] == [1]
        # Everything else rides through untouched.
        assert payloads[0]["master_port"] == 1 and payloads[0]["world_size"] == 5

    def test_offset_advances_per_deployment(self):
        # 2 deployments x 4 workers each + the trainer sender = world_size 9.
        init = {"master_address": "h", "master_port": 1, "rank_offset": 1, "world_size": 9}
        payloads = nccl_init_payloads(init, ["u0", "u1"], 1)
        assert [p["rank_offset"] for p in payloads] == [1, 5]

    def test_dp_servers_of_one_deployment_share_an_offset(self):
        # 2 deployments x dp=2 servers = 4 servers; 4 workers per deployment.
        init = {"master_address": "h", "master_port": 1, "rank_offset": 1, "world_size": 9}
        payloads = nccl_init_payloads(init, ["u0", "u1", "u2", "u3"], 2)
        assert [p["rank_offset"] for p in payloads] == [1, 1, 5, 5]

    def test_world_size_that_does_not_divide_is_rejected(self):
        # 3 deployments cannot share 7 workers; this is the shape that would
        # otherwise mis-map ranks and hang in the rendezvous.
        init = {"master_address": "h", "master_port": 1, "rank_offset": 1, "world_size": 8}
        with pytest.raises(ValueError, match="does not divide"):
            nccl_init_payloads(init, ["u0", "u1", "u2"], 1)

    def test_no_workers_is_rejected(self):
        init = {"master_address": "h", "master_port": 1, "rank_offset": 1, "world_size": 1}
        with pytest.raises(ValueError, match="inference workers"):
            nccl_init_payloads(init, ["u0"], 1)


class TestRdtInitPayloads:
    def test_stamps_deployment_ordinal_and_count(self):
        payloads = rdt_init_payloads({"num_consumers": 8}, ["u0", "u1", "u2"], 1)
        assert [p["replica_rank"] for p in payloads] == [0, 1, 2]
        assert {p["num_replicas"] for p in payloads} == {3}
        assert {p["num_consumers"] for p in payloads} == {8}

    def test_dp_servers_share_a_replica_rank(self):
        payloads = rdt_init_payloads({}, ["u0", "u1", "u2", "u3"], 2)
        assert [p["replica_rank"] for p in payloads] == [0, 0, 1, 1]
        assert {p["num_replicas"] for p in payloads} == {2}


class TestClientFanout:
    def test_init_uses_the_rewrite_and_hits_every_server(self, make_client):
        client = make_client(["http://a", "http://b"], dp=1, init_payload_fn=rdt_init_payloads)
        client.init_weight_transfer_engine({"num_consumers": 4})
        calls = dict(client._session.calls)
        assert set(calls) == {f"http://a{INIT_ENGINE_ENDPOINT}", f"http://b{INIT_ENGINE_ENDPOINT}"}
        assert calls[f"http://a{INIT_ENGINE_ENDPOINT}"]["init_info"]["replica_rank"] == 0
        assert calls[f"http://b{INIT_ENGINE_ENDPOINT}"]["init_info"]["replica_rank"] == 1

    def test_init_without_a_rewrite_sends_the_same_dict_everywhere(self, make_client):
        client = make_client(["http://a", "http://b"])
        client.init_weight_transfer_engine({"packed": True})
        assert [body["init_info"] for _, body in client._session.calls] == [{"packed": True}] * 2

    def test_a_rewrite_returning_the_wrong_count_is_rejected(self, make_client):
        client = make_client(["http://a", "http://b"], init_payload_fn=lambda info, urls, dp: [dict(info)])
        with pytest.raises(ValueError, match="one per server"):
            client.init_weight_transfer_engine({})

    def test_lifecycle_endpoints(self, make_client):
        client = make_client(["http://a"])
        client.start_weight_update()
        client.update_weights({"names": ["w"]})
        client.finish_weight_update()
        assert [url for url, _ in client._session.calls] == [
            f"http://a{START_UPDATE_ENDPOINT}",
            f"http://a{UPDATE_WEIGHTS_ENDPOINT}",
            f"http://a{FINISH_UPDATE_ENDPOINT}",
        ]
        bodies = [body for _, body in client._session.calls]
        assert bodies[0] is None
        assert bodies[1] == {"update_info": {"names": ["w"]}}
        # weight_version is omitted entirely when unset, so the route's default applies.
        assert bodies[2] is None

    def test_ipc_handles_are_pickled_for_http(self, make_client):
        """``IPCTrainerWeightTransferEngine`` emits raw ipc_handles, which are not
        JSON-serializable; pickling them is this transport's job."""
        import json
        import pickle

        # A non-JSON-native payload standing in for reduce_tensor's args, which
        # carry storage *types* and raw handle bytes. Module-level so it pickles.
        handles = {"GPU-uuid": ("rebuild", _NotJsonable(), 7)}
        with pytest.raises(TypeError):
            json.dumps(handles)
        client = make_client(["http://a"])
        client.update_weights({"names": ["w"], "shapes": [[2, 2]], "ipc_handles": handles})

        body = client._session.calls[0][1]["update_info"]
        assert "ipc_handles" not in body
        assert pickle.loads(base64.b64decode(body["ipc_handles_pickled"])) == handles
        # Everything else rides through untouched.
        assert body["names"] == ["w"] and body["shapes"] == [[2, 2]]

    def test_non_ipc_update_info_passes_through_unchanged(self, make_client):
        client = make_client(["http://a"])
        update_info = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]]}
        client.update_weights(update_info)
        assert client._session.calls[0][1] == {"update_info": update_info}

    def test_finish_carries_a_weight_version_when_given(self, make_client):
        client = make_client(["http://a"])
        client.finish_weight_update("v7")
        assert client._session.calls[0][1] == {"weight_version": "v7"}

    def test_every_server_is_posted_even_when_one_fails(self, make_client):
        """A failure must not leave POSTs in flight against the other servers:
        every future is drained before the first exception is raised."""
        client = make_client(["http://a", "http://b", "http://c"], fail_url="http://b")
        with pytest.raises(RuntimeError):
            client.start_weight_update()
        assert len(client._session.calls) == 3

    def test_error_surfaces_the_response_body_detail(self, make_client):
        client = make_client(
            ["http://a"],
            fail_url="http://a",
            fail_body={"error": {"message": "engine not initialized"}},
        )
        with pytest.raises(RuntimeError, match="engine not initialized"):
            client.start_weight_update()

    def test_error_falls_back_to_fastapi_detail(self, make_client):
        client = make_client(["http://a"], fail_url="http://a", fail_body={"detail": "missing init_info"})
        with pytest.raises(RuntimeError, match="missing init_info"):
            client.start_weight_update()

    def test_close_is_idempotent(self, make_client):
        client = make_client(["http://a"])
        client.close()
        assert client._session.closed

    def test_no_servers_is_rejected(self):
        with pytest.raises(ValueError, match="at least one server_url"):
            SkyrlWeightSyncClient([], 1)
