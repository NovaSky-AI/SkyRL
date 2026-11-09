"""The server actor's sample cache: what a group waits on between its trials' finishes and its compose."""

import pytest

pytest.importorskip("skycap")

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from ray.exceptions import RayActorError  # noqa: E402

from skyrl.train.generators.base import StoredRolloutGroup  # noqa: E402
from skyrl.train.generators.skycap_store import (  # noqa: E402
    SERVER_DIED_METRIC,
    SampleCache,
    StoredCapturePool,
    StoredPools,
)

pytestmark = pytest.mark.integrations


def test_samples_are_taken_once_and_discarded_ones_are_gone() -> None:
    cache = SampleCache()
    cache.put("a", ["sample-a"])
    cache.put("b", ["sample-b"])

    assert cache.take("a") == ["sample-a"]
    with pytest.raises(KeyError, match="no samples cached for a"):
        cache.take("a")
    cache.discard(["b", "never-finished"])
    assert len(cache) == 0


def test_a_group_s_samples_wait_however_long_its_slowest_trial_takes() -> None:
    cache = SampleCache()
    cache.put("early", ["x"])
    for i in range(1000):
        cache.put(f"sibling-{i}", ["y"])

    assert cache.take("early") == ["x"]


def _pool(url: str) -> SimpleNamespace:
    return SimpleNamespace(urls=[url])


def _group(url: str) -> StoredRolloutGroup:
    return StoredRolloutGroup(trajectories=[], group_fields={"rollout_metrics": {"server": url}})


def test_a_group_whose_server_died_runs_again_on_the_next_live_server() -> None:
    pools = StoredPools([_pool("a"), _pool("b"), _pool("c")])
    ran = []

    async def group(pool):
        ran.append(pool.urls[0])
        if pool.urls[0] == "a":
            raise RayActorError()
        return _group(pool.urls[0])

    rerun = asyncio.run(pools.run(group))
    # The rerun counts the death in its own rollout metrics.
    assert rerun.group_fields["rollout_metrics"] == {"server": "b", SERVER_DIED_METRIC: 1}
    # The dead server is skipped from then on, and groups that ran once count nothing.
    later = [asyncio.run(pools.run(group)).group_fields["rollout_metrics"] for _ in range(3)]
    assert later == [{"server": "c"}, {"server": "b"}, {"server": "c"}]
    assert ran == ["a", "b", "c", "b", "c"]


def test_a_rerun_group_without_rollout_metrics_gets_them() -> None:
    pools = StoredPools([_pool("a"), _pool("b"), _pool("c")])

    async def group(pool):
        if pool.urls[0] != "c":
            raise RayActorError()
        return StoredRolloutGroup(trajectories=[], group_fields={"rollout_metrics": None})

    assert asyncio.run(pools.run(group)).group_fields["rollout_metrics"] == {SERVER_DIED_METRIC: 2}


def test_a_group_fails_once_every_server_has_died() -> None:
    pools = StoredPools([_pool("a"), _pool("b")])

    async def group(pool):
        raise RayActorError()

    with pytest.raises(RuntimeError, match="every skycap server actor has died"):
        asyncio.run(pools.run(group))


def test_a_stored_finish_gives_the_server_the_pool_s_deadline() -> None:
    calls = []

    class Method:
        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            future = asyncio.get_running_loop().create_future()
            future.set_result("finished")
            return future

    pool = StoredCapturePool("http://server", SimpleNamespace(finish_trajectory=Method()), timeout=7.0)
    trajectory = SimpleNamespace(id="t1", paths="final")

    assert asyncio.run(pool.finish(trajectory, {"reward": 1.0})) == "finished"
    assert calls == [(("t1", {"reward": 1.0}), {"paths": "final", "timeout": 7.0})]
