"""The skycap server pool: Ray actors on their own ports, used round-robin, flushed on stop."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("skycap")
pytest.importorskip("harbor")

import ray  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from examples.train_integrations.harbor_skycap.servers import (
    start_servers,  # noqa: E402
)
from tests.integrations.harbor_skycap.fakes import (  # noqa: E402
    FakeRenderer,
    MockRouter,
)

pytestmark = pytest.mark.integrations

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def local_ray():
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        object_store_memory=200 * 1024**2,
        runtime_env={"env_vars": {"PYTHONPATH": str(REPO)}},
    )
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_a_pool_of_servers_serves_a_batch_and_writes_it(local_ray, tmp_path, monkeypatch) -> None:
    from examples.train_integrations.harbor_skycap import harbor_generator
    from tests.integrations.harbor_skycap.fakes import FakeTrial
    from tests.integrations.harbor_skycap.test_harbor_skycap import (
        batch,
        generator_cfg,
        harbor_cfg,
    )

    router = MockRouter()
    server = TestServer(router.app(), host="0.0.0.0")
    await server.start_server()
    servers = start_servers(
        # The fake renderer goes to each actor with the settings, in place of a tokenizer.
        {"upstream_url": str(server.make_url("")).rstrip("/"), "renderer": FakeRenderer(), "model": "policy"},
        num_servers=2,
        num_cpus_per_server=1,
        placement_strategy="SPREAD",
        record_dir=str(tmp_path),
        ttl=60.0,
    )
    gen = None
    try:
        # Each server picked its own port.
        assert len(set(servers.urls)) == 2
        FakeTrial.configs = []
        monkeypatch.setattr(harbor_generator, "Trial", FakeTrial)
        gen = harbor_generator.HarborSkycapGenerator(
            generator_cfg(), harbor_cfg(), servers.urls, SimpleNamespace(weight_version=1)
        )
        out = await gen.generate(batch("linear", repetitions=4), disable_tqdm=True)
        assert sum(out["loss_masks"][0]) > 0
        # Trajectories were spread over both servers.
        used = {c["agent"]["kwargs"]["api_base"].split("/t/")[0] for c in FakeTrial.configs}
        assert used == set(servers.urls)
    finally:
        if gen is not None:
            await gen.close()
        servers.stop()
        await server.close()
    assert len(list(tmp_path.glob("*.json.zst"))) == 4


EXPOSURES = "tests.integrations.harbor_skycap.test_exposure"


@pytest.mark.asyncio
async def test_each_server_exposes_its_harness_routes_to_agents_in_sandboxes(local_ray, tmp_path, monkeypatch) -> None:
    from examples.train_integrations.harbor_skycap import harbor_generator
    from tests.integrations.harbor_skycap.fakes import FakeTrial
    from tests.integrations.harbor_skycap.test_harbor_skycap import (
        batch,
        generator_cfg,
        harbor_cfg,
    )

    router = MockRouter()
    server = TestServer(router.app(), host="0.0.0.0")
    await server.start_server()
    servers = start_servers(
        {"upstream_url": str(server.make_url("")).rstrip("/"), "renderer": FakeRenderer(), "model": "policy"},
        num_servers=2,
        num_cpus_per_server=1,
        placement_strategy="SPREAD",
        record_dir=str(tmp_path),
        ttl=60.0,
        exposure=f"{EXPOSURES}:LoopbackExposure",
    )
    gen = None
    try:
        FakeTrial.configs = []
        monkeypatch.setattr(harbor_generator, "Trial", FakeTrial)
        agent = harbor_cfg()
        agent["agent"]["name"] = "mini-swe-agent"
        gen = harbor_generator.HarborSkycapGenerator(
            generator_cfg(), agent, servers.urls, SimpleNamespace(weight_version=1)
        )
        out = await gen.generate(batch("linear", repetitions=4), disable_tqdm=True)
        assert sum(out["loss_masks"][0]) > 0
        # Each actor built its own exposure, and the agents called both servers on their exposed listeners.
        used = {c["agent"]["env"]["OPENAI_API_BASE"].split("/t/")[0] for c in FakeTrial.configs}
        assert len(used) == 2 and not used & set(servers.urls)
    finally:
        if gen is not None:
            await gen.close()
        servers.stop()
        await server.close()
    assert len(list(tmp_path.glob("*.json.zst"))) == 4


def test_a_failed_exposure_fails_the_start_and_releases_the_pool(local_ray, tmp_path) -> None:
    before = ray.available_resources().get("CPU")
    with pytest.raises(ray.exceptions.RayTaskError, match="no way in"):
        start_servers(
            {"upstream_url": "http://127.0.0.1:9", "renderer": FakeRenderer(), "model": "policy"},
            num_servers=2,
            num_cpus_per_server=1,
            placement_strategy="SPREAD",
            record_dir=str(tmp_path),
            ttl=60.0,
            exposure=f"{EXPOSURES}:FailingExposure",
        )
    # The actors were killed and the placement group removed: its CPUs come back.
    import time

    deadline = time.monotonic() + 30
    while ray.available_resources().get("CPU") != before and time.monotonic() < deadline:
        time.sleep(0.5)
    assert ray.available_resources().get("CPU") == before
