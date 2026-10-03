"""The skycap server pool: Ray actors on their own ports, used round-robin, flushed on stop."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("skycap")
pytest.importorskip("harbor")

import ray  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from examples.train_integrations.harbor_skycap.exposure import (  # noqa: E402
    exposure_factory,
)
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


#: Exposes each server's gateway at its loopback URL and logs its starts and stops.
RECORDING = "tests.integrations.harbor_skycap.test_exposure:RecordingExposure"


@pytest.mark.asyncio
async def test_each_server_exposes_its_harness_routes_and_closes_them_on_stop(local_ray, tmp_path, monkeypatch) -> None:
    from examples.train_integrations.harbor_skycap import harbor_generator
    from tests.integrations.harbor_skycap.fakes import FakeTrial
    from tests.integrations.harbor_skycap.test_harbor_skycap import (
        batch,
        generator_cfg,
        harbor_cfg,
    )

    log = tmp_path / "log"
    router = MockRouter()
    server = TestServer(router.app(), host="0.0.0.0")
    await server.start_server()
    servers = start_servers(
        {"upstream_url": str(server.make_url("")).rstrip("/"), "renderer": FakeRenderer(), "model": "policy"},
        num_servers=2,
        num_cpus_per_server=1,
        placement_strategy="SPREAD",
        record_dir=str(tmp_path / "record"),
        ttl=60.0,
        exposure=exposure_factory(RECORDING, kwargs={"log": str(log)}),
    )
    gen = None
    try:
        # Each actor built its own exposure from the import path and started it on its own gateway.
        assert set(servers.harness_urls) == set(servers.urls)
        exposed = set(servers.harness_urls.values())
        assert len(exposed) == 2 and not exposed & set(servers.urls)
        assert sorted(line.rsplit(" ", 1)[0] for line in log.read_text().splitlines()) == ["start 0", "start 1"]
        FakeTrial.configs = []
        monkeypatch.setattr(harbor_generator, "Trial", FakeTrial)
        gen = harbor_generator.HarborSkycapGenerator(
            generator_cfg(), harbor_cfg(), servers.urls, SimpleNamespace(weight_version=1), servers.harness_urls
        )
        out = await gen.generate(batch("linear", repetitions=4), disable_tqdm=True)
        assert sum(out["loss_masks"][0]) > 0
        # The agents called through the exposed URLs, on both servers.
        used = {c["agent"]["kwargs"]["api_base"].split("/t/")[0] for c in FakeTrial.configs}
        assert used == exposed
    finally:
        if gen is not None:
            await gen.close()
        servers.stop()
        await server.close()
    assert sorted(line for line in log.read_text().splitlines() if line.startswith("stop")) == ["stop 0", "stop 1"]
    assert len(list((tmp_path / "record").glob("*.json.zst"))) == 4


def test_a_failed_exposure_fails_the_start_and_closes_the_others(local_ray, tmp_path) -> None:
    log = tmp_path / "log"
    with pytest.raises(ray.exceptions.RayTaskError, match="no way in for server 1"):
        start_servers(
            {"upstream_url": "http://127.0.0.1:9", "renderer": FakeRenderer(), "model": "policy"},
            num_servers=2,
            num_cpus_per_server=1,
            placement_strategy="SPREAD",
            record_dir=str(tmp_path / "record"),
            ttl=60.0,
            exposure=exposure_factory(RECORDING, kwargs={"log": str(log), "fail_on": 1}),
        )
    assert sorted(line for line in log.read_text().splitlines() if line.startswith("stop")) == ["stop 0", "stop 1"]
