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
from tests.integrations.harbor_skycap.fakes import MockRouter  # noqa: E402

pytestmark = pytest.mark.integrations

REPO = Path(__file__).resolve().parents[3]


def fake_backend(settings):
    """Built inside each actor: token mode with the fake renderer, against the mock router."""
    from examples.train_integrations.harbor_skycap.engine import SkyRLEngine
    from skycap.tokens.backend import TokensBackend
    from tests.integrations.harbor_skycap.fakes import FakeRenderer

    return TokensBackend(settings["engine_url"], FakeRenderer(), engine=SkyRLEngine(), model="policy")


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
        {"engine_url": str(server.make_url("")).rstrip("/")},
        num_servers=2,
        num_cpus_per_server=1,
        placement_strategy="SPREAD",
        record_dir=str(tmp_path),
        ttl=60.0,
        backend_factory=fake_backend,
    )
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
        servers.stop()
        await server.close()
    assert len(list(tmp_path.glob("*.json.zst"))) == 4
