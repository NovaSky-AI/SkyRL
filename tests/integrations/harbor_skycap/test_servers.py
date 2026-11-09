"""The skycap server pool: Ray actors on their own ports, used round-robin, flushed on stop."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("skycap")
pytest.importorskip("harbor")

import pytest_asyncio  # noqa: E402
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
    # The cluster total, not a snapshot: an earlier test's removed placement group may still be releasing its CPUs.
    total = ray.cluster_resources()["CPU"]
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
    while ray.available_resources().get("CPU") != total and time.monotonic() < deadline:
        time.sleep(0.5)
    assert ray.available_resources().get("CPU") == total


@pytest_asyncio.fixture
async def stored(local_ray, tmp_path, monkeypatch):
    """Two server actors, and a generator that keeps each group on one of them."""
    from examples.train_integrations.harbor_skycap import harbor_generator
    from tests.integrations.harbor_skycap.fakes import FakeTrial
    from tests.integrations.harbor_skycap.test_harbor_skycap import (
        generator_cfg,
        harbor_cfg,
    )

    router = MockRouter()
    server = TestServer(router.app(), host="0.0.0.0")
    await server.start_server()
    servers = start_servers(
        {
            "upstream_url": str(server.make_url("")).rstrip("/"),
            "renderer": FakeRenderer(),
            "model": "policy",
            "sampling_mask": True,
        },
        num_servers=2,
        num_cpus_per_server=1,
        placement_strategy="SPREAD",
        record_dir=str(tmp_path),
        ttl=60.0,
        max_concurrent_finishes=8,
    )
    FakeTrial.configs = []
    monkeypatch.setattr(harbor_generator, "Trial", FakeTrial)
    made = []

    def make(train_paths="all", records=None, support_id_bits=None, check_first_batch=True, **cfg):
        made.append(
            harbor_generator.HarborSkycapGenerator(
                generator_cfg(**cfg),
                harbor_cfg(),
                servers.urls,
                SimpleNamespace(weight_version=1),
                train_paths=train_paths,
                records=records,
                servers=servers,
                support_id_bits=support_id_bits,
                check_first_batch=check_first_batch,
            )
        )
        return made[-1]

    try:
        yield SimpleNamespace(make=make, servers=servers, trials=FakeTrial, router=router)
    finally:
        for gen in made:
            await gen.close()
        servers.stop()
        await server.close()


def _servers_used(configs):
    return {c["agent"]["kwargs"]["api_base"].split("/t/")[0] for c in configs}


def _cached_samples(servers):
    """Each server's finished trajectories whose samples still wait for their group."""
    return ray.get([actor.__ray_call__.remote(lambda actor: len(actor.samples)) for actor in servers.actors])


@pytest.mark.asyncio
async def test_a_stored_group_is_composed_on_its_one_server_and_left_in_its_object_store(stored) -> None:
    from skyrl.backends.skyrl_train.rollout_shard import PAYLOAD_KEYS
    from tests.integrations.harbor_skycap.test_harbor_skycap import batch

    gen, servers, trials = stored.make(), stored.servers, stored.trials

    group = await gen.generate_to_ray(batch("linear", repetitions=3), disable_tqdm=True)
    first_server = _servers_used(trials.configs)
    trials.configs.clear()
    second = await gen.generate_to_ray(batch("linear", repetitions=3), disable_tqdm=True)

    # Each group's trials used one server, and the next group the other one.
    assert len(first_server) == 1 and _servers_used(trials.configs) | first_server == set(servers.urls)
    assert len(group.trajectories) == 3
    for trajectory in group.trajectories:
        assert set(trajectory.metadata) & PAYLOAD_KEYS == set()
        payload = ray.get(trajectory.payload)
        assert set(payload) >= {"prompt_token_ids", "response_ids", "loss_masks"}
        assert all(sum(mask) > 0 for mask in payload["loss_masks"])
        assert trajectory.metadata["is_last_step"][-1]
    assert group.group_fields["rollout_metrics"]["generate/skycap/num_masked_instances"] == 0
    assert second.summary.num_loss_tokens == group.summary.num_loss_tokens
    assert _cached_samples(servers) == [0, 0]


@pytest.mark.asyncio
async def test_a_stored_group_packs_its_support_ids_at_the_policy_vocabulary_width(stored) -> None:
    from skyrl.backends.skyrl_train.rollout_shard import StoredSampleSupport
    from tests.backends.skyrl_train.sample_support_helpers import dense
    from tests.integrations.harbor_skycap.fakes import TOP_K
    from tests.integrations.harbor_skycap.test_harbor_skycap import batch

    ssr = SimpleNamespace(served_model_name="policy", enable_return_sample_support_set=True)
    group = await stored.make(inference_engine=ssr, support_id_bits=17).generate_to_ray(
        batch("linear"), disable_tqdm=True
    )

    (payload,) = ray.get([trajectory.payload for trajectory in group.trajectories])
    (rows,) = payload["rollout_sample_support"]
    assert isinstance(rows, StoredSampleSupport) and rows.bits == 17
    # Unpacked, the rows line up with the response as the HTTP path's do: the sampled token first.
    support = dense(rows.unpack())
    response, mask = payload["response_ids"][0], payload["loss_masks"][0]
    assert support.shape == (len(response), TOP_K)
    for token, trained, row in zip(response, mask, support):
        assert (row[0] == token) if trained else (row == -1).all()


@pytest.mark.asyncio
async def test_generate_spreads_a_batch_over_every_server_given_their_actors(stored) -> None:
    from tests.integrations.harbor_skycap.test_harbor_skycap import batch

    out = await stored.make().generate(batch("linear", repetitions=4), disable_tqdm=True)

    assert all(sum(mask) > 0 for mask in out["loss_masks"])
    assert _servers_used(stored.trials.configs) == set(stored.servers.urls)


@pytest.mark.asyncio
async def test_a_stored_group_masks_and_retries_as_the_http_path_does(stored) -> None:
    from examples.train_integrations.harbor_skycap import harbor_generator
    from tests.integrations.harbor_skycap.test_harbor_skycap import batch

    gen, servers, trials = stored.make(), stored.servers, stored.trials

    out = await gen.generate(batch("crash", "linear"), disable_tqdm=True)

    # The crashing trial was retried on fresh trajectories, then masked with its instance only.
    assert len(trials.configs) == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL + 1
    crashed = [i for i, t in enumerate(out["trajectory_ids"]) if t.instance_id == "crash"]
    assert [out["loss_masks"][i] for i in crashed] == [[0]] and out["stop_reasons"][crashed[0]] == "error"
    assert out["rollout_metrics"]["generate/skycap/num_masked_instances"] == 1
    # The retried attempts' samples were dropped, not left cached.
    assert _cached_samples(servers) == [0, 0]


@pytest.mark.asyncio
async def test_a_first_stored_group_with_nothing_to_train_stops_the_run(stored) -> None:
    from tests.integrations.harbor_skycap.test_harbor_skycap import batch

    with pytest.raises(RuntimeError, match=r"first batch \(2\).*skycap\.check_first_batch=false"):
        await stored.make().generate_to_ray(batch("silent", repetitions=2), disable_tqdm=True)
    # Nothing was composed, and the server drops what it had cached for the group (without being awaited).
    deadline = time.monotonic() + 30
    while _cached_samples(stored.servers) != [0, 0] and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert _cached_samples(stored.servers) == [0, 0]


@pytest.mark.asyncio
async def test_stored_groups_keep_path_selection_trial_metrics_and_record_locations(stored) -> None:
    from examples.train_integrations.harbor_skycap.record_index import RecordLog
    from skycap import record
    from tests.integrations.harbor_skycap.test_harbor_skycap import batch

    records = RecordLog()
    gen = stored.make(train_paths="final", records=records)
    group = await gen.generate_to_ray(batch("linear", "summarize"), disable_tqdm=True)

    assert sum(t.num_rows for t in group.trajectories) == 2
    assert group.group_fields["rollout_metrics"]["generate/harbor/num_attempts"] == 2
    entries = records.take("train")
    assert len(entries) == 2
    for entry in entries:
        assert entry.record is not None
        document = record.read_document(Path(entry.record["path"]).parent, entry.id)
        assert document["samples"]["paths"] == "final"
