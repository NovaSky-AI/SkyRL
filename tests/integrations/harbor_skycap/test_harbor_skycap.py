"""Harbor through skycap, end to end on CPU: a fake trial talks HTTP to a real skycap
server in token mode, which calls a mock SkyRL router."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("skycap")
pytest.importorskip("harbor")

import aiohttp  # noqa: E402
import pytest_asyncio  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,  # noqa: E402
)
from examples.train_integrations.harbor_skycap import harbor_generator  # noqa: E402
from examples.train_integrations.harbor_skycap.compose import (  # noqa: E402
    TrialOutcome,
    compose,
)
from examples.train_integrations.harbor_skycap.engine import SkyRLEngine  # noqa: E402
from examples.train_integrations.harbor_skycap.harbor_generator import (
    HarborSkycapGenerator,  # noqa: E402
)
from examples.train_integrations.harbor_skycap.service import (
    SkycapService,  # noqa: E402
)
from skycap import Sample, record  # noqa: E402
from skycap.tokens.backend import TokensBackend  # noqa: E402
from skycap.tokens.engine import EngineError  # noqa: E402
from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    pack_sample_support,  # noqa: E402
)
from skyrl.train.generators.base import TrajectoryID  # noqa: E402
from skyrl.train.utils.trainer_utils import validate_generator_output  # noqa: E402
from tests.integrations.harbor_skycap.fakes import (  # noqa: E402
    EXPERTS_PER_TOKEN,
    LAYERS,
    TOP_K,
    FakeRenderer,
    FakeTrial,
    MockRouter,
    decode,
)

pytestmark = pytest.mark.integrations


def generator_cfg(**overrides):
    values = dict(
        step_wise_trajectories=True,
        merge_stepwise_output=False,
        use_cache_salt=True,
        apply_overlong_filtering=False,
        rate_limit=None,
        inference_engine=SimpleNamespace(served_model_name="policy"),
        sampling_params=SimpleNamespace(top_k=TOP_K),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest_asyncio.fixture
async def router():
    mock = MockRouter()
    server = TestServer(mock.app())
    await server.start_server()
    mock.url = str(server.make_url("")).rstrip("/")
    yield mock
    await server.close()


@pytest.fixture
def skycap(router, tmp_path):
    backend = TokensBackend(
        router.url,
        FakeRenderer(),
        engine=SkyRLEngine(),
        model="policy",
        sampling_overrides={"top_k": TOP_K},
        sampling_mask=True,
    )
    service = SkycapService(backend, record_dir=str(tmp_path / "record"), host="127.0.0.1")
    service.start()
    yield service
    service.stop()


@pytest.fixture
def trials(monkeypatch):
    FakeTrial.configs = []
    monkeypatch.setattr(harbor_generator, "Trial", FakeTrial)
    return FakeTrial


def harbor_cfg() -> dict:
    with open(HARBOR_DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def batch(*scripts: str, repetitions: int = 1) -> dict:
    prompts, ids = [], []
    for script in scripts:
        for repetition in range(repetitions):
            prompts.append(script)
            ids.append(TrajectoryID(instance_id=script, repetition_id=repetition))
    return {"prompts": prompts, "trajectory_ids": ids, "batch_metadata": SimpleNamespace(global_step=3)}


def generator(skycap, **cfg) -> HarborSkycapGenerator:
    engine_client = SimpleNamespace(weight_version=7)
    return HarborSkycapGenerator(generator_cfg(**cfg), harbor_cfg(), [skycap.url], engine_client)


@pytest.mark.asyncio
async def test_a_linear_trial_is_one_complete_multi_turn_row(skycap, router, trials) -> None:
    out = await generator(skycap).generate(batch("linear"), disable_tqdm=True)
    validate_generator_output(1, out, step_wise=True)

    assert out["is_last_step"] == [True] and out["rewards"] == [1.0]
    prompt, response, mask = out["prompt_token_ids"][0], out["response_ids"][0], out["loss_masks"][0]
    # The prompt is the task; both replies are trained, and the user turn between them is context.
    assert decode(prompt).endswith("userlinearassistant")
    assert mask[0] == 1 and 0 in mask and mask[-1] == 1
    assert len(response) == len(mask) == len(out["rollout_logprobs"][0])
    # Routes aren't handed to the trainer yet (it refuses R3 with step-wise output), but skycap records them.
    assert out["rollout_expert_indices"] is None
    (trajectory_id,) = record.list_ids(skycap.server.record_dir)
    for node in record.load(skycap.server.record_dir, trajectory_id).graph:
        routed = node.tokens.routed_experts
        assert routed.shape == (len(node.tokens.token_ids), LAYERS, EXPERTS_PER_TOKEN)
    # Support rows line up with the response: the sampled token first, none where untrained.
    support = out["rollout_sample_support"][0]
    assert support.shape == (len(response), TOP_K)
    for token, trained, row in zip(response, mask, support):
        assert (row[0] == token) if trained else (row == -1).all()
    # The router saw the trajectory as one session, salted, with the imposed top_k, and was told when it ended.
    assert len(set(router.sessions)) == 1 and router.released == router.sessions[:1]
    assert all(r["cache_salt"] and r["sampling_params"]["top_k"] == TOP_K for r in router.requests)


@pytest.mark.asyncio
async def test_a_summarizing_trial_emits_one_row_per_path_grouped_under_its_id(skycap, trials) -> None:
    out = await generator(skycap).generate(batch("summarize"), disable_tqdm=True)
    validate_generator_output(1, out, step_wise=True)

    assert len(out["response_ids"]) == 2
    assert out["is_last_step"] == [False, True]
    assert out["rewards"] == [0.0, 1.0]
    assert len({t.to_string() for t in out["trajectory_ids"]}) == 1
    assert out["rollout_metrics"]["generate/avg_num_paths"] == 2


@pytest.mark.asyncio
async def test_the_harness_is_pointed_at_skycap_not_the_engine(skycap, trials) -> None:
    await generator(skycap).generate(batch("linear"), disable_tqdm=True)
    kwargs = trials.configs[0]["agent"]["kwargs"]

    assert kwargs["api_base"].startswith(f"{skycap.url}/t/")
    assert "collect_rollout_details" not in kwargs
    assert kwargs["llm_kwargs"]["api_key"] and kwargs["llm_kwargs"]["extra_body"]["cache_salt"]


@pytest.mark.asyncio
async def test_a_timeout_masks_the_whole_instance(skycap, trials) -> None:
    out = await generator(skycap).generate(batch("timeout", "linear", repetitions=2), disable_tqdm=True)
    validate_generator_output(4, out, step_wise=True)

    timed_out = [i for i, t in enumerate(out["trajectory_ids"]) if t.instance_id == "timeout"]
    assert all(out["loss_masks"][i] == [0] and out["rewards"][i] == 0.0 for i in timed_out)
    assert out["rollout_metrics"]["generate/num_masked_instances"] == 1
    # The masked rows still carry support, so the batch collates.
    assert len(out["rollout_sample_support"]) == len(out["response_ids"])


@pytest.mark.asyncio
async def test_a_crashing_trial_is_retried_on_a_fresh_trajectory_then_masked(skycap, trials) -> None:
    out = await generator(skycap).generate(batch("crash"), disable_tqdm=True)

    assert out["loss_masks"] == [[0]] and out["stop_reasons"] == ["error"]
    urls = [config["agent"]["kwargs"]["api_base"] for config in trials.configs]
    assert len(urls) == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL and len(set(urls)) == len(urls)
    # Each attempt was finished with the error, so skycap holds nothing open.
    assert not any(t.is_open for t in skycap.server.trajectories.values())


def test_the_generator_refuses_configs_it_cannot_serve() -> None:
    with pytest.raises(ValueError, match="step_wise_trajectories"):
        HarborSkycapGenerator(generator_cfg(step_wise_trajectories=False), {}, ["http://x"])
    with pytest.raises(ValueError, match="merge_stepwise_output"):
        HarborSkycapGenerator(generator_cfg(merge_stepwise_output=True), {}, ["http://x"])
    r3 = SimpleNamespace(served_model_name="policy", enable_return_routed_experts=True)
    with pytest.raises(ValueError, match="R3"):
        HarborSkycapGenerator(generator_cfg(inference_engine=r3), {}, ["http://x"])


def test_the_engine_rejects_support_that_does_not_cover_the_completion() -> None:
    support = pack_sample_support(np.zeros((1, TOP_K), dtype=np.int32))
    body = {
        "choices": [
            {
                "token_ids": [5, 6],
                "finish_reason": "stop",
                "logprobs": {"content": [{"logprob": -1.0}, {"logprob": -1.0}]},
                "rollout_sample_support": support,
            }
        ]
    }
    with pytest.raises(EngineError, match="sample-support rows"):
        SkyRLEngine().parse(body)


def test_the_engine_asks_for_support_only_with_a_sampling_mask() -> None:
    kwargs = dict(prompt_ids=[1], sampling={"top_k": 3}, model="policy", cache_salt="s")
    assert SkyRLEngine().request(sampling_mask=True, **kwargs)["return_sample_support"] is True
    assert "return_sample_support" not in SkyRLEngine().request(sampling_mask=False, **kwargs)


def test_overlong_filtering_masks_a_context_length_trial_but_keeps_it() -> None:
    sample = Sample(leaf=1, path=[0, 1], messages=[], targets=[1], input_ids=[1, 2, 3], loss_mask=[0, 1, 1])
    sample.logprobs = [0.0, -1.0, -1.0]
    outcome = TrialOutcome(
        trajectory_id=TrajectoryID("a", 0), samples=[sample], reward=0.0, stop_reason="context_length"
    )

    assert compose([outcome], overlong_filtering=True)["loss_masks"] == [[0, 0]]
    assert compose([outcome], overlong_filtering=False)["loss_masks"] == [[1, 1]]


@pytest.mark.asyncio
async def test_the_service_writes_open_trajectories_when_stopped(router, tmp_path) -> None:
    backend = TokensBackend(router.url, FakeRenderer(), engine=SkyRLEngine(), model="policy")
    service = SkycapService(backend, record_dir=str(tmp_path), host="127.0.0.1")
    service.start()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{service.url}/trajectories", json={"meta": {}}) as response:
            created = await response.json()
    # Off this loop: stopping releases the open trajectory's session on the router, which this loop serves.
    await asyncio.to_thread(service.stop)

    assert (tmp_path / f"{created['id']}.json.zst").exists()
