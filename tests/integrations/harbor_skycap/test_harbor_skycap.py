"""Harbor through skycap, end to end on CPU: a fake trial talks HTTP to a real skycap
server in token mode, which calls a mock SkyRL router."""

import asyncio
from datetime import datetime
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
from skycap import CaptureService, Sample, record  # noqa: E402
from skycap.tokens.engine import EngineError  # noqa: E402
from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    pack_sample_support,  # noqa: E402
)
from skyrl.train.generators.base import TrajectoryID  # noqa: E402
from skyrl.train.generators.utils import concatenate_generator_outputs  # noqa: E402
from skyrl.train.utils.trainer_utils import validate_generator_output  # noqa: E402
from tests.integrations.harbor_skycap.fakes import (  # noqa: E402
    AGENT,
    EXPERTS_PER_TOKEN,
    LAYERS,
    SETUP,
    START_TIMEOUT,
    TOP_K,
    VERIFY,
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
    service = CaptureService(
        router.url,
        mode="tokens",
        renderer=FakeRenderer(),
        engine=SkyRLEngine(),
        model="policy",
        sampling_overrides={"top_k": TOP_K},
        sampling_mask=True,
        record_dir=str(tmp_path / "record"),
        host="127.0.0.1",
    )
    service.start()
    yield service
    service.stop()


def service_for(router, record_dir, **options) -> CaptureService:
    return CaptureService(
        router.url,
        mode="tokens",
        renderer=FakeRenderer(),
        engine=SkyRLEngine(),
        model="policy",
        record_dir=str(record_dir),
        host="127.0.0.1",
        **options,
    )


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


@pytest_asyncio.fixture
async def generator(skycap):
    """Makes generators against the test's skycap, and closes each one's pool when the test ends."""
    made = []

    def make(**cfg) -> HarborSkycapGenerator:
        made.append(
            HarborSkycapGenerator(generator_cfg(**cfg), harbor_cfg(), [skycap.url], SimpleNamespace(weight_version=7))
        )
        return made[-1]

    yield make
    for gen in made:
        await gen.close()


@pytest.mark.asyncio
async def test_a_linear_trial_is_one_complete_multi_turn_row(skycap, router, trials, generator) -> None:
    out = await generator().generate(batch("linear"), disable_tqdm=True)
    validate_generator_output(1, out, step_wise=True)

    assert out["is_last_step"] == [True] and out["rewards"] == [1.0]
    metrics = out["rollout_metrics"]
    assert metrics["generate/skycap/num_unbridged_trajectories"] == 0
    # Harbor's phases, per attempt: one attempt, nothing retried or failed.
    for phase, seconds in (("environment_setup", SETUP), ("agent_execution", AGENT), ("verifier", VERIFY)):
        for stat in ("mean", "p90", "max"):
            assert metrics[f"generate/skycap/{phase}_time_{stat}"] == pytest.approx(seconds)
    assert metrics["generate/skycap/num_attempts"] == 1
    assert metrics["generate/skycap/num_retried_attempts"] == 0
    assert not any(k.startswith("generate/skycap/num_failed_attempts/") for k in metrics)
    prompt, response, mask = out["prompt_token_ids"][0], out["response_ids"][0], out["loss_masks"][0]
    # The prompt is the task; both replies are trained, and the user turn between them is context.
    assert decode(prompt).endswith("userlinearassistant")
    assert mask[0] == 1 and 0 in mask and mask[-1] == 1
    assert len(response) == len(mask) == len(out["rollout_logprobs"][0])
    # Without R3 the trainer gets no routes, though skycap records them.
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
async def test_the_generator_keeps_one_pool_across_batches(skycap, trials, generator) -> None:
    """Fully async training calls `generate` once per prompt; one pool serves every call, round-robin across all."""
    gen = generator()
    pool = gen.pool
    for _ in range(2):
        out = await gen.generate(batch("linear"), disable_tqdm=True)
        assert out["rewards"] == [1.0]
    assert gen.pool is pool


@pytest.mark.asyncio
async def test_a_summarizing_trial_emits_one_row_per_path_grouped_under_its_id(skycap, trials, generator) -> None:
    out = await generator().generate(batch("summarize"), disable_tqdm=True)
    validate_generator_output(1, out, step_wise=True)

    assert len(out["response_ids"]) == 2
    assert out["is_last_step"] == [False, True]
    # The reward is the trial's, so every path carries it; the advantage is computed once, from the last row.
    assert out["rewards"] == [1.0, 1.0]
    assert len({t.to_string() for t in out["trajectory_ids"]}) == 1
    assert out["rollout_metrics"]["generate/skycap/avg_num_paths"] == 2
    # The rewritten history couldn't extend the tokens before it: one call, in one trajectory.
    assert out["rollout_metrics"]["generate/skycap/num_unbridged_trajectories"] == 1
    assert out["rollout_metrics"]["generate/skycap/num_unbridged_calls"] == 1


@pytest.mark.asyncio
async def test_concatenated_outputs_keep_skycap_metrics_apart_from_the_recomputed_ones(
    skycap, trials, generator
) -> None:
    groups = [batch("summarize"), batch("summarize")]
    groups[1]["trajectory_ids"] = [TrajectoryID(instance_id="summarize", repetition_id=1)]
    outs = [await generator().generate(group, disable_tqdm=True) for group in groups]
    metrics = concatenate_generator_outputs(outs, step_wise=True)["rollout_metrics"]

    # The shared stats are recomputed over the whole batch, so none may also appear under skycap's name,
    # where they would be averaged per group instead.
    skycap_keys = {k for k in metrics if k.startswith("generate/skycap/")}
    assert "generate/avg_num_tokens" in metrics
    assert not {k.replace("generate/skycap/", "generate/") for k in skycap_keys} & set(metrics)
    # Counts add up across the concatenated groups; a per-group percentile is averaged, not summed.
    assert metrics["generate/skycap/num_unbridged_calls"] == 2
    assert metrics["generate/skycap/num_attempts"] == 2
    assert metrics["generate/skycap/environment_setup_time_p90"] == pytest.approx(SETUP)


@pytest.mark.asyncio
async def test_the_harness_is_pointed_at_skycap_not_the_engine(skycap, trials, generator) -> None:
    await generator().generate(batch("linear"), disable_tqdm=True)
    kwargs = trials.configs[0]["agent"]["kwargs"]

    assert kwargs["api_base"].startswith(f"{skycap.url}/t/")
    assert "collect_rollout_details" not in kwargs
    assert kwargs["llm_kwargs"]["api_key"] and kwargs["llm_kwargs"]["extra_body"]["cache_salt"]


@pytest.mark.asyncio
async def test_a_timeout_masks_the_whole_instance(skycap, trials, generator) -> None:
    out = await generator().generate(batch("timeout", "linear", repetitions=2), disable_tqdm=True)
    # 4 prompts: two instances ("timeout", "linear"), two repetitions each.
    validate_generator_output(4, out, step_wise=True)

    timed_out = [i for i, t in enumerate(out["trajectory_ids"]) if t.instance_id == "timeout"]
    assert all(out["loss_masks"][i] == [0] and out["rewards"][i] == 0.0 for i in timed_out)
    metrics = out["rollout_metrics"]
    assert metrics["generate/skycap/num_masked_instances"] == 1
    # A timed-out agent isn't retried; its attempts count as failed, and its verifier never ran.
    assert metrics["generate/skycap/num_attempts"] == 4
    assert metrics["generate/skycap/num_retried_attempts"] == 0
    assert metrics["generate/skycap/num_failed_attempts/AgentTimeoutError"] == 2
    assert metrics["generate/skycap/agent_execution_time_max"] == pytest.approx(AGENT)
    assert metrics["generate/skycap/verifier_time_mean"] == pytest.approx(VERIFY)
    # The masked rows still carry support, so the batch collates.
    assert len(out["rollout_sample_support"]) == len(out["response_ids"])


@pytest.mark.asyncio
async def test_a_crashing_trial_is_retried_on_a_fresh_trajectory_then_masked(skycap, trials, generator) -> None:
    out = await generator().generate(batch("crash"), disable_tqdm=True)

    assert out["loss_masks"] == [[0]] and out["stop_reasons"] == ["error"]
    urls = [config["agent"]["kwargs"]["api_base"] for config in trials.configs]
    assert len(urls) == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL and len(set(urls)) == len(urls)
    # Each attempt was finished with the error, so skycap holds nothing open.
    assert not any(t.is_open for t in skycap.server.trajectories.values())
    # Both attempts raised before Harbor returned a result: counted, with no phase times.
    metrics = out["rollout_metrics"]
    assert metrics["generate/skycap/num_attempts"] == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL
    assert metrics["generate/skycap/num_retried_attempts"] == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL - 1
    assert metrics["generate/skycap/num_failed_attempts/RuntimeError"] == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL
    assert not any(k.endswith("_time_mean") and k.startswith("generate/skycap/") for k in metrics)


@pytest.mark.asyncio
async def test_a_sandbox_start_timeout_rescued_by_a_retry_still_shows_in_the_metrics(skycap, trials, generator) -> None:
    """The run this is for: sandboxes timing out at start, retried into a clean batch whose
    error and mask counts stay at zero."""
    out = await generator().generate(batch("slow_start", "linear"), disable_tqdm=True)

    assert out["rewards"] == [1.0, 1.0] and set(out["stop_reasons"]) == {"complete"}
    metrics = out["rollout_metrics"]
    assert metrics["generate/skycap/num_error_trajectories"] == 0
    assert metrics["generate/skycap/num_masked_instances"] == 0
    # Three attempts: the timed-out one and its retry, plus "linear"'s.
    assert metrics["generate/skycap/num_attempts"] == 3
    assert metrics["generate/skycap/num_retried_attempts"] == 1
    assert metrics["generate/skycap/num_failed_attempts/EnvironmentStartTimeoutError"] == 1
    # The failed start counts with its time up to the failure; it never reached the agent.
    setup = np.array([START_TIMEOUT, SETUP, SETUP])
    assert metrics["generate/skycap/environment_setup_time_mean"] == pytest.approx(setup.mean())
    assert metrics["generate/skycap/environment_setup_time_p90"] == pytest.approx(np.percentile(setup, 90))
    assert metrics["generate/skycap/environment_setup_time_max"] == pytest.approx(START_TIMEOUT)
    assert metrics["generate/skycap/agent_execution_time_mean"] == pytest.approx(AGENT)


@pytest.mark.asyncio
async def test_a_trial_with_no_captured_tokens_is_retried_then_masked_not_rewarded(skycap, trials, generator) -> None:
    out = await generator().generate(batch("silent", "linear"), disable_tqdm=True)

    silent = [i for i, t in enumerate(out["trajectory_ids"]) if t.instance_id == "silent"]
    assert [out["rewards"][i] for i in silent] == [0.0] and out["stop_reasons"][silent[0]] == "error"
    assert len(trials.configs) == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL + 1
    # "silent"'s results have no timestamps: skipped, so the times are "linear"'s alone.
    metrics = out["rollout_metrics"]
    assert metrics["generate/skycap/num_retried_attempts"] == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL - 1
    assert metrics["generate/skycap/environment_setup_time_mean"] == pytest.approx(SETUP)


def test_a_phase_without_both_timestamps_has_no_duration() -> None:
    started = SimpleNamespace(started_at=datetime(2026, 1, 1), finished_at=None)
    assert harbor_generator._seconds(started) is None
    assert harbor_generator._seconds(None) is None
    done = SimpleNamespace(started_at=datetime(2026, 1, 1), finished_at=datetime(2026, 1, 1, 0, 0, 3))
    assert harbor_generator._seconds(done) == 3.0


def test_a_batch_with_nothing_to_train_still_carries_padded_support() -> None:
    outcome = TrialOutcome(trajectory_id=TrajectoryID("a", 0), stop_reason="error")

    support = compose([outcome], overlong_filtering=False, top_k=TOP_K, sample_support=True)["rollout_sample_support"]
    assert [array.shape for array in support] == [(1, TOP_K)]
    assert compose([outcome], overlong_filtering=False, top_k=TOP_K)["rollout_sample_support"] is None


R3 = SimpleNamespace(served_model_name="policy", enable_return_routed_experts=True)


def assert_routes_are_the_rows_own(out) -> None:
    """The mock router's route row ``p`` names position ``p`` of its call, and every call starts at 0."""
    for prompt, response, routes in zip(out["prompt_token_ids"], out["response_ids"], out["rollout_expert_indices"]):
        length = len(prompt) + len(response)
        assert routes.shape == (length - 1, LAYERS, EXPERTS_PER_TOKEN)
        assert (routes[:, 0, 0] == np.arange(length - 1) % 256).all()


@pytest.mark.asyncio
async def test_with_r3_each_row_carries_routes_for_its_whole_prompt_and_response(generator, router, trials) -> None:
    out = await generator(inference_engine=R3).generate(batch("linear", "summarize"), disable_tqdm=True)

    validate_generator_output(2, out, step_wise=True)
    # One row for the linear trial, two for the summarizing one, whose second path restarts the history.
    assert len(out["rollout_expert_indices"]) == 3
    assert_routes_are_the_rows_own(out)
    assert out["rollout_metrics"]["generate/skycap/num_missing_route_trajectories"] == 0
    # A turn that extends the previous one asks only for the routes skycap doesn't hold yet.
    starts = [r["sampling_params"].get("routed_experts_prompt_start", 0) for r in router.requests]
    assert any(start > 0 for start in starts)


@pytest.mark.asyncio
async def test_with_r3_a_masked_instance_gets_distinct_dummy_routes(generator, trials) -> None:
    out = await generator(inference_engine=R3).generate(batch("timeout", "linear"), disable_tqdm=True)

    validate_generator_output(2, out, step_wise=True)
    timed_out = out["trajectory_ids"].index(TrajectoryID("timeout", 0))
    assert out["rollout_expert_indices"][timed_out].tolist() == [[list(range(EXPERTS_PER_TOKEN))] * LAYERS]


@pytest.mark.asyncio
async def test_with_r3_a_trial_without_routes_is_retried_then_masked(generator, router, trials) -> None:
    router.routes = False
    out = await generator(inference_engine=R3).generate(batch("linear"), disable_tqdm=True)

    assert out["loss_masks"] == [[0]] and out["stop_reasons"] == ["error"]
    assert len(trials.configs) == harbor_generator.MAX_NUM_RETRIES_PER_TRIAL
    assert out["rollout_metrics"]["generate/skycap/num_missing_route_trajectories"] == 1
    # Nothing trained, so nothing to replay.
    assert out["rollout_expert_indices"] is None


def test_the_generator_refuses_configs_it_cannot_serve() -> None:
    with pytest.raises(ValueError, match="step_wise_trajectories"):
        HarborSkycapGenerator(generator_cfg(step_wise_trajectories=False), {}, ["http://x"])
    with pytest.raises(ValueError, match="merge_stepwise_output"):
        HarborSkycapGenerator(generator_cfg(merge_stepwise_output=True), {}, ["http://x"])
    with pytest.raises(ValueError, match="served_model_name"):
        HarborSkycapGenerator(
            generator_cfg(inference_engine=SimpleNamespace(served_model_name="a/b")), {}, ["http://x"]
        )


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


def test_with_r3_an_overlong_filtered_trial_needs_no_routes() -> None:
    """Its loss mask is cleared, so it trains nothing; without filtering, a missing route is a bug."""
    routed = Sample(leaf=1, path=[0, 1], messages=[], targets=[1], input_ids=[1, 2, 3], loss_mask=[0, 1, 1])
    routed.routed_experts = np.zeros((3, LAYERS, EXPERTS_PER_TOKEN), dtype=np.uint8)
    unrouted = Sample(leaf=1, path=[0, 1], messages=[], targets=[1], input_ids=[4, 5, 6], loss_mask=[0, 1, 1])
    outcomes = [
        TrialOutcome(trajectory_id=TrajectoryID("a", 0), samples=[routed], reward=1.0),
        TrialOutcome(trajectory_id=TrajectoryID("a", 1), samples=[unrouted], stop_reason="context_length"),
    ]

    out = compose(outcomes, overlong_filtering=True, routed_experts=True)
    validate_generator_output(2, out, step_wise=True, routes_expected=True)
    assert out["loss_masks"][1] == [0, 0]
    assert out["rollout_expert_indices"][1].tolist() == [[list(range(EXPERTS_PER_TOKEN))] * LAYERS]
    with pytest.raises(ValueError, match="1 of 2 trained paths have no routed experts"):
        compose(outcomes, overlong_filtering=False, routed_experts=True)


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
    service = service_for(router, tmp_path)
    service.start()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{service.url}/trajectories", json={"meta": {}}) as response:
            created = await response.json()
    # Off this loop: stopping releases the open trajectory's session on the router, which this loop serves.
    await asyncio.to_thread(service.stop)

    assert (tmp_path / f"{created['id']}.json.zst").exists()


@pytest.mark.asyncio
async def test_thinking_survives_litellm_so_the_replayed_history_stays_one_path(router, tmp_path) -> None:
    """Terminus-2 talks to skycap through LiteLLM's `hosted_vllm/` provider, which splits `<think>`
    out of `content` unless the reply carries `reasoning_content: null` the way vLLM's does."""
    import litellm

    router.reply = "<think>\nhmm\n</think>\n\nanswer"
    service = service_for(router, tmp_path, use_raw_content=True)
    service.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{service.url}/trajectories", json={"meta": {}}) as response:
                created = await response.json()
        history = [{"role": "user", "content": "q"}]
        for _ in range(2):
            reply = await litellm.acompletion(
                model="hosted_vllm/policy", messages=history, api_base=created["base_url"], api_key="k"
            )
            content = reply.choices[0].message.content
            assert content.startswith("<think>")
            history += [{"role": "assistant", "content": content}, {"role": "user", "content": "more"}]
        graph = service.server.trajectories[created["id"]].graph
        assert len(graph.paths()) == 1
    finally:
        await asyncio.to_thread(service.stop)
