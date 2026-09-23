import base64
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from examples.model_checks.logprob_checks import (
    check_agreement,
    compare_logprobs,
    perturb_adapters,
)


def test_logprob_check_preserves_alignment_and_reports_tail_errors():
    result = compare_logprobs([-1, -2, -3], [-1, -2, -3.3])
    assert result["tokens"] == 3
    assert result["mean_abs"] == pytest.approx(0.1)
    assert result["max_abs"] == pytest.approx(0.3)
    assert result["p99_abs"] > 0.29


@pytest.mark.parametrize(
    "reference,actual",
    [
        ([], []),
        ([-1], [-1, -2]),
        ([-1], [float("nan")]),
        ([float("inf")], [-1]),
        ([[-1]], [[-1]]),
    ],
)
def test_logprob_check_rejects_missing_nonfinite_or_misaligned_scores(reference, actual):
    with pytest.raises(AssertionError):
        compare_logprobs(reference, actual)


def test_perturbation_is_adapter_only_and_replica_deterministic():
    base = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    adapter = torch.nn.Parameter(torch.zeros(4))
    replica = torch.nn.Parameter(torch.zeros(4))
    a = torch.nn.Parameter(torch.ones(4))
    receipt = perturb_adapters(
        [
            ("weight", base),
            ("adapter.linear_in.weight", a),
            ("adapter.linear_out.weight", adapter),
        ]
    )
    perturb_adapters([("adapter.linear_out.weight", replica)])
    torch.testing.assert_close(base, torch.ones(4))
    torch.testing.assert_close(adapter, replica, rtol=0, atol=0)
    assert adapter.abs().sum() > 0
    torch.testing.assert_close(a, torch.ones(4), rtol=0, atol=0)
    assert receipt["changed_b_elements"] == 4


def test_perturbation_rejects_an_already_updated_adapter():
    with pytest.raises(AssertionError, match="zero-init B"):
        perturb_adapters([("adapter.linear_out.weight", torch.nn.Parameter(torch.ones(4)))])


@pytest.mark.parametrize("multiplier", [10, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_explicit_stimulus_preserves_a_and_scales_seeded_b(multiplier, dtype):
    a = torch.nn.Parameter(torch.ones(4, dtype=dtype))
    b = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    direction = torch.nn.Parameter(torch.zeros_like(b))
    perturb_adapters([("adapter.linear_out.weight", direction)], multiplier=1)
    report = perturb_adapters(
        [("adapter.linear_in.weight", a), ("adapter.linear_out.weight", b)],
        multiplier=multiplier,
    )
    assert torch.equal(a, torch.ones_like(a))
    assert torch.equal(b, direction * multiplier)
    assert report["multiplier"] == multiplier


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf")])
def test_invalid_stimulus_fails_before_adapter_mutation(multiplier):
    b = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="positive and finite"):
        perturb_adapters([("adapter.linear_out.weight", b)], multiplier=multiplier)
    assert torch.equal(b, torch.zeros_like(b))


def test_perturbation_rejects_full_weight_training():
    with pytest.raises(AssertionError, match="trainable base"):
        perturb_adapters([("weight", torch.nn.Parameter(torch.ones(4)))])


def test_perturbation_rejects_missing_trainable_adapters():
    with pytest.raises(AssertionError):
        perturb_adapters([("adapter.weight", torch.nn.Parameter(torch.ones(4), requires_grad=False))])


@pytest.mark.parametrize("mean_error,max_error", [(0.05, 0.5), (0.050001, 0.5), (0.05, 0.500001)])
def test_agreement_accepts_inclusive_limits_and_rejects_excess(mean_error, max_error):
    result = {"mean_abs": mean_error, "max_abs": max_error}
    if mean_error == 0.05 and max_error == 0.5:
        check_agreement(result, 0.05, 0.5)
    else:
        with pytest.raises(AssertionError):
            check_agreement(result, 0.05, 0.5)


@pytest.fixture
def checks(monkeypatch):
    # Model construction is outside this orchestration test.
    for name, attribute in [
        ("skyrl.backends.skyrl_train.workers.megatron.megatron_worker", "MegatronPolicyWorkerBase"),
        ("skyrl.backends.skyrl_train.workers.worker", "PPORayActorGroup"),
    ]:
        module = ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[3] / "examples/model_checks/run_logprobs.py"
    spec = importlib.util.spec_from_file_location("logprob_example_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("colocated", [False, True])
@pytest.mark.parametrize(
    "lora,fault",
    [
        (True, fault)
        for fault in [None, "stale", "parity", "repeat", "leaked_update", "routes", "repeat_exception", "zero_parity"]
    ]
    + [(False, None)],
)
async def test_check_detects_missing_update_mismatch_and_repeat_noise(
    monkeypatch, checks, fault, colocated, replay, lora
):
    current = {"updated": False, "published": False, "trainer_on_gpu": True, "inference": "asleep"}
    calls = []
    route = np.zeros((2, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(checks, "build_probe_sequences", lambda _: [[1, 2]])
    monkeypatch.setattr(checks, "resolve_policy_model_name", lambda _: "adapter")
    monkeypatch.setattr(checks.ray, "get", lambda value: value)

    def build_batch(sequences, pad_id, routes=None):
        assert sequences == [[1, 2]]
        if replay:
            np.testing.assert_array_equal(routes, [route])
        else:
            assert routes is None

    monkeypatch.setattr(checks, "build_batch", build_batch)

    def score_trainer(*_):
        assert current["trainer_on_gpu"]
        if colocated:
            assert current["inference"] == "asleep"
        if replay:
            assert calls and current["published"] == current["updated"]
        return [-1.0 if current["updated"] else -2.0]

    monkeypatch.setattr(checks, "score_trainer", score_trainer)

    def offload(offload_optimizer, offload_model):
        assert current["trainer_on_gpu"]
        if offload_optimizer:
            assert current["inference"] == "asleep" and offload_model == lora
        else:
            assert not lora and current["inference"] == "weights" and offload_model
        if offload_model:
            current["trainer_on_gpu"] = False

    def backload(backload_optimizer, backload_model):
        assert current["inference"] == "asleep"
        assert not current["trainer_on_gpu"] and backload_model and not backload_optimizer
        current["trainer_on_gpu"] = True

    async def wake_up(tags):
        if tags == ["weights"]:
            assert current["trainer_on_gpu"] == (not lora)
            assert current["inference"] == "asleep"
            current["inference"] = "weights"
        else:
            assert not current["trainer_on_gpu"]
            assert tags == ["kv_cache"] and current["inference"] == "weights"
            current["inference"] = "ready"

    async def sleep():
        assert current["inference"] == "ready" and not current["trainer_on_gpu"]
        current["inference"] = "asleep"

    policy = SimpleNamespace(
        offload_to_cpu=offload,
        backload_to_gpu=backload,
        async_run_ray_method=lambda *_: current.update(updated=True),
    )
    client = SimpleNamespace(wake_up=wake_up, sleep=sleep, model_name="base")

    async def publish(*_):
        if colocated:
            assert current["trainer_on_gpu"] == (not lora) and current["inference"] == "weights"
        if fault != "stale":
            current["published"] = current["updated"]

    async def score(*_):
        if colocated:
            assert not current["trainer_on_gpu"] and current["inference"] == "ready"
        updated = current["published"] or (fault == "leaked_update" and current["updated"])
        value = -1.0 if updated else -2.0
        if (updated and fault == "parity") or (not updated and fault == "zero_parity"):
            value += 0.1
        if updated and fault == "repeat" and len(calls) == 4:
            value += 0.001
        if fault == "repeat_exception" and len(calls) == 1:
            raise RuntimeError("repeat completion failed")
        calls.append(value)
        captured = route + 1 if fault == "routes" and len(calls) == 5 else route
        return [value], [captured] if replay else None

    monkeypatch.setattr(checks, "publish", publish)
    monkeypatch.setattr(checks, "score_sampler", score)
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=8 if lora else 0)),
                megatron_config=SimpleNamespace(moe_enable_routing_replay=replay),
            ),
            placement=SimpleNamespace(colocate_all=colocated),
        ),
        generator=SimpleNamespace(inference_engine=SimpleNamespace(enable_return_routed_experts=replay)),
    )
    report = {}
    call = checks.check_logprobs(policy, client, cfg, SimpleNamespace(pad_token_id=0), report)
    if fault == "repeat_exception":
        with pytest.raises(RuntimeError, match="repeat completion failed"):
            await call
        assert report["zero"]["inference"] == [-2.0]
        if replay:
            assert report["zero"]["routes"] == [route.tolist()]
        assert "repeat" not in report["zero"]
    elif fault and (fault != "routes" or replay):
        with pytest.raises(AssertionError):
            await call
        assert report["perturbed"]
        if fault == "zero_parity":
            assert report["perturbed"]["trainer"] == report["perturbed"]["inference"] == [-1.0]
            assert report["perturbed"]["repeat"] == [-1.0]
            assert report["perturbed"]["stale"] == [-1.9]
    else:
        result = await call
        assert result is report
        if lora:
            assert result["perturbed"]["trainer"] == result["perturbed"]["inference"] == [-1.0]
            assert result["perturbed"]["stale"] == [-2.0]
        else:
            assert result["full_ft"]["trainer"] == result["full_ft"]["inference"] == [-2.0]
            assert not current["updated"]
        if colocated:
            assert current["trainer_on_gpu"] and current["inference"] == "asleep"


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_routes", [False, True])
@pytest.mark.parametrize("fault", [None, "tokens", "short_routes", "float_routes", "missing_routes", "nonfinite"])
async def test_completion_pairs_scores_with_exact_prompt_routes(checks, fault, capture_routes):
    tokens = [3, 7, 11]
    routes = np.arange(6, dtype=np.int64).reshape(3, 2, 1)
    if fault == "short_routes":
        routes = routes[:-1]
    if fault == "float_routes":
        routes = routes.astype(float)
    buffer = io.BytesIO()
    np.save(buffer, routes, allow_pickle=False)
    choice = {
        "prompt_token_ids": tokens if fault != "tokens" else [3, 7, 12],
        "prompt_logprobs": [None, {"7": {"logprob": -0.2}}, {"11": {"logprob": -0.3}}],
        "routed_experts": base64.b64encode(buffer.getvalue()).decode(),
    }
    if fault == "missing_routes":
        del choice["routed_experts"]
    if fault == "nonfinite":
        choice["prompt_logprobs"][1]["7"]["logprob"] = float("nan")

    async def completion(payload):
        body = payload["json"]
        assert body["prompt"] == tokens
        assert ("routed_experts_prompt_start" in body) == capture_routes
        if capture_routes:
            assert body["routed_experts_prompt_start"] == 0
        assert body["max_tokens"] == 1 and not body["add_special_tokens"]
        return {"choices": [choice]}

    async def reset():
        pass

    call = checks.score_sampler(
        SimpleNamespace(completion=completion, reset_prefix_cache=reset),
        [tokens],
        "adapter",
        capture_routes=capture_routes,
    )
    if fault in ("tokens", "nonfinite") or (capture_routes and fault):
        with pytest.raises((AssertionError, ValueError, KeyError)):
            await call
    else:
        scores, captured = await call
        assert scores == [-0.2, -0.3]
        if capture_routes:
            np.testing.assert_array_equal(captured, [routes])
            batch = checks.build_batch([tokens, tokens[:2]], 0, [captured[0], captured[0][:2]])
            assert batch["router_padding_mask"].tolist() == [[False, False, False], [True, False, False]]
            assert batch["response_mask"].sum().item() == 3
        else:
            assert captured is None
