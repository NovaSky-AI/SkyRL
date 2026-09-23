import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    path = Path(__file__).parents[3] / "examples/model_checks/run_nemotron_logprobs.py"
    spec = importlib.util.spec_from_file_location("logprob_example_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,colocated", [(None, False), ("stale", False), ("parity", False), ("repeat", False), (None, True)]
)
async def test_check_detects_missing_update_mismatch_and_repeat_noise(monkeypatch, checks, fault, colocated):
    current = {"updated": False, "trainer_on_gpu": True, "inference": "asleep"}
    sampler_calls = []
    monkeypatch.setattr(checks, "build_probe_sequences", lambda _: [[1, 2]])
    monkeypatch.setattr(checks, "build_batch", lambda *_: None)
    monkeypatch.setattr(checks, "resolve_policy_model_name", lambda _: "adapter")
    monkeypatch.setattr(checks, "perturb_trainer", lambda _: current.update(updated=True))

    def score_trainer(*_):
        assert current["trainer_on_gpu"]
        if colocated:
            assert current["inference"] == "asleep"
        return [-1.0 if current["updated"] else -2.0]

    monkeypatch.setattr(checks, "score_trainer", score_trainer)

    def offload(offload_optimizer, offload_model):
        assert current["trainer_on_gpu"]
        if offload_model:
            assert current["inference"] == "published"
            current["trainer_on_gpu"] = False
        else:
            assert offload_optimizer and current["inference"] == "asleep"

    def backload(backload_optimizer, backload_model):
        assert current["inference"] == "asleep"
        assert not current["trainer_on_gpu"] and backload_model and not backload_optimizer
        current["trainer_on_gpu"] = True

    async def wake_up(tags):
        if tags == ["weights"]:
            assert current["trainer_on_gpu"] and current["inference"] == "asleep"
            current["inference"] = "weights"
        else:
            assert tags == ["kv_cache"]
            assert not current["trainer_on_gpu"] and current["inference"] == "published"
            current["inference"] = "ready"

    async def sleep():
        assert current["inference"] == "ready" and not current["trainer_on_gpu"]
        current["inference"] = "asleep"

    policy = SimpleNamespace(offload_to_cpu=offload, backload_to_gpu=backload)
    client = SimpleNamespace(wake_up=wake_up, sleep=sleep)

    async def publish(*_):
        if colocated:
            assert current["trainer_on_gpu"] and current["inference"] == "weights"
            current["inference"] = "published"

    async def score(*_):
        if colocated:
            assert not current["trainer_on_gpu"] and current["inference"] == "ready"
        updated = current["updated"]
        value = -1.0 if updated and fault != "stale" else -2.0
        if updated and fault == "parity":
            value += 0.1
        if updated and fault == "repeat" and len(sampler_calls) == 3:
            value += 0.001
        sampler_calls.append(value)
        return [value]

    monkeypatch.setattr(checks, "publish", publish)
    monkeypatch.setattr(checks, "score_sampler", score)
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(rank=8))),
            placement=SimpleNamespace(colocate_all=colocated),
        )
    )
    call = checks.check_logprobs(policy, client, cfg, SimpleNamespace(pad_token_id=0))
    if fault:
        with pytest.raises(AssertionError):
            await call
    else:
        result = await call
        assert result["perturbed"] == {"trainer": [-1.0], "inference": [-1.0]}
        if colocated:
            assert current["trainer_on_gpu"] and current["inference"] == "asleep"
