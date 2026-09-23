from types import SimpleNamespace

import pytest

from examples.model_checks import check_logprobs as checks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,colocated", [(None, False), ("stale", False), ("parity", False), ("repeat", False), (None, True)]
)
async def test_check_detects_missing_update_mismatch_and_repeat_noise(monkeypatch, fault, colocated):
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
