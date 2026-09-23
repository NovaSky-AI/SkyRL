from types import SimpleNamespace

import pytest

from examples.model_checks import check_logprobs as checks


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "stale", "parity", "repeat"])
async def test_check_detects_missing_update_mismatch_and_repeat_noise(monkeypatch, fault):
    current = {"updated": False}
    sampler_calls = []
    monkeypatch.setattr(checks, "build_probe_sequences", lambda _: [[1, 2]])
    monkeypatch.setattr(checks, "build_batch", lambda *_: None)
    monkeypatch.setattr(checks, "resolve_policy_model_name", lambda _: "adapter")
    monkeypatch.setattr(checks, "perturb_trainer", lambda _: current.update(updated=True))
    monkeypatch.setattr(checks, "score_trainer", lambda *_: [-1.0 if current["updated"] else -2.0])

    async def publish(*_):
        pass

    async def score(*_):
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
            placement=SimpleNamespace(colocate_all=False),
        )
    )
    call = checks.check_logprobs(None, None, cfg, SimpleNamespace(pad_token_id=0))
    if fault:
        with pytest.raises(AssertionError):
            await call
    else:
        result = await call
        assert result["perturbed"] == {"trainer": [-1.0], "inference": [-1.0]}
