"""Paired completion contract and synthetic replay phase checks."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from examples.model_checks import megatron_lora, run_lora_logprobs
from examples.model_checks.paired_completion import score_with_routes


def make_choice(tokens, routes=None):
    payload = io.BytesIO()
    np.save(payload, np.zeros((len(tokens), 78, 8), dtype=np.int32) if routes is None else routes)
    return {
        "prompt_token_ids": tokens.copy(),
        "prompt_logprobs": [None] + [{str(token): {"logprob": -index}} for index, token in enumerate(tokens[1:], 1)],
        "routed_experts": base64.b64encode(payload.getvalue()).decode(),
        "token_ids": [999],
    }


@pytest.mark.asyncio
async def test_completion_pairs_full_routes_and_selected_scores_without_generated_token():
    sequences = [list(range(65)), list(range(129))]
    requests = []

    async def reset():
        requests.append("reset")

    async def complete(request):
        body = request["json"]
        requests.append(body)
        assert body == {
            "model": "adapter",
            "prompt": body["prompt"],
            "max_tokens": 1,
            "temperature": 1.0,
            "n": 1,
            "stream": False,
            "prompt_logprobs": 0,
            "add_special_tokens": False,
            "return_token_ids": True,
            "routed_experts_prompt_start": 0,
        }
        return {"choices": [make_choice(body["prompt"])]}

    scores, routes = await score_with_routes(
        SimpleNamespace(reset_prefix_cache=reset, completion=complete), sequences, "adapter"
    )
    assert requests[0] == "reset"
    assert [request["prompt"] for request in requests[1:]] == sequences
    assert scores == [-index for length in (65, 129) for index in range(1, length)]
    assert [route.shape for route in routes] == [(65, 78, 8), (129, 78, 8)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "tokens",
        "length",
        "first",
        "missing",
        "nan",
        "inf",
        "negative_inf",
        "sentinel",
        "shape",
        "float",
        "negative",
        "range",
    ],
)
async def test_completion_rejects_unusable_evidence(defect):
    tokens = list(range(65))
    routes = np.zeros((65, 78, 8), dtype=np.int32)
    if defect == "shape":
        routes = routes[:-1]
    if defect == "float":
        routes = routes.astype(float)
    if defect in ("negative", "range"):
        routes[0, 0, 0] = -1 if defect == "negative" else 256
    choice = make_choice(tokens, routes)
    if defect == "tokens":
        choice["prompt_token_ids"][0] = 999
    if defect == "length":
        choice["prompt_logprobs"].pop()
    if defect == "first":
        choice["prompt_logprobs"][0] = {}
    if defect == "missing":
        choice["prompt_logprobs"][1] = {}
    if defect in ("nan", "inf", "negative_inf", "sentinel"):
        choice["prompt_logprobs"][1]["1"]["logprob"] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative_inf": -float("inf"),
            "sentinel": -9999,
        }[defect]

    async def reset():
        pass

    async def complete(request):
        return {"choices": [choice]}

    with pytest.raises((ValueError, KeyError)):
        await score_with_routes(SimpleNamespace(reset_prefix_cache=reset, completion=complete), [tokens], "adapter")


def test_replay_batch_keeps_full_prompt_routes_and_masks_only_padding():
    sequences = [[1, 2, 3], [4, 5, 6, 7, 8]]
    routes = [np.ones((3, 78, 8), dtype=np.int32), np.full((5, 78, 8), 2, dtype=np.int32)]
    batch = megatron_lora.build_batch(sequences, 0, routes)
    assert batch["router_padding_mask"].tolist() == [[True, True, False, False, False], [False] * 5]
    assert batch["rollout_expert_indices"][0, -3:].eq(1).all()
    assert batch["rollout_expert_indices"][1].eq(2).all()
    assert batch["response_mask"].sum().item() == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat_shift,updated_repeat_shift", [(0.0, 0.0), (2e-6, 0.0), (0.0, 2e-6)])
async def test_replayed_phases_preserve_prepublication_evidence_and_use_fresh_routes(
    monkeypatch, repeat_shift, updated_repeat_shift
):
    calls = []
    route_ids = iter(range(6))

    async def paired(client, sequences, model):
        phase = next(route_ids)
        calls.append(("score", phase, model))
        scores = [-2.0 + (repeat_shift if phase == 2 else 0.0)] if phase < 4 else [-1.8]
        if phase == 5:
            scores[0] += updated_repeat_shift
        return scores, [np.full((2, 78, 8), phase, dtype=np.int32)]

    def build(sequences, pad, routes):
        return int(routes[0][0, 0, 0])

    trainer_calls = []

    def trainer(policy, batch):
        trainer_calls.append(batch)
        return {"unreplayed": [-2.3], 1: [-2.0] if trainer_calls.count(1) == 1 else [-1.7], 2: [-2.0], 4: [-1.8]}[batch]

    async def publish(*args):
        calls.append(("publish",))

    monkeypatch.setattr(run_lora_logprobs, "score_with_routes", paired)
    monkeypatch.setattr(run_lora_logprobs, "build_batch", build)
    monkeypatch.setattr(run_lora_logprobs, "score_trainer", trainer)
    monkeypatch.setattr(run_lora_logprobs, "publish", publish)
    monkeypatch.setattr(run_lora_logprobs, "resolve_policy_model_name", lambda cfg: "adapter")
    monkeypatch.setattr(run_lora_logprobs, "perturb_trainer", lambda policy, multiplier: {"multiplier": multiplier})
    report = {}
    args = SimpleNamespace(mean_atol=0.05, max_atol=0.5, lora_b_multiplier=10)
    coroutine = run_lora_logprobs.check_replayed_policy(
        None, SimpleNamespace(model_name="base"), None, "unreplayed", [[1, 2]], 0, report, args
    )
    if repeat_shift:
        with pytest.raises(AssertionError, match="repeat_noise exceeds"):
            await coroutine
        assert calls.count(("publish",)) == 1
        return
    if updated_repeat_shift:
        with pytest.raises(AssertionError, match="updated_repeat_noise exceeds"):
            await coroutine
        assert report["updated_repeat_routes"][0][0][0][0] == 5
        return
    await coroutine
    assert report["updated_repeat_noise"]["max_abs"] == 0
    assert trainer_calls == ["unreplayed", 1, 2, 1, 4]
    assert report["trainer_updated_before_publication"] == [-1.7]
    assert report["trainer_updated"] == [-1.8]
    assert report["stale_parity_before_publication"]["mean_abs"] == pytest.approx(0.3)
    assert report["zero_parity_unreplayed"]["mean_abs"] == pytest.approx(0.3)
    assert report["updated_parity"]["max_abs"] == 0
    assert [call[0] for call in calls] == ["score", "publish", "score", "score", "score", "publish", "score", "score"]


@pytest.mark.parametrize(
    "selector",
    [
        "trainer.policy.megatron_config.moe_enable_routing_replay",
        "generator.inference_engine.enable_return_routed_experts",
    ],
)
def test_capture_and_replay_must_be_enabled_together(selector):
    with pytest.raises(ValueError, match="enabled together"):
        run_lora_logprobs.validate_config({selector: True})
