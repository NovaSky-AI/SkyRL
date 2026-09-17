"""CPU checks for the native LoRA publication runner."""

import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from examples.model_checks import megatron_lora, run_lora_logprobs
from examples.model_checks.run_lora_logprobs import validate_config


def test_trainer_scores_preserve_unequal_length_sample_positions(monkeypatch):
    batch = {"response_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])}
    output = SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": [-1.0, -2.0]},
            {"logprobs": [-3.0, -4.0, -5.0, -6.0]},
        ]
    )
    policy = SimpleNamespace(actor_infos=[], async_run_ray_method=lambda *args, **kwargs: output)
    monkeypatch.setattr(megatron_lora.ray, "get", lambda value: value)
    monkeypatch.setattr(megatron_lora.WorkerOutput, "cat", lambda *args: output)
    assert megatron_lora.score_trainer(policy, batch) == [
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
    ]

    output.loss_fn_outputs[0]["logprobs"].append(0.0)
    with pytest.raises(AssertionError):
        megatron_lora.score_trainer(policy, batch)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
def test_trainer_nonfinite_scores_do_not_enter_json_receipt(monkeypatch, bad_score):
    batch = {"response_mask": torch.tensor([[1]])}
    output = SimpleNamespace(loss_fn_outputs=[{"logprobs": [bad_score]}])
    policy = SimpleNamespace(actor_infos=[], async_run_ray_method=lambda *args, **kwargs: output)
    monkeypatch.setattr(megatron_lora.ray, "get", lambda value: value)
    monkeypatch.setattr(megatron_lora.WorkerOutput, "cat", lambda *args: output)
    report = {"passed": False}
    with pytest.raises(ValueError, match="nonfinite trainer logprobs"):
        report["trainer_updated"] = megatron_lora.score_trainer(policy, batch)
    assert json.loads(json.dumps(report, allow_nan=False)) == {"passed": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
async def test_sampler_nonfinite_scores_do_not_enter_json_receipt(bad_score):
    async def reset_prefix_cache():
        return None

    async def sample(request):
        return {"prompt_logprobs": [None, bad_score]}

    client = SimpleNamespace(reset_prefix_cache=reset_prefix_cache, sample=sample)
    report = {"passed": False}
    with pytest.raises(ValueError, match="nonfinite sampler logprobs"):
        report["updated"] = await megatron_lora.score_sampler(client, [[1, 2]], "adapter")
    assert json.loads(json.dumps(report, allow_nan=False)) == {"passed": False}


@pytest.mark.parametrize(
    "key,value",
    [
        ("strategy", "fsdp"),
        ("trainer.placement.colocate_all", True),
        ("trainer.policy.model.lora.rank", 0),
        ("trainer.policy.megatron_config.lora_config.merge_lora", True),
        ("generator.inference_engine.run_engines_locally", False),
        ("generator.inference_engine.external_proxy_url", "http://example.com"),
        ("generator.inference_engine.external_server_urls", ["http://example.com"]),
        ("generator.inference_engine.enable_pd", True),
    ],
)
def test_diagnostic_rejects_unsupported_or_externally_owned_runtime(key, value):
    config = {
        "strategy": "megatron",
        "trainer.placement.colocate_all": False,
        "trainer.policy.model.lora.rank": 32,
        "trainer.policy.megatron_config.lora_config.merge_lora": False,
        "generator.inference_engine.run_engines_locally": True,
    }
    validate_config(config)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cli_records_success_only_after_runtime_finishes(tmp_path, monkeypatch, cleanup_fails):
    output_dir = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lora_logprobs",
            "--backend-config",
            "config.json",
            "--output-dir",
            str(output_dir),
            "--mean-atol",
            "0.05",
            "--max-atol",
            "0.5",
        ],
    )

    async def run_fixture(args, report):
        assert args.lora_b_multiplier == 10
        report["updated_parity"] = {"mean_abs": 0.01}
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(run_lora_logprobs, "run", run_fixture)
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            run_lora_logprobs.main()
    else:
        run_lora_logprobs.main()
    report = json.loads((output_dir / "logprobs.json").read_text())
    assert report["passed"] is (not cleanup_fails)
    assert report["updated_parity"]["mean_abs"] == 0.01


@pytest.mark.parametrize("multiplier", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_stimulus_before_creating_output(tmp_path, monkeypatch, multiplier):
    output_dir = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lora_logprobs",
            "--backend-config",
            "missing.json",
            "--output-dir",
            str(output_dir),
            "--mean-atol",
            "0.05",
            "--max-atol",
            "0.5",
            "--lora-b-multiplier",
            multiplier,
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_lora_logprobs.main()
    assert error.value.code == 2
    assert not output_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("wrong_publication", [False, True])
@pytest.mark.parametrize("update_size", [0.2, 0.01])
async def test_run_checks_the_actual_published_update_and_cleans_up(
    monkeypatch, tmp_path, wrong_publication, update_size, replay
):
    calls = []
    changed = False
    publications = 0
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(
                model=SimpleNamespace(path="model"),
                megatron_config=SimpleNamespace(moe_enable_routing_replay=replay),
            )
        )
    )
    client = SimpleNamespace(model_name="base")
    monkeypatch.setattr(run_lora_logprobs, "load_config", lambda *args: cfg)
    monkeypatch.setattr(
        run_lora_logprobs,
        "get_tokenizer",
        lambda *args: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(run_lora_logprobs, "build_sequences", lambda *args: [[1, 2, 3]])
    monkeypatch.setattr(run_lora_logprobs, "build_batch", lambda sequences, pad, routes: routes)
    monkeypatch.setattr(run_lora_logprobs, "resolve_policy_model_name", lambda *args: "adapter")

    @asynccontextmanager
    async def open_runtime(*args):
        try:
            yield "policy", client
        finally:
            snapshot = json.loads((tmp_path / "logprobs.json").read_text())
            assert snapshot["stale_parity"] == report["stale_parity"]
            assert not (tmp_path / "logprobs.json.tmp").exists()
            calls.append("cleanup")

    async def publish(*args):
        nonlocal publications
        publications += 1
        calls.append("publish")

    def perturb(*args):
        nonlocal changed
        assert args == ("policy", 32)
        changed = True
        calls.append("update_trainer")
        return {"seed": 0}

    def score_trainer(policy, routes):
        if replay:
            assert routes[0].unique().tolist() == [publications]
        else:
            assert routes is None
        calls.append("score_trainer")
        return [-2.0 + update_size, -3.0 + update_size] if changed else [-2.0, -3.0]

    async def score_routed_sampler(client, sequences, model):
        calls.append("capture_routes")
        return await score_sampler(client, sequences, model), [torch.full((3, 2, 1), publications)]

    async def score_sampler(client, sequences, model):
        calls.append(f"score_{model}")
        if publications < 2:
            return [-2.0, -3.0]
        return [-2.2, -3.2] if wrong_publication else [-1.8, -2.8]

    for name, function in [
        ("open_runtime", open_runtime),
        ("publish", publish),
        ("perturb_trainer", perturb),
        ("score_trainer", score_trainer),
        ("score_sampler", score_sampler),
        ("score_routed_sampler", score_routed_sampler),
    ]:
        monkeypatch.setattr(run_lora_logprobs, name, function)
    args = SimpleNamespace(
        backend_config="config.json",
        output_dir=tmp_path,
        mean_atol=0.05,
        max_atol=0.5,
        lora_b_multiplier=32,
    )
    report = {}
    if update_size < 0.05:
        with pytest.raises(AssertionError, match="insufficient test stimulus"):
            await run_lora_logprobs.run(args, report)
        assert publications == 1
    elif wrong_publication:
        with pytest.raises(AssertionError):
            await run_lora_logprobs.run(args, report)
    else:
        await run_lora_logprobs.run(args, report)
        assert report["update_delta"]["mean_abs"] < 1e-12
    expected = [
        "score_base",
        "publish",
        "score_adapter",
        "score_trainer",
        "score_adapter",
        "score_trainer",
        "update_trainer",
        "score_trainer",
        "score_adapter",
    ]
    if update_size >= 0.05:
        expected += ["publish"]
        if replay:
            expected += ["score_adapter", "score_trainer"]
            assert report["zero_routes"] != report["updated_routes"]
            assert report["trainer_prepublication"] == report["trainer_updated"]
            assert report["prepublication_stale_parity"]["mean_abs"] >= 0.05
        if not replay:
            expected += ["score_adapter"]
    if replay:
        expected = [
            item
            for call in expected
            for item in (
                ["capture_routes", call] if call.startswith("score_adapter") or call == "score_base" else [call]
            )
        ]
    assert calls == expected + ["cleanup"]


@pytest.mark.parametrize(
    "nodes,gpus,tp,pp,cp,ep,temperature,error",
    [
        (1, 8, 8, 1, 1, 8, 1.0, None),
        (1, 8, 4, 1, 2, 8, 1.0, None),
        (2, 8, 8, 2, 1, 8, 1.0, None),
        (2, 8, 4, 1, 2, 8, 1.0, None),
        (1, 8, 8, 1, 1, 8, 0.7, "temperature=1"),
        (1, 8, 1, 1, 1, 8, 1.0, "DP=1 or DP=2"),
        (1, 8, 3, 1, 1, 1, 1.0, "divisible"),
        (1, 8, 4, 2, 2, 8, 1.0, "divisible"),
    ],
)
def test_load_rejects_incomparable_scores_before_startup(
    tmp_path, monkeypatch, nodes, gpus, tp, pp, cp, ep, temperature, error
):
    overrides = {
        "strategy": "megatron",
        "trainer.placement.colocate_all": False,
        "trainer.policy.model.lora.rank": 32,
        "trainer.policy.megatron_config.lora_config.merge_lora": False,
        "generator.inference_engine.run_engines_locally": True,
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(overrides))
    parallel = SimpleNamespace(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            algorithm=SimpleNamespace(temperature=temperature),
            placement=SimpleNamespace(policy_num_nodes=nodes, policy_num_gpus_per_node=gpus),
            policy=SimpleNamespace(megatron_config=parallel),
        )
    )
    monkeypatch.setattr(run_lora_logprobs.SkyRLTrainConfig, "from_cli_overrides", lambda _: cfg)
    if error:
        with pytest.raises(ValueError, match=error):
            run_lora_logprobs.load_config(path, tmp_path)
        assert not (tmp_path / "backend-config.json").exists()
    else:
        assert run_lora_logprobs.load_config(path, tmp_path) is cfg
    assert cfg.trainer.algorithm.temperature == temperature


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [2, 3, 4])
async def test_route_capture_requires_full_fixed_sequence(length):
    events = []

    async def reset_prefix_cache():
        events.append("reset")

    async def generate(request, model):
        assert events == ["reset"]
        assert request["prompt_token_ids"] == [[1, 2, 3]]
        assert request["sampling_params"]["routed_experts_prompt_start"] == 0
        assert model == "adapter"
        return {
            "rollout_expert_indices": [torch.zeros((length, 2, 1), dtype=torch.int64)],
            "prompt_logprobs": [[None, -1.0, -2.0]],
        }

    client = SimpleNamespace(reset_prefix_cache=reset_prefix_cache, generate=generate)
    if length == 3:
        scores, routes = await megatron_lora.score_routed_sampler(client, [[1, 2, 3]], "adapter")
        assert routes[0].shape == (3, 2, 1)
        assert scores == [-1.0, -2.0]
    else:
        with pytest.raises(ValueError, match="every fixed probe token"):
            await megatron_lora.score_routed_sampler(client, [[1, 2, 3]], "adapter")


def test_routed_batch_preserves_unequal_sequence_alignment():
    sequences = [[1, 2, 3], [4, 5, 6, 7]]
    routes = [np.full((len(tokens), 2, 1), index + 1, dtype=np.int64) for index, tokens in enumerate(sequences)]
    batch = megatron_lora.build_batch(sequences, 0, routes)
    for row, route in enumerate(routes):
        valid = ~batch["router_padding_mask"][row]
        assert valid.sum() == len(route)
        assert torch.equal(batch["rollout_expert_indices"][row][valid], torch.from_numpy(route))
    assert batch["response_mask"].sum(dim=1).tolist() == [2, 3]


@pytest.mark.parametrize("replay,capture", [(True, False), (False, True)])
def test_replay_requires_matching_capture_before_runtime(replay, capture):
    with pytest.raises(ValueError, match="enabled together"):
        validate_config(
            {
                "trainer.policy.megatron_config.moe_enable_routing_replay": replay,
                "generator.inference_engine.enable_return_routed_experts": capture,
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [None, [None], [0.0, -1.0, -2.0], [None, None, -2.0], [None, float("nan"), -2.0]])
async def test_routed_scores_reject_missing_misaligned_or_nonfinite_tokens(values):
    async def generate(request, model):
        assert request["sampling_params"]["prompt_logprobs"] == 0
        return {
            "rollout_expert_indices": [np.zeros((3, 2, 1), dtype=np.int32)],
            "prompt_logprobs": None if values is None else [values],
        }

    client = SimpleNamespace(reset_prefix_cache=AsyncMock(), generate=generate)
    with pytest.raises(ValueError):
        await megatron_lora.score_routed_sampler(client, [[1, 2, 3]], "adapter")
