"""CPU checks for the configured GLM numerical runner."""

import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import torch

from examples.model_checks import megatron_lora
from examples.tinker.glm53 import run_lora_logprobs
from examples.tinker.glm53.run_lora_logprobs import validate_config


def test_trainer_scores_preserve_unequal_length_sample_positions(monkeypatch):
    batch = {"response_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])}
    output = SimpleNamespace(loss_fn_outputs=[{"logprobs": [-1.0, -2.0]}, {"logprobs": [-3.0, -4.0, -5.0, -6.0]}])
    policy = SimpleNamespace(actor_infos=[], async_run_ray_method=lambda *args, **kwargs: output)
    monkeypatch.setattr(megatron_lora.ray, "get", lambda value: value)
    monkeypatch.setattr(megatron_lora.WorkerOutput, "cat", lambda *args: output)
    assert megatron_lora.score_trainer(policy, batch) == [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]

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
        ],
    )

    async def run_fixture(args, report):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_publication", [False, True])
@pytest.mark.parametrize("update_size", [0.2, 0.01])
async def test_run_checks_the_actual_published_update_and_cleans_up(
    monkeypatch, tmp_path, wrong_publication, update_size
):
    calls = []
    changed = False
    publications = 0
    cfg = SimpleNamespace(trainer=SimpleNamespace(policy=SimpleNamespace(model=SimpleNamespace(path="model"))))
    client = SimpleNamespace(model_name="base")
    monkeypatch.setattr(run_lora_logprobs, "load_config", lambda *args: cfg)
    monkeypatch.setattr(run_lora_logprobs, "get_tokenizer", lambda *args: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(run_lora_logprobs, "build_sequences", lambda *args: [[1, 2, 3]])
    monkeypatch.setattr(run_lora_logprobs, "build_batch", lambda *args: "batch")
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
        changed = True
        calls.append("update_trainer")
        return {"seed": 0}

    def score_trainer(*args):
        calls.append("score_trainer")
        return [-2.0 + update_size, -3.0 + update_size] if changed else [-2.0, -3.0]

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
    ]:
        monkeypatch.setattr(run_lora_logprobs, name, function)
    args = SimpleNamespace(backend_config="config.json", output_dir=tmp_path, mean_atol=0.05)
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
        expected += ["publish", "score_adapter"]
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
