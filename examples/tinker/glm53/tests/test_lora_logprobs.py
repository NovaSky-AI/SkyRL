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
            "--delta-mean-atol",
            "0.005",
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
async def test_run_checks_the_actual_published_update_and_cleans_up(monkeypatch, tmp_path, wrong_publication):
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
        return [-1.98, -2.98] if changed else [-2.0, -3.0]

    async def score_sampler(client, sequences, model):
        calls.append(f"score_{model}")
        if publications < 2:
            return [-2.0, -3.0]
        return [-2.02, -3.02] if wrong_publication else [-1.98, -2.98]

    for name, function in [
        ("open_runtime", open_runtime),
        ("publish", publish),
        ("perturb_trainer", perturb),
        ("score_trainer", score_trainer),
        ("score_sampler", score_sampler),
    ]:
        monkeypatch.setattr(run_lora_logprobs, name, function)
    args = SimpleNamespace(backend_config="config.json", output_dir=tmp_path, mean_atol=0.05, delta_mean_atol=0.005)
    report = {}
    if wrong_publication:
        with pytest.raises(AssertionError):
            await run_lora_logprobs.run(args, report)
    else:
        await run_lora_logprobs.run(args, report)
        assert report["update_delta"]["mean_abs"] < 1e-12
    assert calls == [
        "score_base",
        "publish",
        "score_adapter",
        "score_trainer",
        "score_adapter",
        "score_trainer",
        "update_trainer",
        "score_trainer",
        "score_adapter",
        "publish",
        "score_adapter",
        "cleanup",
    ]
