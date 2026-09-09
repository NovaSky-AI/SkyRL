"""CPU checks for the configured GLM numerical runner."""

import json
import sys

import pytest

from examples.tinker.glm53 import run_lora_logprobs
from examples.tinker.glm53.run_lora_logprobs import validate_config


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
