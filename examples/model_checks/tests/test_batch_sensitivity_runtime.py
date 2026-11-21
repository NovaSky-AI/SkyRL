"""Native-import checks; run in the same Megatron environment as the GPU diagnostic."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.model_checks import run_batch_sensitivity as runner


def test_causal_tokens_and_masks_are_identical_across_batch_sizes():
    tokens = list(range(1, 66))
    single = runner.build_batch(tokens, 1, 0)
    duplicate = runner.build_batch(tokens, 2, 0)
    for key in ("sequences", "attention_mask", "response_mask", "loss_mask"):
        assert torch.equal(single[key].expand_as(duplicate[key]), duplicate[key])
    assert single["sequences"][0].tolist() == tokens
    assert single["response_mask"].sum().item() == 64
    assert duplicate["response_mask"].sum(dim=1).tolist() == [64, 64]


def test_worker_rejects_nonzero_adapter_and_preserves_production_scoring_path():
    worker = object.__new__(runner.BatchSensitivityWorker)
    parameter = torch.zeros(2, 3)
    worker.actor_module = [SimpleNamespace(named_parameters=lambda: [("layer.adapter.linear_out.weight", parameter)])]
    calls = []
    worker.forward = lambda data, loss_fn: calls.append((data, loss_fn))
    worker.score_zero_adapter("batch")
    assert calls == [("batch", None)]
    parameter[0, 0] = 1
    with pytest.raises(RuntimeError, match="zero-initialized"):
        worker.score_zero_adapter("batch")
    assert calls == [("batch", None)]
    worker.actor_module = []
    with pytest.raises(RuntimeError, match="No LoRA"):
        worker.score_zero_adapter("batch")
    assert calls == [("batch", None)]


def test_dispatch_keeps_rows_separate_and_rejects_wrong_scored_length(monkeypatch):
    output = SimpleNamespace(loss_fn_outputs=[{"logprobs": [-1.0, -2.0]}, {"logprobs": [-1.1, -2.1]}])
    calls = []

    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    monkeypatch.setattr(runner.ray, "get", lambda result: result)
    monkeypatch.setattr(runner.WorkerOutput, "cat", lambda *args: output)
    policy = SimpleNamespace(actor_infos=[], async_run_ray_method=dispatch)
    assert runner.score_batch(policy, "batch", 2, 2) == [[-1.0, -2.0], [-1.1, -2.1]]
    assert calls == [(("mesh", "score_zero_adapter"), {"data": "batch"})]
    output.loss_fn_outputs[1]["logprobs"].append(-3.0)
    with pytest.raises(ValueError):
        runner.score_batch(policy, "batch", 2, 2)


def make_args(tmp_path):
    fixtures = Path(__file__).parents[1] / "fixtures"
    fixture = fixtures / "qwen3_8b_tokens.json"
    model = tmp_path / json.loads(fixture.read_text())["model_revision"]
    return SimpleNamespace(
        model=model, fixture=fixture, backend_config=fixtures / "qwen3_8b_config.json", output_dir=tmp_path
    )


@pytest.mark.parametrize(
    "override",
    [
        {"trainer.bf16": False},
        {"trainer.policy.model.lora.rank": 0},
        {"trainer.policy.megatron_config.context_parallel_size": 2},
        {"trainer.placement.policy_num_gpus_per_node": 2},
        {"trainer.remove_microbatch_padding": True},
        {"trainer.max_tokens_per_microbatch": 129},
    ],
)
def test_preflight_rejects_configs_that_change_the_control(tmp_path, override):
    args = make_args(tmp_path)
    config = json.loads(args.backend_config.read_text())
    config.update(override)
    args.backend_config = tmp_path / "config.json"
    args.backend_config.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runner.load_config(args, 65)


def test_tokenizer_without_padding_or_eos_is_rejected_before_batching(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "get_tokenizer", lambda _: SimpleNamespace(pad_token_id=None, eos_token_id=None))
    with pytest.raises(ValueError, match="pad_token_id or eos_token_id"):
        runner.run(make_args(tmp_path), {})


def test_ray_cleanup_runs_even_when_saving_a_failed_run_raises(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    shutdowns = []
    monkeypatch.setattr(runner, "get_tokenizer", lambda _: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(runner.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(runner.ray, "shutdown", lambda: shutdowns.append(True))

    def fail_initialize(_cfg):
        raise RuntimeError("initialization failed")

    def fail_write(*_args):
        raise OSError("report storage unavailable")

    monkeypatch.setattr(runner, "initialize_ray", fail_initialize)
    monkeypatch.setattr(runner, "write_report", fail_write)
    with pytest.raises(OSError, match="report storage"):
        runner.run(args, {})
    assert shutdowns == [True]
