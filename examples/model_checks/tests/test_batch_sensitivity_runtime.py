"""Native-import checks; run in the same Megatron environment as the GPU diagnostic."""

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
    with pytest.raises(AssertionError):
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
