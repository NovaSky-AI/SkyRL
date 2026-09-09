"""CPU checks for exact-length data, failure accounting and bounded cleanup."""

import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from tinker import types

from examples.tinker.glm53 import run_client as example
from skyrl.tinker import sdk_checks as module


@pytest.mark.parametrize("context", [2, 7, 32768, 262144])
def test_exact_context_preserves_shift_and_scores_every_position(context):
    datum = module.build_full_context_datum([11, 12, 13], context)
    inputs = datum.model_input.to_ints()
    targets = datum.loss_fn_inputs["target_tokens"].data
    weights = datum.loss_fn_inputs["weights"].data
    assert len(inputs) == len(targets) == len(weights) == context
    assert inputs[1:] == targets[:-1]
    assert all(value == 1 for value in weights)
    assert targets[-1] == [11, 12, 13][context % 3]


@pytest.mark.parametrize("tokens,context", [([], 32), ([1], 1)])
def test_reject_empty_or_invalid_fixture(tokens, context):
    with pytest.raises(ValueError):
        module.build_full_context_datum(tokens, context)


def test_gspo_keeps_reference_scores_targets_and_full_context_masks():
    datum = module.build_full_context_datum([11, 12, 13], 7)
    scores = types.TensorData(data=[-0.5] * 7, dtype="float32", shape=[7])
    reference = SimpleNamespace(loss_fn_outputs=[{"logprobs": scores}, {"logprobs": scores}])
    batch = module.build_gspo_batch([datum, datum], reference)
    for item, advantage in zip(batch, [1.0, -1.0], strict=True):
        fields = item.loss_fn_inputs
        assert fields["logprobs"].data == scores.data
        assert fields["advantages"].data == [advantage] * 7
        assert fields["weights"].data == [1.0] * 7
        assert fields["target_tokens"].data == datum.loss_fn_inputs["target_tokens"].data
        assert batch[0].model_input.to_ints() == datum.model_input.to_ints()
    assert "advantages" not in datum.loss_fn_inputs


def test_failure_records_elapsed_time_without_claiming_completion():
    report = io.StringIO()
    with pytest.raises(RuntimeError, match="worker failed"):
        with module.measure_phase(report, "backward"):
            raise RuntimeError("worker failed")
    records = [json.loads(line) for line in report.getvalue().splitlines()]
    assert [record["status"] for record in records] == ["running", "failed"]
    assert records[-1]["seconds"] >= 0
    assert records[-1]["error_type"] == "RuntimeError"
    assert records[-1]["error"] == "worker failed"


@pytest.mark.parametrize("metadata", [{"is_lora": False}, {"lora_rank": 16}])
def test_reject_explicit_wrong_adapter_metadata(tmp_path, metadata):
    info = types.GetInfoResponse.model_validate(
        {"model_id": "model-test", "model_data": {"model_name": "test-model"}, **metadata}
    )
    trainer = SimpleNamespace(get_info=lambda: info)
    with pytest.raises(ValueError, match="rank-32"):
        module.prepare_full_context_inputs(trainer, SimpleNamespace(output_dir=tmp_path), io.StringIO())


@pytest.mark.parametrize("values", [[-1.0], [-1.0, float("nan")]])
def test_reject_short_or_nonfinite_training_results(values):
    result = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": types.TensorData(data=values, dtype="float32", shape=[len(values)])}],
        metrics={"loss": 1.0},
    )
    with pytest.raises(ValueError, match="finite logprob"):
        module.check_training_result(result, context=2, batch_size=1)


def test_unload_waits_for_terminal_completion_and_preserves_model_identity():
    calls = []

    def respond(request):
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("unload_model"):
            return httpx.Response(200, json={"request_id": "42"})
        if len(calls) == 2:
            return httpx.Response(408, json={"detail": "not ready"})
        return httpx.Response(200, json={"type": "unload_model", "model_id": "model-test"})

    client = httpx.Client(base_url="http://example.com/", transport=httpx.MockTransport(respond))
    with patch.object(module.httpx, "Client", return_value=client):
        module.unload_model("http://example.com", "model-test")
    assert calls == [
        ("/api/v1/unload_model", {"model_id": "model-test"}),
        ("/api/v1/retrieve_future", {"request_id": "42"}),
        ("/api/v1/retrieve_future", {"request_id": "42"}),
    ]


def test_unload_poll_uses_remaining_budget_not_short_http_default():
    clock = [0.0]
    polls = []

    def respond(request):
        if request.url.path.endswith("unload_model"):
            clock[0] = 5.0
            return httpx.Response(200, json={"request_id": "42"})
        polls.append(request.extensions["timeout"]["read"])
        if len(polls) == 1:
            clock[0] = 90.0
            return httpx.Response(408)
        return httpx.Response(200, json={"type": "unload_model", "model_id": "model-test"})

    client = httpx.Client(base_url="http://example.com/", transport=httpx.MockTransport(respond))
    with (
        patch.object(module.httpx, "Client", return_value=client),
        patch.object(module.time, "monotonic", side_effect=lambda: clock[0]),
    ):
        module.unload_model("http://example.com", "model-test")
    assert polls == [115.0, 30.0]


def test_unload_read_timeout_preserves_cause_and_does_not_resubmit():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path.endswith("unload_model"):
            return httpx.Response(200, json={"request_id": "42"})
        raise httpx.ReadTimeout("long poll exceeded budget")

    client = httpx.Client(base_url="http://example.com/", transport=httpx.MockTransport(respond))
    with patch.object(module.httpx, "Client", return_value=client):
        with pytest.raises(TimeoutError, match="polling budget") as failure:
            module.unload_model("http://example.com", "model-test")
    assert isinstance(failure.value.__cause__, httpx.ReadTimeout)
    assert calls == ["/api/v1/unload_model", "/api/v1/retrieve_future"]


@pytest.mark.parametrize("failure", [None, "publication", "sample", "reference", "backward", "optimizer", "checkpoint"])
def test_client_refreshes_references_before_each_gspo_update_and_cleans_up(tmp_path, failure):
    from unittest.mock import Mock

    events = []
    trainer = Mock(model_id="model-test")
    trainer.get_info.return_value = types.GetInfoResponse.model_validate(
        {"model_id": "model-test", "model_data": {"model_name": "test-model"}}
    )
    trainer.get_tokenizer.return_value.encode.side_effect = [[11, 12, 13], [21, 22, 23]]

    def future(name, value):
        def result():
            events.append(name)
            if failure == name:
                raise RuntimeError("worker failed")
            return value

        return SimpleNamespace(result=result)

    def output(logprob, count):
        return SimpleNamespace(
            loss_fn_outputs=[{"logprobs": types.TensorData(data=[logprob] * 7, dtype="float32", shape=[7])}] * count,
            metrics={"loss": 0.0},
        )

    def forward(data, loss):
        assert loss == "cross_entropy"
        assert len(data) == 2
        assert "advantages" not in data[0].loss_fn_inputs
        return future("reference", output(-1.0 - events.count("optimizer"), len(data)))

    def backward(data, loss):
        assert loss == "gspo"
        assert events.count("reference") == events.count("optimizer") + 1
        assert len(data) == 2
        for datum, advantage in zip(data, [1.0, -1.0], strict=True):
            fields = datum.loss_fn_inputs
            assert fields["logprobs"].data == [-1.0 - events.count("optimizer")] * 7
            assert fields["advantages"].data == [advantage] * 7
            assert fields["weights"].data == [1.0] * 7
        return future("backward", output(-1.0, len(data)))

    trainer.forward.side_effect = forward
    trainer.forward_backward.side_effect = backward
    trainer.optim_step.side_effect = lambda params: future(
        "optimizer", SimpleNamespace(metrics={"skyrl.ai/grad_norm": 1.0})
    )
    sampler = Mock()
    sampler.sample.side_effect = lambda *a, **kw: future(
        "sample", SimpleNamespace(sequences=[SimpleNamespace(tokens=[7])])
    )

    def publish():
        events.append("publication")
        if failure == "publication":
            raise RuntimeError("worker failed")
        return sampler

    trainer.save_weights_and_get_sampling_client.side_effect = publish
    trainer.save_state.side_effect = lambda name: future("checkpoint", SimpleNamespace(path="tinker://test/state"))
    service = Mock()
    service.create_lora_training_client.return_value = trainer
    args = SimpleNamespace(
        output_dir=tmp_path / "result",
        base_url="http://example.com",
        model_path="test-model",
        inference_profile_url=None,
        inference_profile_url_file=None,
        context=7,
        batch_size=2,
        steps=2,
        learning_rate=1e-5,
    )
    with (
        patch.object(module.tinker, "ServiceClient", return_value=service),
        patch.object(module, "unload_model", side_effect=lambda url, model: events.append("unload")),
    ):
        expected = (
            ["publication", "sample"]
            + ["reference", "backward", "optimizer", "publication", "sample"] * 2
            + ["checkpoint", "unload"]
        )
        if failure:
            with pytest.raises(RuntimeError, match="worker failed"):
                example.run(args)
            assert events == expected[: expected.index(failure) + 1] + ["unload"]
        else:
            example.run(args)
            assert events == expected
            assert json.loads((args.output_dir / "run.json").read_text())["backwards_per_step"] == 1
            assert json.loads((args.output_dir / "run.json").read_text())["loss_fn"] == "gspo"
            saved = json.loads((args.output_dir / "step_1_batch.json").read_text())[1]
            assert saved["loss_fn_inputs"]["logprobs"]["data"] == [-2.0] * 7
            assert saved["loss_fn_inputs"]["advantages"]["data"] == [-1.0] * 7
            metadata = json.loads((args.output_dir / "run.json").read_text())
            assert metadata["warmup_steps"] == metadata["measured_steps"] == 1
            fixture = (args.output_dir / "datums.json").read_bytes()
            assert metadata["datums_sha256"] == hashlib.sha256(fixture).hexdigest()
            records = [json.loads(line) for line in (args.output_dir / "phases.jsonl").read_text().splitlines()]
            completed = {record["phase"]: record for record in records if record["status"] == "completed"}
            assert completed["prepare_inputs"]["input_positions"] == [7, 7]
            assert completed["prepare_inputs"]["scored_positions"] == [7, 7]
            assert "initial/publication" in completed
            assert "warmup" in completed and "step_1" in completed and "step_0" not in completed


def test_explicit_tokenizer_preserves_inputs_for_remote_local_model(tmp_path):
    info = types.GetInfoResponse.model_validate(
        {"model_id": "model-test", "model_data": {"model_name": "/remote/model"}}
    )
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [11, 12] if text.startswith("A river") else [21, 22])
    trainer = SimpleNamespace(get_info=lambda: info, get_tokenizer=lambda: tokenizer)
    args = SimpleNamespace(output_dir=tmp_path, context=7, batch_size=2, steps=3, learning_rate=1e-5)
    expected = module.prepare_full_context_inputs(trainer, args, io.StringIO())
    with patch.object(trainer, "get_tokenizer", side_effect=AssertionError("remote path unavailable locally")):
        actual = module.prepare_full_context_inputs(trainer, args, io.StringIO(), tokenizer)
    assert module.serialize_batch(actual) == module.serialize_batch(expected)
    assert (tmp_path / "datums.json").read_text() == module.serialize_batch(expected) + "\n"
