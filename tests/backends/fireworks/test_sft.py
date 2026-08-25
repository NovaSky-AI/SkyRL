from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.fireworks.sft import (
    FireworksSFTDispatch,
    build_tinker_sft_datums,
    training_batch_to_sft_datum_specs,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import FireworksConfig, OptimizerConfig

tinker = pytest.importorskip("tinker")


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class _TrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
        self.forward_calls = []
        self.optim_calls = []

    def forward_backward(self, datums, loss_fn):
        self.forward_backward_calls.append((datums, loss_fn))
        return _Future(SimpleNamespace(metrics={"loss:sum": 1.25}))

    def forward(self, datums, loss_fn):
        self.forward_calls.append((datums, loss_fn))
        return _Future(SimpleNamespace(metrics={"loss:sum": 1.5}))

    def optim_step(self, params, **kwargs):
        self.optim_calls.append((params, kwargs))
        return _Future(SimpleNamespace(metrics={"grad_norm": 0.75}))


class _ForwardOnlyClient:
    def __init__(self, result):
        self.result_value = result

    def forward(self, datums, loss_fn):
        self.datums = datums
        self.loss_fn = loss_fn
        return _Future(self.result_value)


def _batch() -> TrainingInputBatch:
    return TrainingInputBatch(
        {
            "sequences": torch.tensor(
                [
                    [10, 0, 11, 30, 31],
                    [0, 20, 40, 41, 42],
                    [10, 0, 11, 30, 31],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
            ),
            "loss_mask": torch.tensor(
                [
                    [0.0, 0.25, 0.0],
                    [0.0, 0.25, 0.5],
                    [0.0, 0.0, 0.0],
                ]
            ),
        }
    )


def _dispatch_with_forward_result(result):
    return FireworksSFTDispatch(
        SimpleNamespace(training_client=_ForwardOnlyClient(result)),
        FireworksConfig(max_seq_len=4, request_timeout_s=10),
        OptimizerConfig(lr=2e-5),
    )


def test_training_batch_to_sft_specs_preserves_collator_weights() -> None:
    specs = training_batch_to_sft_datum_specs(_batch(), max_seq_len=4)

    assert len(specs) == 2
    assert specs[0].model_input_token_ids == (10, 0, 11, 30)
    assert specs[0].target_tokens == (0, 11, 30, 31)
    assert specs[0].weights == pytest.approx((0.0, 0.0, 0.25, 0.0))
    assert specs[1].model_input_token_ids == (20, 40, 41)
    assert specs[1].target_tokens == (40, 41, 42)
    assert specs[1].weights == pytest.approx((0.0, 0.25, 0.5))
    assert sum(sum(spec.weights) for spec in specs) == pytest.approx(1.0)


def test_training_batch_to_sft_specs_rejects_non_contiguous_padding() -> None:
    batch = _batch()
    batch["attention_mask"][1] = torch.tensor([0, 1, 0, 1, 1])

    with pytest.raises(ValueError, match="contiguous left padding"):
        training_batch_to_sft_datum_specs(batch)


def test_training_batch_to_sft_specs_rejects_invalid_weights() -> None:
    batch = _batch()
    batch["loss_mask"][0, 1] = torch.nan

    with pytest.raises(ValueError, match="finite non-negative"):
        training_batch_to_sft_datum_specs(batch)


def test_training_batch_to_sft_specs_rejects_length_over_limit() -> None:
    with pytest.raises(ValueError, match="exceeding max_seq_len=3"):
        training_batch_to_sft_datum_specs(_batch(), max_seq_len=3)


def test_build_tinker_sft_datums() -> None:
    pytest.importorskip("tinker")

    datums = build_tinker_sft_datums(_batch(), max_seq_len=4)

    assert [datum.model_input.length for datum in datums] == [4, 3]
    assert datums[0].loss_fn_inputs["target_tokens"].dtype == "int64"
    assert datums[0].loss_fn_inputs["weights"].dtype == "float32"


def test_sft_dispatch_uses_cross_entropy_for_train_and_eval() -> None:
    client = _TrainingClient()
    dispatch = FireworksSFTDispatch(
        SimpleNamespace(training_client=client),
        FireworksConfig(max_seq_len=4, request_timeout_s=10),
        OptimizerConfig(lr=2e-5),
    )

    train_output = dispatch.forward_backward("policy", _batch(), loss_fn="cross_entropy")
    eval_output = dispatch.forward("policy", _batch(), loss_fn="cross_entropy")
    grad_norm = dispatch.optim_step("policy")

    assert client.forward_backward_calls[0][1] == "cross_entropy"
    assert client.forward_calls[0][1] == "cross_entropy"
    assert train_output.metrics["final_loss"] == pytest.approx(1.25)
    assert eval_output.metrics["loss"] == pytest.approx(1.5)
    assert client.optim_calls[0][1] == {"grad_accumulation_normalization": None}
    assert grad_norm == pytest.approx(0.75)


def test_sft_dispatch_forward_computes_weighted_loss() -> None:
    result = SimpleNamespace(
        metrics={},
        loss_fn_outputs=[
            {"logprobs": tinker.TensorData(data=[-1.0, -2.0, -3.0, -4.0], dtype="float32")},
            {"logprobs": {"data": [-1.0, -2.0, -3.0]}},
        ],
    )
    dispatch = _dispatch_with_forward_result(result)

    output = dispatch.forward("policy", _batch(), loss_fn="cross_entropy")

    assert output.metrics["loss"] == pytest.approx(2.75)
    assert output.loss_fn_outputs == []


@pytest.mark.parametrize(
    "loss_fn_outputs",
    [
        None,
        [],
        [{"logprobs": tinker.TensorData(data=[-1.0, -2.0, -3.0, -4.0], dtype="float32")}],
        [{}, {"logprobs": tinker.TensorData(data=[-1.0, -2.0, -3.0], dtype="float32")}],
        [
            {"logprobs": tinker.TensorData(data=[-1.0, -2.0, -3.0, -4.0], dtype="float32")},
            {"logprobs": tinker.TensorData(data=[-1.0, -2.0], dtype="float32")},
        ],
    ],
)
def test_sft_dispatch_forward_omits_loss_for_invalid_outputs(loss_fn_outputs) -> None:
    result = SimpleNamespace(metrics={}, loss_fn_outputs=loss_fn_outputs)
    dispatch = _dispatch_with_forward_result(result)

    output = dispatch.forward("policy", _batch(), loss_fn="cross_entropy")

    assert "loss" not in output.metrics
