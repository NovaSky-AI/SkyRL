import math

import pytest
import torch

from skyrl.backends.fireworks.grpo import (
    build_tinker_grpo_datums,
    training_batch_to_grpo_datum_specs,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


def _batch() -> TrainingInputBatch:
    # Row 0 contains token id 0 as real prompt text. Row 1 uses the same value
    # as left padding, proving that attention_mask (not token value) drives
    # unpadding. Response fields have width 3 and are right aligned.
    return TrainingInputBatch(
        {
            "sequences": torch.tensor(
                [
                    [10, 0, 11, 30, 31],
                    [0, 20, 40, 41, 42],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.long,
            ),
            "loss_mask": torch.tensor(
                [
                    [0, 1, 0],
                    [0, 1, 1],
                ],
                dtype=torch.float32,
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 2.0, float("nan")],
                    [4.0, 5.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            "rollout_logprobs": torch.tensor(
                [
                    [0.0, -0.1, float("nan")],
                    [-0.2, -0.3, -0.4],
                ],
                dtype=torch.float32,
            ),
        }
    )


def test_training_batch_to_grpo_datum_specs_shifts_and_masks() -> None:
    specs = training_batch_to_grpo_datum_specs(_batch(), max_seq_len=4)

    assert len(specs) == 2
    assert specs[0].model_input_token_ids == (10, 0, 11, 30)
    assert specs[0].target_tokens == (0, 0, 30, 31)
    assert specs[0].rollout_logprobs == pytest.approx((0.0, 0.0, -0.1, 0.0))
    assert specs[0].advantages == pytest.approx((0.0, 0.0, 2.0, 0.0))

    assert specs[1].model_input_token_ids == (20, 40, 41)
    assert specs[1].target_tokens == (40, 41, 42)
    assert specs[1].rollout_logprobs == pytest.approx((0.0, -0.3, -0.4))
    assert specs[1].advantages == pytest.approx((0.0, 5.0, 6.0))


@pytest.mark.parametrize("field", ["advantages", "rollout_logprobs"])
def test_training_batch_to_grpo_datum_specs_rejects_non_finite_trainable_values(
    field: str,
) -> None:
    batch = _batch()
    batch[field][0, -2] = math.nan

    with pytest.raises(
        ValueError, match="non-finite value at trainable response index"
    ):
        training_batch_to_grpo_datum_specs(batch)


def test_training_batch_to_grpo_datum_specs_rejects_non_contiguous_padding() -> None:
    batch = _batch()
    batch["attention_mask"][1] = torch.tensor([0, 1, 0, 1, 1])

    with pytest.raises(ValueError, match="contiguous left padding"):
        training_batch_to_grpo_datum_specs(batch)


def test_training_batch_to_grpo_datum_specs_rejects_length_over_limit() -> None:
    with pytest.raises(ValueError, match="exceeding max_seq_len=3"):
        training_batch_to_grpo_datum_specs(_batch(), max_seq_len=3)


def test_build_tinker_grpo_datums() -> None:
    pytest.importorskip("tinker")

    datums = build_tinker_grpo_datums(_batch(), max_seq_len=4)

    assert [datum.model_input.length for datum in datums] == [4, 3]
    target_tokens = datums[0].loss_fn_inputs["target_tokens"]
    target_data = (
        target_tokens.data if hasattr(target_tokens, "data") else target_tokens
    )
    assert list(target_data) == [0, 0, 30, 31]
    assert datums[0].loss_fn_inputs["target_tokens"].dtype == "int64"
    assert datums[0].loss_fn_inputs["logprobs"].dtype == "float32"
    assert datums[0].loss_fn_inputs["advantages"].dtype == "float32"
