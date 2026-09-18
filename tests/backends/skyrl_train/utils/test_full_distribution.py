import pytest
import torch

from skyrl.backends.skyrl_train.utils.full_distribution import check_full_logprobs


def test_check_full_logprobs_accepts_exact_selected_rows():
    rollout = torch.tensor(
        [[[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]]],
        dtype=torch.float32,
    )
    trainer = rollout.clone()
    trainer[0, 0, 0] = 123.0
    mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)

    check_full_logprobs(trainer, rollout, mask)


def test_check_full_logprobs_rejects_one_bit_difference():
    rollout = torch.tensor([[[-1.0, -2.0]]], dtype=torch.float32)
    trainer = rollout.clone()
    trainer[0, 0, 1] = torch.nextafter(trainer[0, 0, 1], torch.tensor(0.0))

    with pytest.raises(ValueError, match="1 vocabulary entries differ"):
        check_full_logprobs(trainer, rollout, torch.ones((1, 1)))


@pytest.mark.parametrize(
    "trainer,rollout,mask,message",
    [
        (torch.zeros((1, 2, 3)), torch.zeros((1, 1, 3)), torch.ones((1, 2)), "shapes"),
        (torch.zeros((1, 1, 3), dtype=torch.float64), torch.zeros((1, 1, 3)), torch.ones((1, 1)), "float32"),
        (
            torch.tensor([[[float("nan")]]], dtype=torch.float32),
            torch.zeros((1, 1, 1)),
            torch.ones((1, 1)),
            "finite",
        ),
    ],
)
def test_check_full_logprobs_rejects_invalid_inputs(trainer, rollout, mask, message):
    with pytest.raises(ValueError, match=message):
        check_full_logprobs(trainer, rollout, mask)
