import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.training_batch import (
    REAL_SAMPLE_MASK,
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean


def make_batch():
    return TrainingInputBatch(
        {
            "sequences": torch.zeros(5, 6, dtype=torch.long),
            # Row 3 is a real filtered trajectory; row 4 is synthetic.
            "loss_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            REAL_SAMPLE_MASK: torch.tensor([True, True, True, True, False]),
        }
    )


@pytest.mark.parametrize("partitions", [[[0, 1, 2, 3, 4]], [[0], [1], [2], [3], [4]], [[2, 1], [4, 0, 3]]])
def test_loss_and_gradients_are_independent_of_partition(partitions):
    batch = make_batch()
    normalization = MinibatchLossNormalization.from_batch(batch, dp_group=None, device="cpu")
    assert normalization == MinibatchLossNormalization(4, 7)
    parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    features = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 10
    kl = (parameter * features).square()
    entropy = (parameter * features).exp()
    mask = batch["loss_mask"]
    expected = 0.2 * (kl * mask).sum(dim=-1).div(mask.sum(dim=-1).clamp(min=1)).sum() / 4
    expected -= 0.1 * (entropy * mask).sum() / 7
    actual = parameter * 0
    for indices in partitions:
        actual += 0.2 * normalization.sequence_mean_contribution(
            masked_mean(kl[indices], mask[indices], dim=-1), batch.real_sample_mask[indices]
        )
        actual -= 0.1 * normalization.token_mean_contribution(
            masked_mean(entropy[indices], mask[indices]), mask[indices]
        )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, parameter, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, parameter)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_alignment_padding_does_not_change_sample_identity():
    batch = make_batch()[:4]
    padded = pad_training_input_batch(batch, 3)
    assert padded.real_sample_mask.tolist() == [True, True, True, True, False, False, False]
    assert MinibatchLossNormalization.from_batch(padded, dp_group=None, device="cpu") == MinibatchLossNormalization(
        4, 7
    )
    assert padded[2:6].real_sample_mask.tolist() == [True, True, False, False]


def test_padding_batch_without_explicit_mask():
    batch = make_batch()[:4]
    del batch[REAL_SAMPLE_MASK]
    assert pad_training_input_batch(batch, 2).real_sample_mask.tolist() == [True] * 4 + [False] * 2


def test_empty_losses_are_finite_and_differentiable():
    batch = make_batch()
    batch["loss_mask"].zero_()
    batch[REAL_SAMPLE_MASK].zero_()
    normalization = MinibatchLossNormalization.from_batch(batch, dp_group=None, device="cpu")
    values = torch.ones(5, 4, requires_grad=True)
    loss = normalization.token_mean_contribution(masked_mean(values, batch["loss_mask"]), batch["loss_mask"])
    loss += normalization.sequence_mean_contribution(values.mean(-1), batch.real_sample_mask)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(values.grad) == 0


def _distributed_counts(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        data = make_batch()[:3] if rank == 0 else make_batch()[3:]
        normalization = MinibatchLossNormalization.from_batch(data, dp_group=dist.group.WORLD, device="cpu")
        assert normalization == MinibatchLossNormalization(4, 7)
    finally:
        dist.destroy_process_group()


def test_distributed_counts_include_rank_with_no_loss_tokens(tmp_path):
    mp.spawn(_distributed_counts, args=(f"file://{tmp_path / 'gloo'}",), nprocs=2, join=True)
