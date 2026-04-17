"""
Tests for prompt-based mini-batching.

uv run --isolated --extra dev pytest tests/train/test_prompt_mini_batch.py -v
"""

from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import MeshDispatch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.dataset.preprocess import compute_prompt_mini_batch_boundaries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uids_stepwise(
    prompts: List[Tuple[str, int, List[int]]],
) -> List[str]:
    """Build uid list for a step-wise batch.

    Args:
        prompts: List of (instance_id, spp, turns_per_sample) tuples.
            ``turns_per_sample`` is a list of length ``spp`` giving
            the number of turns for each trajectory of that prompt.

    Returns:
        Flat uid list — same uid for all sequences of the same prompt.
    """
    uids: List[str] = []
    for instance_id, _, turns_list in prompts:
        for num_turns in turns_list:
            for _ in range(num_turns):
                uids.append(instance_id)
    return uids


def _make_uids_fixed(train_batch_size: int, spp: int) -> List[str]:
    """Build uid list for a non-step-wise batch (fixed spp per prompt).

    spp: samples per prompt.

    Example:
    train_batch_size = 4, spp = 2
    uids = ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"]
    """
    return [f"p{i}" for i in range(train_batch_size) for _ in range(spp)]


def _make_batch(num_sequences: int, seq_len: int = 8) -> TrainingInputBatch:
    """Create a minimal TrainingInputBatch with the given number of sequences."""
    batch = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (num_sequences, seq_len)),
            "attention_mask": torch.ones(num_sequences, seq_len, dtype=torch.long),
            "response_mask": torch.ones(num_sequences, seq_len, dtype=torch.long),
            "advantages": torch.randn(num_sequences, seq_len),
            "loss_mask": torch.ones(num_sequences, seq_len, dtype=torch.float),
        }
    )
    batch.metadata = {
        "is_last_step": [False] * num_sequences,
    }
    return batch


# ---------------------------------------------------------------------------
# Tests for compute_prompt_mini_batch_boundaries
# ---------------------------------------------------------------------------


class TestComputePromptMiniBatchBoundaries:
    def test_nonstepwise_training(self):
        """Test non-stepwise training with different mini batch sizes.

        For non-stepwise training, len(uids) % mini_batch_size should be 0.
        """
        train_batch_size = 4
        spp = 2
        is_stepwise = False
        uids = ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"]

        for mini_batch_size, expected_boundaries in [
            (1, [(0, 2), (2, 4), (4, 6), (6, 8)]),
            (2, [(0, 4), (4, 8)]),
            (4, [(0, 8)]),
        ]:
            boundaries = compute_prompt_mini_batch_boundaries(uids, mini_batch_size, train_batch_size, is_stepwise, spp)
            assert boundaries == expected_boundaries

    def test_noncontiguous_uids_raise(self):
        """Non-contiguous uids should raise an assertion error."""
        train_batch_size = 4
        spp = 2
        is_stepwise = False
        uids = ["p0", "p0", "p1", "p0", "p2", "p2", "p3"]
        with pytest.raises(AssertionError, match="uid 'p0' appears in non-contiguous positions at index 3."):
            compute_prompt_mini_batch_boundaries(uids, 2, train_batch_size, is_stepwise, spp)

    def test_train_batch_size_not_equal_unique_uids_raise(self):
        """When the number of prompts is not equal to the train batch size, raise an assertion error."""
        is_stepwise = False
        train_batch_size = 4
        spp = 2
        mini_batch_size = 2
        uids = ["p0", "p0", "p1", "p1", "p2", "p2"]
        with pytest.raises(AssertionError):
            compute_prompt_mini_batch_boundaries(uids, mini_batch_size, train_batch_size, is_stepwise, spp)

    def test_stepwise_training(self):
        """Step-wise: prompts have variable numbers of turns."""
        # Test 1: Each trajectory can have 1-4 turns, train_batch_size = 4, spp = 2.
        mini_batch_size = 2
        train_batch_size = 4
        spp = 2
        is_stepwise = True
        uids = _make_uids_stepwise(
            [
                ("p0", 2, [3, 2]),  # 5 seqs with a 3-turn trajectory and a 2-turn trajectory
                ("p1", 2, [1, 4]),  # 5 seqs
                ("p2", 2, [2, 1]),  # 3 seqs
                ("p3", 2, [1, 1]),  # 2 seqs
            ]
        )
        assert uids == ["p0", "p0", "p0", "p0", "p0", "p1", "p1", "p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3"]
        assert len(uids) == 15
        assert [(0, 10), (10, 15)] == compute_prompt_mini_batch_boundaries(
            uids, mini_batch_size, train_batch_size, is_stepwise, spp
        )

        # Test 2: Each mini batch only has 1 prompt.
        mini_batch_size = 1
        train_batch_size = 2
        spp = 3
        is_stepwise = True
        uids = _make_uids_stepwise(
            [
                ("p0", 3, [2, 1, 3]),  # 6 seqs
                ("p1", 3, [1, 1, 1]),  # 3 seqs
            ]
        )
        assert [(0, 6), (6, 9)] == compute_prompt_mini_batch_boundaries(
            uids, mini_batch_size, train_batch_size, is_stepwise, spp
        )

    @pytest.mark.parametrize(
        "train_batch_size, spp, mini_batch_size",
        [
            (8, 4, 4),
            (256, 5, 128),
            (16, 1, 4),
            (32, 8, 32),
            (128, 5, 64),
        ],
    )
    def test_non_stepwise_boundaries_are_uniform(self, train_batch_size, spp, mini_batch_size):
        """
        For non-step-wise, every boundary must be [i*mb_size, (i+1)*mb_size). We run various
        parametrization to make sure the assertion in `compute_prompt_mini_batch_boundaries()` passes.
        """
        is_stepwise = False
        uids = _make_uids_fixed(train_batch_size, spp)
        compute_prompt_mini_batch_boundaries(uids, mini_batch_size, train_batch_size, is_stepwise, spp)

    def test_same_step_count_as_non_stepwise(self):
        """Step-wise and non-step-wise produce the same number of mini-batches."""
        train_batch_size = 256
        policy_mini_batch_size = 128
        n_samples = 5

        # Non-step-wise
        non_stepwise_uids = _make_uids_fixed(train_batch_size, n_samples)
        non_stepwise_bounds = compute_prompt_mini_batch_boundaries(
            non_stepwise_uids,
            mini_batch_size=policy_mini_batch_size,
            train_batch_size=train_batch_size,
            is_stepwise=False,
            n_samples_per_prompt=n_samples,
        )

        # Step-wise with variable turns
        prompts = []
        for i in range(train_batch_size):
            turns = [1 + (i * j) % 4 for j in range(n_samples)]
            prompts.append((f"p{i}", n_samples, turns))
        stepwise_uids = _make_uids_stepwise(prompts)
        stepwise_bounds = compute_prompt_mini_batch_boundaries(
            stepwise_uids,
            mini_batch_size=policy_mini_batch_size,
            train_batch_size=train_batch_size,
            is_stepwise=True,
            n_samples_per_prompt=n_samples,
        )

        assert len(stepwise_bounds) == len(non_stepwise_bounds) == 2

        # Non-step-wise boundaries should be uniform
        assert non_stepwise_bounds == [(0, 640), (640, 1280)]


# ---------------------------------------------------------------------------
# Tests for MeshDispatch.stage_chunks
# ---------------------------------------------------------------------------


class TestStageChunksVariable:
    def test_uniform_minibatches_dp1(self):
        """All mini-batches same size, dp_size=1 => no padding needed."""
        batch = _make_batch(10)
        boundaries = [(0, 5), (5, 10)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            all_chunk_refs = MeshDispatch.stage_chunks(dp_size=1, data=batch, mini_batch_boundaries=boundaries)

        # 2 mini batches, each with 1 chunk for the single DP rank
        assert len(all_chunk_refs) == 2
        assert len(all_chunk_refs[0]) == 1
        assert len(all_chunk_refs[1]) == 1

        # No padding — chunks should exactly match original batch slices.
        assert torch.equal(chunks_put[0]["sequences"], batch["sequences"][:5])
        assert torch.equal(chunks_put[1]["sequences"], batch["sequences"][5:10])

    def test_variable_minibatches_dp2_padding(self):
        """Variable sizes with dp_size=2 => odd-sized mini-batches get padded."""
        batch = _make_batch(7)
        boundaries = [(0, 3), (3, 7)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            # chunks_put is the physical things being put. `all_chunk_refs` is the dummy references, here is just
            # a list of indices for each chunk.
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            all_chunk_refs = MeshDispatch.stage_chunks(dp_size=2, data=batch, mini_batch_boundaries=boundaries)

        # 2 mini batches, one size 3, one size 4. With dp_size=2, first mini batch pads to 4, gets 2 chunks. Second mini batch gets 2 chunks.
        assert len(all_chunk_refs) == 2
        assert len(all_chunk_refs[0]) == 2  # 3->4, split into 2
        assert len(all_chunk_refs[1]) == 2  # 4, split into 2
        assert len(chunks_put) == 4  # put got called 4 times.
        assert all(len(chunk) == 2 for chunk in chunks_put)  # each chunk is size 2

        # Reconstruct each mini-batch from its chunks and verify against original batch.
        # Mini-batch 0 chunk 1: batch[0:2]. Loss mask should be the same.
        assert torch.equal(chunks_put[0]["sequences"], batch["sequences"][:2])
        assert torch.equal(chunks_put[0]["loss_mask"], batch["loss_mask"][:2])

        # Mini-batch 0 chunk 2: batch[2:3] padded to 2 (row 0 cloned as padding). Loss mask second row should be zero.
        expected_mb0_chunk2 = torch.cat([batch["sequences"][2:3], batch["sequences"][0:1]], dim=0)
        assert torch.equal(chunks_put[1]["sequences"], expected_mb0_chunk2)
        assert torch.equal(chunks_put[1]["loss_mask"][0], batch["loss_mask"][2])
        assert torch.all(chunks_put[1]["loss_mask"][1] == 0)

        # Mini-batch 1 chunk 1 and 2: batch[3:7], no padding needed (already divisible by 2). Should be identical to original batch.
        assert torch.equal(chunks_put[2]["sequences"], batch["sequences"][3:5])
        assert torch.equal(chunks_put[3]["sequences"], batch["sequences"][5:7])
        assert torch.equal(chunks_put[2]["loss_mask"], batch["loss_mask"][3:5])
        assert torch.equal(chunks_put[3]["loss_mask"], batch["loss_mask"][5:7])

    def test_dp_size_4_heavy_padding(self):
        """dp_size=4, mini-batch of 5 => padded to 8."""
        batch = _make_batch(5)
        boundaries = [(0, 5)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            all_chunk_refs = MeshDispatch.stage_chunks(dp_size=4, data=batch, mini_batch_boundaries=boundaries)

        assert len(all_chunk_refs) == 1
        assert len(all_chunk_refs[0]) == 4
        for chunk in chunks_put:
            assert len(chunk) == 2

        # Reconstruct: batch[0:5] padded to 8 (3 padding rows, all clones of row 0).
        mb = torch.cat([c["sequences"] for c in chunks_put], dim=0)
        assert torch.equal(mb[:5], batch["sequences"])  # original rows preserved
        for i in range(5, 8):
            assert torch.equal(mb[i], batch["sequences"][0])  # padding is row 0

        # Loss mask: padding rows should be zero.
        mb_loss = torch.cat([c["loss_mask"] for c in chunks_put], dim=0)
        assert torch.equal(mb_loss[:5], batch["loss_mask"][:5])
        assert torch.all(mb_loss[5:] == 0)
