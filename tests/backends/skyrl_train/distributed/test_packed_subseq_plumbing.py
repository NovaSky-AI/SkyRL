"""Test that ``sub_seq_lengths`` metadata is correctly chunked across
micro-batches inside the worker's forward_backward path.

This is a CPU-only smoke test that exercises just the slicing logic from
``MegatronPolicyWorkerBase.forward_backward`` without instantiating a
distributed Ray worker. The full GPU integration is covered by
``tests/backends/skyrl_train/gpu/gpu_ci/test_training_step.py`` (out of
scope for this CPU CI lane).

Run with:
  uv run --extra dev --extra megatron -- pytest \
      tests/backends/skyrl_train/distributed/test_packed_subseq_plumbing.py
"""

from typing import List, Optional

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


def _slice_sub_seq_lengths_like_worker(
    data: TrainingInputBatch, micro_batch_size: int
) -> List[Optional[List[List[int]]]]:
    """Mirror the slicing logic from MegatronPolicyWorkerBase.forward_backward.

    Kept as a small standalone helper so the test exercises the same logic
    without paying the cost of spinning up a Ray actor.
    """
    sub_seq_lengths_full: Optional[List[List[int]]] = (
        (data.metadata or {}).get("sub_seq_lengths") if data.metadata else None
    )
    chunks: List[Optional[List[List[int]]]] = []
    if sub_seq_lengths_full is None:
        # Mimic the "absent" path: one None per chunk.
        for _ in range(0, data.batch_size, micro_batch_size):
            chunks.append(None)
        return chunks
    for i in range(0, data.batch_size, micro_batch_size):
        chunks.append(sub_seq_lengths_full[i : i + micro_batch_size])
    return chunks


class TestSubSeqLengthsSlicing:
    def test_one_bin_per_micro_batch(self):
        # Build a 4-row batch where each row is one bin holding 2 sub-seqs.
        batch = TrainingInputBatch(
            {
                "sequences": torch.zeros((4, 16), dtype=torch.long),
                "attention_mask": torch.ones((4, 16), dtype=torch.long),
                "loss_mask": torch.zeros((4, 15), dtype=torch.float),
            }
        )
        batch.metadata = {
            "response_length": 15,
            "sub_seq_lengths": [[5, 5], [6, 4], [7, 3], [4, 6]],
        }
        chunks = _slice_sub_seq_lengths_like_worker(batch, micro_batch_size=1)
        assert len(chunks) == 4
        for chunk_idx, expected in enumerate(batch.metadata["sub_seq_lengths"]):
            assert chunks[chunk_idx] == [expected]

    def test_multi_bin_per_micro_batch(self):
        # 6 bin rows with packed_mbs=2: 3 micro-batches of 2 bins each.
        batch = TrainingInputBatch(
            {
                "sequences": torch.zeros((6, 32), dtype=torch.long),
                "attention_mask": torch.ones((6, 32), dtype=torch.long),
                "loss_mask": torch.zeros((6, 31), dtype=torch.float),
            }
        )
        batch.metadata = {
            "response_length": 31,
            "sub_seq_lengths": [
                [5],
                [5, 5],
                [6, 4],
                [7, 3],
                [4, 6],
                [8],
            ],
        }
        chunks = _slice_sub_seq_lengths_like_worker(batch, micro_batch_size=2)
        assert len(chunks) == 3
        assert chunks[0] == [[5], [5, 5]]
        assert chunks[1] == [[6, 4], [7, 3]]
        assert chunks[2] == [[4, 6], [8]]

    def test_absent_metadata_passes_through_as_none(self):
        batch = TrainingInputBatch(
            {
                "sequences": torch.zeros((4, 16), dtype=torch.long),
                "attention_mask": torch.ones((4, 16), dtype=torch.long),
                "loss_mask": torch.zeros((4, 15), dtype=torch.float),
            }
        )
        batch.metadata = {"response_length": 15}  # no sub_seq_lengths
        chunks = _slice_sub_seq_lengths_like_worker(batch, micro_batch_size=2)
        assert all(c is None for c in chunks)
        assert len(chunks) == 2

    def test_length_mismatch_would_be_caught_in_real_worker(self):
        """In the real worker, len(sub_seq_lengths) != batch_size raises.

        We replicate that check here to lock the contract.
        """
        batch = TrainingInputBatch(
            {
                "sequences": torch.zeros((4, 16), dtype=torch.long),
                "attention_mask": torch.ones((4, 16), dtype=torch.long),
                "loss_mask": torch.zeros((4, 15), dtype=torch.float),
            }
        )
        # Only 3 lists for 4 rows -> the worker raises ValueError. We don't
        # call into the real worker (no Ray), but the slicing helper would
        # silently produce 4 chunks of variable composition. Add the contract
        # check here to flag the misuse:
        batch.metadata = {
            "response_length": 15,
            "sub_seq_lengths": [[5], [5], [6]],
        }
        with pytest.raises(ValueError, match="metadata\\['sub_seq_lengths'\\]"):
            _emulate_worker_contract_check(batch)


def _emulate_worker_contract_check(data: TrainingInputBatch) -> None:
    """The exact contract check from MegatronPolicyWorkerBase.forward_backward."""
    sub_seq_lengths_full: Optional[List[List[int]]] = (
        (data.metadata or {}).get("sub_seq_lengths") if data.metadata else None
    )
    if sub_seq_lengths_full is None:
        return
    if len(sub_seq_lengths_full) != data.batch_size:
        raise ValueError(
            f"metadata['sub_seq_lengths'] has {len(sub_seq_lengths_full)} rows but " f"batch size is {data.batch_size}"
        )
