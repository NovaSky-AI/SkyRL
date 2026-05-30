"""Test that the ``sub_seq_lengths`` data field is correctly chunked across
micro-batches inside the worker's forward_backward path.

``sub_seq_lengths`` is a ``TrainingInputBatch`` *data* field, a
``TensorList`` (one 1-D int tensor per bin row). As a data field,
``MeshDispatch`` shards it per-DP rank, and the worker slices its
(already per-rank) shard per micro-batch the same way ``BatchIterator``
chunks the tensor rows (``data.chunk(micro_batch_size)``).

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

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch


def _slice_sub_seq_lengths_like_worker(data: TrainingInputBatch, micro_batch_size: int) -> List[Optional[TensorList]]:
    """Mirror the slicing logic from MegatronPolicyWorkerBase.forward_backward.

    The data field arrives already sharded to this DP rank, so the worker just
    slices it per micro-batch. Kept as a small standalone helper so the test
    exercises the same logic without paying the cost of spinning up a Ray actor.
    """
    sub_seq_lengths_field: Optional[TensorList] = data.get("sub_seq_lengths")
    chunks: List[Optional[TensorList]] = []
    if sub_seq_lengths_field is None:
        # Mimic the "absent" path: one None per chunk.
        for _ in range(0, data.batch_size, micro_batch_size):
            chunks.append(None)
        return chunks
    for i in range(0, data.batch_size, micro_batch_size):
        chunks.append(sub_seq_lengths_field[i : i + micro_batch_size])
    return chunks


def _to_lists(chunk: Optional[TensorList]) -> Optional[List[List[int]]]:
    """TensorList -> list[list[int]] (the .tolist() boundary done in forward_step)."""
    if chunk is None:
        return None
    return [t.tolist() for t in chunk]


def _make_tensor_list(rows: List[List[int]]) -> TensorList:
    return TensorList([torch.tensor(r, dtype=torch.long) for r in rows])


class TestSubSeqLengthsSlicing:
    def test_one_bin_per_micro_batch(self):
        # Build a 4-row batch where each row is one bin holding 2 sub-seqs.
        sub_seq_lengths = _make_tensor_list([[5, 5], [6, 4], [7, 3], [4, 6]])
        batch = TrainingInputBatch(
            {
                "sequences": torch.zeros((4, 16), dtype=torch.long),
                "attention_mask": torch.ones((4, 16), dtype=torch.long),
                "loss_mask": torch.zeros((4, 15), dtype=torch.float),
                "sub_seq_lengths": sub_seq_lengths,
            }
        )
        batch.metadata = {"response_length": 15}
        chunks = _slice_sub_seq_lengths_like_worker(batch, micro_batch_size=1)
        assert len(chunks) == 4
        expected_rows = [[5, 5], [6, 4], [7, 3], [4, 6]]
        for chunk_idx, expected in enumerate(expected_rows):
            assert _to_lists(chunks[chunk_idx]) == [expected]

    def test_multi_bin_per_micro_batch(self):
        # 6 bin rows with packed_mbs=2: 3 micro-batches of 2 bins each.
        sub_seq_lengths = _make_tensor_list(
            [
                [5],
                [5, 5],
                [6, 4],
                [7, 3],
                [4, 6],
                [8],
            ]
        )
        batch = TrainingInputBatch(
            {
                "sequences": torch.zeros((6, 32), dtype=torch.long),
                "attention_mask": torch.ones((6, 32), dtype=torch.long),
                "loss_mask": torch.zeros((6, 31), dtype=torch.float),
                "sub_seq_lengths": sub_seq_lengths,
            }
        )
        batch.metadata = {"response_length": 31}
        chunks = _slice_sub_seq_lengths_like_worker(batch, micro_batch_size=2)
        assert len(chunks) == 3
        assert _to_lists(chunks[0]) == [[5], [5, 5]]
        assert _to_lists(chunks[1]) == [[6, 4], [7, 3]]
        assert _to_lists(chunks[2]) == [[4, 6], [8]]

    def test_absent_field_passes_through_as_none(self):
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

    def test_length_mismatch_raises_at_construction(self):
        """A data field whose length != batch_size is rejected up front.

        Because ``sub_seq_lengths`` is now a data field, ``TrainingInputBatch``
        enforces the per-row alignment at construction time via
        ``_check_consistency`` -- no separate worker-side contract check needed.
        """
        # 3 sub-seq rows but 4 tensor rows -> batch-size mismatch.
        with pytest.raises(ValueError, match="Batch size mismatch"):
            TrainingInputBatch(
                {
                    "sequences": torch.zeros((4, 16), dtype=torch.long),
                    "attention_mask": torch.ones((4, 16), dtype=torch.long),
                    "loss_mask": torch.zeros((4, 15), dtype=torch.float),
                    "sub_seq_lengths": _make_tensor_list([[5], [5], [6]]),
                }
            )
