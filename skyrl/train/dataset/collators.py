"""Collators that turn tokenized SFT examples into a :class:`TrainingInputBatch`.

Two callables cover the two SFT data paths:

- :class:`DefaultCollator` left-pads sequences to the batch maximum and applies
  the per-non-pad-token loss normalization.
- :class:`PackedDataCollator` performs controller-level FFD bin-packing
  (Megatron-only): once per training step it packs sequences into bins of
  capacity ``max_tokens_per_microbatch``, rounds the bin count up to a multiple
  of ``dp_size`` (so every DP rank gets the same number of micro-batches), and
  emits one row per bin. On the eval path (when the batch size differs from the
  configured training ``batch_size``) it falls back to the un-packed
  :class:`DefaultCollator` behavior.

Both reuse the shared :func:`skyrl.train.sft_trainer.collate_sft_batch` free
function for the un-packed layout.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from loguru import logger

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
)
from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch

from .bin_packing import make_seq_packer


class DefaultCollator:
    """Left-pad examples into a batch and apply loss normalization.

    Normalizes the ``loss_mask`` so that the sum-reduction in
    ``cross_entropy_loss`` produces a per-non-pad-token mean: the scale is
    ``batch_size / (micro_train_batch_size_per_gpu * total_nonpad)`` where
    ``total_nonpad`` is the count of loss-contributing tokens in the batch.
    This accounts for the ``microbatch_weight`` (FSDP) or ``1/num_microbatches``
    (Megatron) applied during gradient accumulation so the effective gradient
    equals ``d[sum(-log_probs_on_nonpad) / total_nonpad]``.
    """

    def __init__(self, tokenizer, micro_train_batch_size_per_gpu: int):
        self.tokenizer = tokenizer
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu

    def __call__(self, examples: list, batch_size: int) -> TrainingInputBatch:
        """Collate ``examples`` and scale the loss mask.

        Args:
            examples: Tokenized examples to collate.
            batch_size: Global batch dimension used in the loss-mask scaling
                factor. The train path passes ``sft_cfg.batch_size`` and the
                eval path passes its per-dispatch chunk size.
        """
        # Imported lazily to avoid a circular import: ``sft_trainer`` imports
        # this module to select a collator at construction time.
        from skyrl.train.sft_trainer import collate_sft_batch

        batch = collate_sft_batch(examples, self.tokenizer)
        micro_batch_size = self.micro_train_batch_size_per_gpu
        total_nonpad = max(batch["loss_mask"].sum().item(), 1)
        batch["loss_mask"] = batch["loss_mask"].float() * (batch_size / (micro_batch_size * total_nonpad))
        return batch


class PackedDataCollator:
    """Pack examples into bin rows via FFD and return a :class:`TrainingInputBatch`.

    Activates on the training-step batch (``batch_size == self.batch_size``).
    Flow:

    1. Compute per-example sequence lengths.
    2. FFD-pack with ``bin_capacity = max_tokens_per_microbatch``,
       ``min_bin_count = dp_size``, ``bin_count_multiple = dp_size``.
    3. Round-robin assign bins to DP shards (this happens implicitly inside
       ``MeshDispatch.dispatch`` because the rows are laid out in shard-major
       order: shard 0 rows first, then shard 1, etc).
    4. Build the per-bin packed row tensors and the per-row ``sub_seq_lengths``
       data field (a :class:`TensorList`).

    On the eval path (``batch_size != self.batch_size``) it delegates to a
    :class:`DefaultCollator` so eval always uses the un-packed layout; packing
    only fires on the training-step batch.
    """

    def __init__(
        self,
        tokenizer,
        max_tokens_per_microbatch: int,
        tp_size: int,
        pp_size: int,
        cp_size: int,
        dp_size: int,
        batch_size: int,
        micro_train_batch_size_per_gpu: int,
        fp8_enabled: bool = False,
    ):
        if max_tokens_per_microbatch is None:
            raise ValueError("PackedDataCollator requires max_tokens_per_microbatch to be set explicitly.")
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.batch_size = batch_size
        self.fp8_enabled = fp8_enabled
        self._default_collator = DefaultCollator(tokenizer, micro_train_batch_size_per_gpu)
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        # The eval fall-through reuses the inner DefaultCollator, so keep both
        # tokenizers in sync.
        self._tokenizer = value
        self._default_collator.tokenizer = value

    def __call__(self, examples: list, batch_size: int) -> TrainingInputBatch:
        # When eval calls the collator with a chunk of the eval set, fall back
        # to the un-packed collate path. Packing only fires on the
        # training-step batch (== self.batch_size).
        if batch_size != self.batch_size:
            return self._default_collator(examples, batch_size=batch_size)

        bin_capacity = self.max_tokens_per_microbatch

        tp_size = self.tp_size
        pp_size = self.pp_size
        cp_size = self.cp_size
        # Each sub-seq's padded length must satisfy these divisibility
        # constraints, which is why ``align_size`` carries all factors:
        #   - Sequence Parallelism (auto-on when tp>1) shards along the seq
        #     dim, so each segment must be divisible by ``tp_size``.
        #   - Context Parallelism splits each segment into ``2*cp_size`` equal
        #     load-balanced causal chunks, so each segment must be divisible by
        #     ``2*cp_size``.
        #   - When FP8 is enabled, Transformer Engine GEMMs require each CP
        #     rank's local token slab to be 16-aligned; globally this means
        #     ``16*cp_size``.
        # This MUST stay in lockstep with the worker's preprocess_packed_seqs
        # (megatron_utils.py): if the divisors drift, the per-rank CP/SP
        # gather/scatter offsets silently corrupt loss/grads (no crash).
        align_size = get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=self.fp8_enabled)

        dp_size = self.dp_size

        # ------------------------------------------------------------------
        # 1. Sequence lengths and full-sequence loss masks
        # ------------------------------------------------------------------
        # Build one mask per token so packed rows can shift loss by one position.
        seq_lengths: List[int] = []
        full_input_ids: List[np.ndarray] = []
        full_loss_masks: List[np.ndarray] = []
        for ex in examples:
            s = len(ex["input_ids"])
            seq_lengths.append(s)
            n_pad = s - ex["num_actions"]
            # Prompt prefix is zero; response mask is copied as float32.
            full_mask = np.empty(s, dtype=np.float32)
            full_mask[:n_pad] = 0.0
            full_mask[n_pad:] = np.asarray(ex["loss_mask"], dtype=np.float32)
            assert (
                full_mask.shape[0] == s
            ), f"Reconstructed full loss_mask length {full_mask.shape[0]} != seq length {s}"
            full_loss_masks.append(full_mask)
            full_input_ids.append(np.asarray(ex["input_ids"], dtype=np.int64))

        # ------------------------------------------------------------------
        # 2. FFD pack with DP-symmetry constraints
        # ------------------------------------------------------------------
        # Each bin row is one worker micro-batch. Megatron's
        # ``forward_backward_func`` runs one micro-batch per bin on each DP
        # rank, and its pipeline schedule requires every DP rank to issue the
        # same number of micro-batches. Forcing the global bin count to a
        # multiple of ``dp_size`` makes the per-DP-rank bin count (and thus
        # ``num_microbatches``) identical across ranks.
        bin_count_multiple = dp_size
        packer = make_seq_packer(
            "first_fit_decreasing",
            bin_capacity=bin_capacity,
            min_bin_count=bin_count_multiple,
            bin_count_multiple=bin_count_multiple,
        )
        bins: List[List[int]] = packer.pack(seq_lengths)

        # Assign bins to DP shards via round-robin (bin_idx % shards).
        # Concretely we want the resulting layout to be shard-major:
        # shard 0's bins occupy rows [0, K/dp), shard 1's bins occupy
        # [K/dp, 2K/dp), etc. MeshDispatch.dispatch chunks the batch
        # by dp_size and sends contiguous slabs, so we lay out the rows
        # already in shard-major order.
        shard_bins: List[List[List[int]]] = [[] for _ in range(dp_size)]
        for bin_idx, bin_indices in enumerate(bins):
            shard_idx = bin_idx % dp_size
            shard_bins[shard_idx].append(bin_indices)
        flat_bins: List[List[int]] = []
        for shard_idx in range(dp_size):
            flat_bins.extend(shard_bins[shard_idx])

        # ------------------------------------------------------------------
        # 3. Compute packed-row lengths (with align_size padding per sub-seq)
        #    and the global max packed length (for PP > 1 uniform padding).
        # ------------------------------------------------------------------
        def _round_up(x: int, m: int) -> int:
            return ((x + m - 1) // m) * m

        bin_packed_lengths: List[int] = []
        bin_subseq_lengths: List[List[int]] = []  # one list per bin row
        for bin_indices in flat_bins:
            subseq_lens = [seq_lengths[idx] for idx in bin_indices]
            # Each sub-seq's length is independently aligned to align_size
            # (matches preprocess_packed_seqs behavior).
            packed_len = sum(_round_up(s, align_size) for s in subseq_lens)
            bin_packed_lengths.append(packed_len)
            bin_subseq_lengths.append(subseq_lens)

        if pp_size > 1:
            # Pad all packed rows to the global max so Megatron's
            # pipeline schedule sees uniform shapes.
            max_packed_len = max(bin_packed_lengths) if bin_packed_lengths else 0
            # Also align the global max to align_size to keep layouts uniform.
            max_packed_len = _round_up(max_packed_len, align_size)
        else:
            max_packed_len = max(bin_packed_lengths) if bin_packed_lengths else 0

        # Guard against degenerate rows (e.g. an empty bin from
        # _adjust_bin_count) — empty bins must not be produced in practice
        # because the redistribution moves one sub-seq into every empty
        # bin. If we ever see one, we widen this assertion.
        for bin_indices in flat_bins:
            assert bin_indices, "FFD produced an empty bin; _adjust_bin_count should prevent this"

        # ------------------------------------------------------------------
        # 4. Build per-row tensors: sequences, attention_mask, loss_mask
        # ------------------------------------------------------------------
        pad_token_id = self.tokenizer.pad_token_id
        num_bins = len(flat_bins)

        n_samples = len(examples)
        logger.info(
            f"sequence packing | packed {n_samples} samples into {num_bins} bins "
            f"(~{num_bins // dp_size}/DP rank, bin_capacity={bin_capacity} tokens)"
        )

        # Fill NumPy buffers by slice, then convert once.
        sequences_np = np.full((num_bins, max_packed_len), pad_token_id, dtype=np.int64)
        attention_mask_np = np.zeros((num_bins, max_packed_len), dtype=np.int64)
        # loss_mask is one position shorter than the row to match
        # `token_logprobs[:, :-1]` semantics inside the loss function.
        loss_mask_np = np.zeros((num_bins, max_packed_len - 1), dtype=np.float32)
        loss_mask_width = max_packed_len - 1

        for row_idx, bin_indices in enumerate(flat_bins):
            row_offset = 0
            for ex_idx in bin_indices:
                s = seq_lengths[ex_idx]
                sequences_np[row_idx, row_offset : row_offset + s] = full_input_ids[ex_idx]
                attention_mask_np[row_idx, row_offset : row_offset + s] = 1

                # loss_mask[p] predicts token p+1; leave each sub-seq's final
                # token zero to prevent cross-boundary loss.
                if s > 1:
                    write_end = min(row_offset + s - 1, loss_mask_width)
                    n_write = write_end - row_offset
                    if n_write > 0:
                        loss_mask_np[row_idx, row_offset:write_end] = full_loss_masks[ex_idx][1 : 1 + n_write]

                # Advance row_offset, padding sub-seq to the TP/CP layout
                # multiple, plus FP8's 16-token local-rank multiple when active.
                row_offset += _round_up(s, align_size)

        # Count response-token loss slots before normalization.
        total_nonpad = int(loss_mask_np.sum())

        sequences = torch.from_numpy(sequences_np)
        attention_mask = torch.from_numpy(attention_mask_np)
        loss_mask = torch.from_numpy(loss_mask_np)

        # ------------------------------------------------------------------
        # 5. Loss normalization
        # ------------------------------------------------------------------
        # The realized gradient is sum(loss * loss_mask) / (num_microbatches
        # * dp_size). Each bin row is one micro-batch, so num_microbatches *
        # dp_size = num_bins. So loss_mask *= num_bins / total_nonpad yields
        # mean_over_nonpad.
        scale = num_bins / max(total_nonpad, 1)
        loss_mask.mul_(scale)

        # ------------------------------------------------------------------
        # 6. Pack into TrainingInputBatch with sub_seq_lengths data field
        # ------------------------------------------------------------------
        # ``sub_seq_lengths`` is genuinely per-sample data: after FFD the
        # batch's "sample" *is* a bin, so ``len(bin_subseq_lengths) == num_bins
        # == batch_size``, co-indexed with ``sequences[r]``. We store it as a
        # ``TensorList`` (one 1-D int tensor per bin, ragged across bins — same
        # pattern as ``image_grid_thw``) so ``MeshDispatch`` shards it per-DP
        # rank automatically alongside ``sequences``/``attention_mask``,
        # eliminating the worker-side per-rank slice. ``preprocess_packed_seqs``
        # and the Megatron packed-logprob scatter want ``list[list[int]]``, so a
        # ``.tolist()`` happens at the ``forward_step`` boundary.
        sub_seq_lengths = TensorList([torch.tensor(lens, dtype=torch.long) for lens in bin_subseq_lengths])
        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "sub_seq_lengths": sub_seq_lengths,
            }
        )
        batch.metadata = {
            "response_length": max_packed_len - 1,
        }
        return batch
