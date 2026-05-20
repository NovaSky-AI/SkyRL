"""
SFT trainer variant that performs FFD bin-packing at the controller, before
dispatching to workers. Once per training step, sequences are packed into
bins of capacity ``max_length``, the bin count is rounded up to a multiple
of ``dp_size`` (so every DP rank gets the same number of micro-batches),
and each bin becomes one row of the resulting :class:`TrainingInputBatch`.

The packed rows carry a per-row ``sub_seq_lengths`` list inside
``TrainingInputBatch.metadata`` so the worker's
:func:`preprocess_packed_seqs` can enumerate every sub-sequence in the
``cu_seqlens`` it emits.

Megatron backend only. CP > 1 is not supported in v1
(`prompts/implement.md` scopes it out).
"""

from __future__ import annotations

from typing import List

import torch

from skyrl.backends.skyrl_train.distributed.megatron.bin_packing import (
    get_packer,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config.sft_config import SFTConfig
from skyrl.train.sft_trainer import SFTTrainer


class SFTTrainerWithPacking(SFTTrainer):
    """SFT trainer with controller-level FFD bin-packing.

    Activates when ``SFTConfig.use_minibatch_packing=True``. The training
    loop, dataset loading, checkpointing, eval, and dispatch are all
    inherited from :class:`SFTTrainer`. The only behavior change is the
    :meth:`collate_batch` override that emits packed rows instead of
    per-example rows.
    """

    def __init__(self, cfg: SFTConfig, skyrl_cfg=None):
        super().__init__(cfg, skyrl_cfg=skyrl_cfg)
        self._validate_packing_cfg()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_packing_cfg(self):
        if self.sft_cfg.strategy != "megatron":
            raise ValueError(
                f"SFTTrainerWithPacking only supports strategy='megatron'; got "
                f"{self.sft_cfg.strategy!r}. Use the FSDP packing path instead."
            )
        if not self.sft_cfg.use_sample_packing:
            raise ValueError(
                "use_minibatch_packing=True requires use_sample_packing=True " "(the worker still uses the THD layout)."
            )
        if not getattr(self.sft_cfg, "use_minibatch_packing", False):
            raise ValueError("SFTTrainerWithPacking was instantiated but use_minibatch_packing=False.")
        if self.sft_cfg.megatron_config.context_parallel_size > 1:
            raise ValueError(
                "use_minibatch_packing=True does not support context_parallel_size > 1 in v1. "
                "Disable packing or set context_parallel_size=1."
            )

    # ------------------------------------------------------------------ #
    # Data path
    # ------------------------------------------------------------------ #

    def _dp_size(self) -> int:
        """Number of DP ranks under the configured Megatron parallelism."""
        total_gpus = self.sft_cfg.placement.num_nodes * self.sft_cfg.placement.num_gpus_per_node
        tp = self.sft_cfg.megatron_config.tensor_model_parallel_size
        pp = self.sft_cfg.megatron_config.pipeline_model_parallel_size
        cp = self.sft_cfg.megatron_config.context_parallel_size
        return total_gpus // (tp * pp * cp)

    def _packed_micro_batch_size(self) -> int:
        """Bins per micro-batch on the worker.

        Defaults to ``micro_batch_size=1`` for packed sequences (each
        bin IS one micro-batch); users can override via
        ``packed_micro_batch_size_per_gpu`` for memory tuning.
        """
        return getattr(self.sft_cfg, "packed_micro_batch_size_per_gpu", None) or 1

    def collate_batch(self, examples: list, batch_size: int) -> TrainingInputBatch:
        """Pack examples into bin rows and return a :class:`TrainingInputBatch`.

        Flow:

        1. Compute per-example sequence lengths.
        2. FFD-pack with ``bin_capacity = max_length``,
           ``min_bin_count = dp_size``, ``bin_count_multiple = dp_size``.
        3. Round-robin assign bins to DP shards (this happens implicitly
           inside ``MeshDispatch.dispatch`` because we lay out bins in
           shard-major order: shard 0 rows first, then shard 1, etc).
        4. Build the per-bin packed row tensors and the per-row
           ``sub_seq_lengths`` metadata.
        """
        # When eval calls collate_batch with a chunk of the eval set, we fall
        # back to the inherited un-packed collate path. Packing only fires on
        # the training-step batch (== self.sft_cfg.batch_size).
        if batch_size != self.sft_cfg.batch_size:
            return super().collate_batch(examples, batch_size=batch_size)

        max_length = self.sft_cfg.max_length
        if max_length is None:
            raise ValueError("SFTTrainerWithPacking requires max_length to be set explicitly.")

        tp_size = self.sft_cfg.megatron_config.tensor_model_parallel_size
        pp_size = self.sft_cfg.megatron_config.pipeline_model_parallel_size
        align_size = tp_size

        dp_size = self._dp_size()

        # ------------------------------------------------------------------
        # 1. Sequence lengths and full-sequence loss masks
        # ------------------------------------------------------------------
        # We need the *full-sequence* loss mask (one entry per token, not
        # just over the response window) so the packed bin row can have a
        # per-position mask with correct boundary zeros.
        seq_lengths: List[int] = []
        full_loss_masks: List[List[int]] = []
        for ex in examples:
            seq_lengths.append(len(ex["input_ids"]))
            n_pad = len(ex["input_ids"]) - ex["num_actions"]
            full_mask = [0] * n_pad + list(ex["loss_mask"])
            assert len(full_mask) == len(ex["input_ids"]), (
                f"Reconstructed full loss_mask length {len(full_mask)} != seq length " f"{len(ex['input_ids'])}"
            )
            full_loss_masks.append(full_mask)

        # ------------------------------------------------------------------
        # 2. FFD pack with DP-symmetry constraints
        # ------------------------------------------------------------------
        # ``packed_mbs`` rows are consumed in a single forward pass on the
        # worker. Megatron's ``forward_backward_func`` receives a single
        # scalar ``micro_batch_size`` for the entire mini-batch, so every
        # per-DP-rank micro-batch must have **exactly** ``packed_mbs`` rows
        # (partial last-MBs with fewer rows raise a shape mismatch inside
        # ``postprocess_packed_seqs`` because ``micro_batch_size`` is fixed
        # at the call site). Forcing the bin count to a multiple of
        # ``dp_size * packed_mbs`` keeps every per-DP-rank shard divisible
        # by ``packed_mbs``.
        packed_mbs = self._packed_micro_batch_size()
        bin_count_multiple = dp_size * packed_mbs
        packer = get_packer(
            "first_fit_decreasing",
            bin_capacity=max_length,
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
        # 3. Compute packed-row lengths (with tp_size alignment per sub-seq)
        #    and the global max packed length (for PP > 1 uniform padding).
        # ------------------------------------------------------------------
        def _round_up(x: int, m: int) -> int:
            return ((x + m - 1) // m) * m

        bin_packed_lengths: List[int] = []
        bin_subseq_lengths: List[List[int]] = []  # one list per bin row
        for bin_indices in flat_bins:
            subseq_lens = [seq_lengths[idx] for idx in bin_indices]
            # Each sub-seq's length is independently aligned to tp_size
            # (matches preprocess_packed_seqs behavior).
            packed_len = sum(_round_up(s, align_size) for s in subseq_lens)
            bin_packed_lengths.append(packed_len)
            bin_subseq_lengths.append(subseq_lens)

        if pp_size > 1:
            # Pad all packed rows to the global max so Megatron's
            # pipeline schedule sees uniform shapes.
            max_packed_len = max(bin_packed_lengths) if bin_packed_lengths else 0
            # Also align the global max to tp_size to keep TP/SP happy.
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

        sequences = torch.full((num_bins, max_packed_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((num_bins, max_packed_len), dtype=torch.long)
        # loss_mask is one position shorter than the row to match
        # `token_logprobs[:, :-1]` semantics inside the loss function.
        loss_mask = torch.zeros((num_bins, max_packed_len - 1), dtype=torch.float)

        total_nonpad = 0  # sum of all 1s in loss_mask (BEFORE scaling)

        for row_idx, bin_indices in enumerate(flat_bins):
            row_offset = 0
            for sub_idx, ex_idx in enumerate(bin_indices):
                ex = examples[ex_idx]
                s = seq_lengths[ex_idx]
                # Write the sub-seq tokens into the row.
                ids = torch.tensor(ex["input_ids"], dtype=torch.long)
                sequences[row_idx, row_offset : row_offset + s] = ids
                attention_mask[row_idx, row_offset : row_offset + s] = 1

                # Build the per-position loss mask for this sub-seq.
                # Position p (in row coords, p in [row_offset, row_offset + s))
                # predicts token at p+1. The loss_mask at p (in the [B, S-1]
                # action_log_probs slot) is 1 iff p+1 is a response/assistant
                # token AND p+1 is in the same sub-seq.
                full_mask = full_loss_masks[ex_idx]  # length s
                # For p in [0, s - 1): mask[p] = full_mask[p + 1].
                # For p == s - 1: 0 (sub-seq boundary or row end).
                # row position p_row = row_offset + p_local.
                for p_local in range(s - 1):
                    target_is_response = full_mask[p_local + 1]
                    row_p = row_offset + p_local
                    if row_p < max_packed_len - 1:
                        loss_mask[row_idx, row_p] = float(target_is_response)
                        if target_is_response:
                            total_nonpad += 1
                # p_local = s - 1 (last token of sub-seq): mask = 0.
                # Already zero by initialization.

                # Advance row_offset, padding sub-seq to tp_size multiple.
                row_offset += _round_up(s, align_size)

        # The total_nonpad we just counted matches sum(loss_mask). Verify in
        # debug logs only — too expensive on hot path for assert.
        if total_nonpad != int(loss_mask.sum().item()):
            # Defensive: recount from the tensor (will diverge only on bugs).
            total_nonpad = int(loss_mask.sum().item())

        # ------------------------------------------------------------------
        # 5. Loss normalization
        # ------------------------------------------------------------------
        # The realized gradient is sum(loss * loss_mask) / (num_microbatches
        # * dp_size). With packing, packed_mbs * num_microbatches * dp_size
        # = num_bins. So loss_mask *= num_bins / (packed_mbs * total_nonpad)
        # yields mean_over_nonpad. ``packed_mbs`` was bound during the FFD
        # constraint step above.
        scale = num_bins / (packed_mbs * max(total_nonpad, 1))
        loss_mask.mul_(scale)

        # ------------------------------------------------------------------
        # 6. Pack into TrainingInputBatch with sub_seq_lengths metadata
        # ------------------------------------------------------------------
        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
            }
        )
        batch.metadata = {
            "response_length": max_packed_len - 1,
            "sub_seq_lengths": bin_subseq_lengths,
        }
        return batch

    # ------------------------------------------------------------------ #
    # Override _validate_batch_parallelism to relax mbs divisibility
    # ------------------------------------------------------------------ #

    def _validate_batch_parallelism(self):
        """With packing, batch_size is the *example* count (not bins) and the
        per-DP-rank bin count == bins_per_shard. The micro batch size in the
        worker config refers to bins-per-micro-batch and is independently
        configurable via ``packed_micro_batch_size_per_gpu``.
        """
        batch_size = self.sft_cfg.batch_size
        total_gpus = self.sft_cfg.placement.num_nodes * self.sft_cfg.placement.num_gpus_per_node
        tp = self.sft_cfg.megatron_config.tensor_model_parallel_size
        pp = self.sft_cfg.megatron_config.pipeline_model_parallel_size
        cp = self.sft_cfg.megatron_config.context_parallel_size
        dp_size = total_gpus // (tp * pp * cp)
        # We require batch_size >= dp_size so every DP rank gets >= 1 bin.
        if batch_size < dp_size:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= dp_size ({dp_size}) when "
                f"use_minibatch_packing=True (each DP rank needs at least one bin)."
            )
        # We do NOT require batch_size % micro_train_batch_size_per_gpu == 0
        # because micro_train_batch_size_per_gpu now refers to bins-per-MB,
        # not examples-per-MB. The bin count is rounded up to a multiple of
        # dp_size by FFD, and bins/MB is a separate knob.
