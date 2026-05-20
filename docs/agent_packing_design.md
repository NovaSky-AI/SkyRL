# Controller-Level Sequence Packing for SkyRL SFT (Megatron)

Date: 2026-05-19

This document describes the implementation of controller-level
sequence packing for SkyRL's SFT trainer. It supersedes
`docs/minibatch_packing_design.md` (worker-side packing) per the architecture
comparison in `docs/packing_architecture_comparison.md` §1-5.

## 1. Goal

Improve SkyRL's packing efficiency for SFT on the Megatron backend, so a
mini-batch with average 188-token sequences (alpaca) or 394-token sequences
(Tulu3) yields ~92% packing efficiency instead of SkyRL's current ~35%
(simulation in `results/packing_sim_2026-05-19/`).

Scope:
- Megatron backend only. FSDP unchanged (separate packing path via
  `flash_attn.bert_padding`).
- `context_parallel_size == 1`. CP > 1 is enforced off with a runtime assert
  (per the task brief).
- `expert_model_parallel_size == 1`. MoE rollout-expert routing is per-row and
  needs separate handling; out of scope for v1.
- TP, PP, SP all supported (PP triggers uniform padding across stages).

## 2. Where packing lives

Per the architecture comparison, the most efficient placement is on the
**controller** at batch construction time, before any worker dispatch.
SkyRL's controller already pads rows to `max_len_in_batch` in
`collate_sft_batch` and then dispatches via `MeshDispatch`. The packing
decision plugs in at the same layer: in a new
`SFTTrainerWithPacking.collate_batch`, we run FFD over the batch's per-row
lengths, build a packed `TrainingInputBatch` whose rows are bins (each bin a
concatenation of original sequences plus optional padding to honor TP / CP /
SP divisibility), and pass that to `train_step` exactly as before. From
`MeshDispatch.dispatch` downward there is no interface change.

This keeps the worker honest about a single semantic: "given a padded batch
of THD-friendly rows, run `preprocess_packed_seqs` -> `forward_step` ->
loss." The worker only needs to learn how to read a `cu_seqlens` metadata
field that already enumerates all sub-sequences inside each row, instead of
inferring one sub-sequence per row from `attention_mask.sum(dim=-1)`.

## 3. Design points

| Concern | Approach |
|---|---|
| Algorithm | FFD only (per task brief). MFFD/FFShuffle ports left as follow-up. |
| Bin capacity | `sft_cfg.max_length` (per task brief step 3). |
| DP symmetry | `min_bin_count=dp_size`, `bin_count_multiple=dp_size`, and `_adjust_bin_count` redistributes one sequence per empty bin so the final bin count is exactly a multiple of `dp_size`. See `bin_packing.py`. |
| Bin -> DP-shard | `shard_idx = bin_idx % shards` round-robin. |
| `cu_seqlens` | Built per packed row in the controller-side collator; un-divided length. CP enforced off so `// cp_size` is identity. |
| PP padding | All packed rows padded to `max_packed_len_in_batch` when PP > 1; identity when PP = 1. |
| Loss normalization | SkyRL pre-scales `loss_mask *= batch_size / (mbs * total_nonpad)` in the controller (cancels Megatron's `1/num_microbatches` + DP-average). With packing, `mbs * num_microbatches == batch_per_dp`, so the scaling factor still resolves correctly when we set `effective_mbs = num_bins_per_dp_rank` and `num_microbatches_per_dp = ceil(num_bins_per_dp_rank / packed_micro_batch_size)`. The cleaner approach we adopt: rewrite the scaling in terms of `total_nonpad` (the only term affected by packing) since `batch_size / (mbs * num_microbatches * dp_size) == 1 / batch_per_dp` and `batch_per_dp` is the number of bins per DP rank, which is `num_microbatches_packed * packed_mbs`. Concretely: `loss_mask *= 1 / total_nonpad`, and `1/num_microbatches` already gets re-cancelled inside the worker because we set `num_microbatches=len(micro_buffer)` exactly as today. See §6. |
| TP/CP per-seq align | We delegate to the existing `preprocess_packed_seqs` which already aligns per-row in `align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size`. Each sub-seq becomes one row entry in `cu_seqlens`, so the existing alignment logic applies to each sub-seq. |
| Iterator | We rely on `BatchIterator` already in the worker. Each bin is a row; the iterator yields chunks of `packed_micro_batch_size` bins. |

## 4. Data shapes

Original (un-packed): `[B, max_len_in_batch]` rows where every row is one
sequence. Per row, `attention_mask.sum() = real_seq_len_i`.

Packed: `[K, max_packed_len_in_batch]` rows where `K = number_of_bins`. Each
row contains one or more sub-sequences concatenated end-to-end, optionally
right-padded with `pad_token_id` to honor TP alignment and (when PP > 1) the
global-max packed length. We carry an auxiliary metadata field
`sub_seq_lengths_per_row: List[List[int]]` (one list per bin row) so the
worker's `preprocess_packed_seqs` can emit `cu_seqlens` enumerating every
sub-sequence, not just one entry per row.

`loss_mask` is also concatenated per bin (right-padded with zeros). For
boundary safety, `loss_mask` already has 0 at the leading prompt-tokens of
each sub-seq (chat template marks only the assistant tokens as 1). The very
last token of every sub-seq prediction (which corresponds to predicting the
first token of the *next* sub-seq when label-shifted) is masked out by setting
`loss_mask[last_token_of_subseq] = 0`. This is the standard "boundary loss
mask" trick. The existing `from_parallel_logits_to_logprobs_packed_sequences`
in `model_utils.py` already does per-subseq `roll(shifts=-1)` using
`cu_seqlens_padded`, so as long as `cu_seqlens_padded` enumerates every
sub-seq, the label shift is boundary-safe at the logit level. The mask zero on
the boundary token is the safety belt that also handles the right-padded
filler tokens.

## 5. DP shard layout

After FFD:

1. `_adjust_bin_count` redistributes one sequence from the largest bins into
   empty bins until total bin count is a multiple of `dp_size`.
2. Round-robin: bin `i` goes to DP rank `i % dp_size`. Each rank gets
   exactly `K // dp_size` bins.
3. Each rank's bins are concatenated as the rows of its TrainingInputBatch.

Across DP ranks, bin counts are identical by construction. Across PP stages,
the packed sequence length is the same because we pad to the global max when
`pp_size > 1`.

## 6. Loss normalization

Current SkyRL formula (`sft_trainer.py:670`):
```
loss_mask *= batch_size / (micro_batch_size * total_nonpad)
```

This pre-scales so that, after Megatron's internal `1/num_microbatches`
divisor and the DP-average all-reduce, the realized loss equals
`mean_over_nonpad_tokens(loss)`:
- effective grad = `sum(loss * loss_mask) / (num_microbatches * dp_size)`
- with `num_microbatches = ceil((batch_size / dp_size) / micro_batch_size)`,
  the product `micro_batch_size * num_microbatches * dp_size = batch_size`,
  so `loss_mask *= batch_size / (mbs * total_nonpad)` reduces the realized
  gradient to `sum / total_nonpad = mean_over_nonpad_tokens`.

With packing, `batch_size` semantics for the scaling formula must be reframed:
the row dimension we ship to workers is now `K` bins. Let:
- `K = num_bins_total` (sum across DP ranks)
- `K_per_dp = K / dp_size` (constant by construction of `_adjust_bin_count`)
- `packed_mbs` = packed micro batch size (rows per worker forward pass)
- `num_microbatches = K_per_dp / packed_mbs`

Then `packed_mbs * num_microbatches * dp_size = K`, and replacing the SkyRL
scaling formula's `batch_size / (mbs * total_nonpad)` with
`K / (packed_mbs * total_nonpad)` yields the same realized gradient
`mean_over_nonpad`. `total_nonpad` is unchanged: it is the sum of `loss_mask`
across the entire mini-batch, before or after packing (re-ordering tokens does
not change their count).

We pass `packed_mbs` to the collator as the new effective micro batch size and
`K` (number of bins) as the new batch dimension. This is what
`SFTTrainerWithPacking.collate_batch` computes.

## 7. Worker changes

`preprocess_packed_seqs` is taught to accept an optional `sub_seq_lengths`
parameter:

```python
def preprocess_packed_seqs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True,
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> tuple[torch.Tensor, PackedSeqParams]:
```

When `sub_seq_lengths is None`, behavior is unchanged (one sub-seq per row,
inferred from `attention_mask.sum`). When provided, each row's
`attention_mask` is replaced by the concatenation of per-subseq masks, and
`cu_seqlens` enumerates every sub-seq (totaling
`sum(len(s) for s in sub_seq_lengths) + 1` entries). TP/CP alignment is
applied per sub-seq.

The worker reads `sub_seq_lengths` from a per-batch metadata field
`sub_seq_lengths` on `TrainingInputBatch.metadata` (a Python list of lists,
which survives Ray pickling unchanged). `BatchIterator` is taught to slice
the metadata's `sub_seq_lengths` alongside the tensor rows. When the field is
absent (RL path, FSDP path, unchanged SFT path), behavior is unchanged.

## 8. Configuration

A new field `use_minibatch_packing: bool = False` is added to `SFTConfig`. The
controller-level minibatch packing path activates when both
`use_sample_packing=True` and `use_minibatch_packing=True`. The flag plumbs
through `build_skyrl_config_for_sft` into a new
`TrainerConfig.use_minibatch_packing` field that the worker reads only to
forward `sub_seq_lengths`-aware preprocessing.

A new field `packed_micro_batch_size_per_gpu: Optional[int] = None` on
`SFTConfig` gives the number of bins per micro-batch (default: 1, so each
bin is one micro-batch). When `None`, defaults to
`micro_train_batch_size_per_gpu` for behavioral compatibility tests, but
v1 ships with `1`.

## 9. Test plan

CPU-only unit tests (`tests/backends/skyrl_train/distributed/test_bin_packing.py`):
1. FFD basic: deterministic output, no bin overflow.
2. `_adjust_bin_count`: empty-bin redistribution preserves valid packing.
3. `min_bin_count` / `bin_count_multiple` constraints satisfied.

CPU-only unit tests for `preprocess_packed_seqs` multi-sub-seq mode
(`tests/backends/skyrl_train/distributed/test_preprocess_packed_seqs_multiseq.py`):
1. With one sub-seq per row, output matches the no-`sub_seq_lengths` path.
2. With two sub-seqs in a row, `cu_seqlens` has 3 entries and the packed
   buffer concatenates both.
3. With TP > 1, each sub-seq is padded to a multiple of `tp_size`.

CPU-only collator test
(`tests/train/test_sft_packing_collate.py`):
1. Sub-seq lengths in metadata match per-row attention sums.
2. Loss-mask sum is invariant before vs after packing.
3. Bin count is a multiple of `dp_size` even when input is not.

GPU smoke (out of scope for this autonomous run; documented for follow-up):
- `tests/backends/skyrl_train/gpu/gpu_ci/test_training_step.py`-style smoke
  with `use_minibatch_packing=True` on Qwen2.5-0.5B, DP=2, TP=1, PP=1.
- E2E parity vs unpacked: 50-step run on Tulu3, Qwen2.5-1.5B-Instruct, FP=4.

## 10. Out-of-scope items (intentional)

- MFFD, FFShuffle, Concatenative: only FFD shipped. The
  `SequencePackerRegistry` abstraction is in place so adding them is a new
  packer class plus an enum value.
- `make_sequence_length_divisible_by`: not surfaced as a config. We always
  honor `tp_size` (and would honor `2*cp_size * tp_size` once CP > 1 is
  enabled). The existing `align_size` inside `preprocess_packed_seqs` handles
  this; we apply it per sub-seq.
- `LossPostProcessor` / `cp_normalize`: not needed at `cp_size = 1` because
  Megatron's CP-specific `* cp_size / num_microbatches` scaling does not
  apply.
- Fused-loss wrapper (`SequencePackingFusionLossWrapper`): SkyRL uses its own
  `cross_entropy` policy loss path; no fused variant is needed for v1.
