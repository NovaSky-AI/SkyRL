"""Under MTP, a training micro-batch's position ids are packed in the token ids' layout, so megatron-core's CP roll
of them stays in bounds and rolls each segment like a global per-segment roll.

The regression: the position ids were packed by a second ``preprocess_packed_seqs`` call without the FP8 alignment
(nor ``sub_seq_lengths``), so with FP8 the token ids' ``cu_seqlens`` ran past the position-id shard and the CP roll's
``index_select`` / ``index_copy_`` went out of bounds (a device-side assert on every rank of the last pipeline stage
of GLM-5.3 at CP16 with MXFP8). Everything here runs on CPU; the CP exchange is emulated across ranks in-process.
"""

import pytest
import torch

pytestmark = pytest.mark.megatron

mtp = pytest.importorskip("megatron.core.transformer.multi_token_prediction", reason="requires the megatron extra")

from skyrl.backends.skyrl_train.distributed.megatron import (  # noqa: E402
    megatron_utils,
)
from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (  # noqa: E402
    get_packed_seq_align_size,
)
from skyrl.backends.skyrl_train.workers.megatron import (  # noqa: E402
    megatron_model_wrapper as wrapper,
)


class _CPGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _row_offsets(row, align):
    """Start column of each sub-sequence in a controller-packed row, as the collator lays them out."""
    offsets, running = [], 0
    for n in row:
        offsets.append(running)
        running += -(-n // align) * align
    return offsets


def _batch(lengths, sub_seq_lengths, cp, fp8_recipe):
    """Rows of token ids, attention mask and per-sequence position ids (one sequence per row unless sub_seq_lengths).

    Controller-packed sub-sequences start at the packing alignment the collator uses (``get_packed_seq_align_size``).
    """
    g = torch.Generator().manual_seed(0)
    rows = sub_seq_lengths or [[n] for n in lengths]
    seq_align = get_packed_seq_align_size(1, cp, fp8_enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe)
    width = max(_row_offsets(r, seq_align)[-1] + r[-1] for r in rows) + 7
    sequences = torch.zeros(len(rows), width, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), width, dtype=torch.bool)
    position_ids = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        for offset, n in zip(_row_offsets(row, seq_align), row):
            sequences[i, offset : offset + n] = torch.randint(10, 150000, (n,), generator=g)
            attention_mask[i, offset : offset + n] = True
            position_ids[i, offset : offset + n] = torch.arange(n)
    return sequences, attention_mask, position_ids


def _pack(monkeypatch, cp, rank, batch, sub_seq_lengths, fp8_recipe):
    for name, value in (
        ("get_tensor_model_parallel_world_size", 1),
        ("get_context_parallel_world_size", cp),
        ("get_context_parallel_rank", rank),
    ):
        monkeypatch.setattr(megatron_utils.mpu, name, lambda value=value: value)
    sequences, attention_mask, position_ids = batch
    return wrapper._pack_microbatch_inputs(
        sequences,
        attention_mask,
        position_ids,
        pre_process=True,
        position_ids_pre_process=True,
        sub_seq_lengths=sub_seq_lengths,
        fp8_enabled=fp8_recipe is not None,
        fp8_recipe=fp8_recipe,
        shard_for_cp=True,
    )


def _cp_roll(monkeypatch, shards, packed_seq_params, cp):
    """megatron-core's packed CP roll on every rank's shard, with the boundary exchange emulated across ranks."""
    sent = {}

    def capture(rank):
        def exchange(first, second, group):
            sent[rank] = (first, second)
            return torch.zeros_like(first), torch.zeros_like(second)

        return exchange

    def deliver(rank):
        def exchange(first, second, group):
            # megatron-core's _exchange_cp_roll_boundaries: the next rank's first, the previous rank's second.
            nxt = sent[rank + 1][0] if rank < cp - 1 else second
            prev = sent[rank - 1][1] if rank > 0 else torch.zeros_like(second)
            return nxt, prev

        return exchange

    for exchange in (capture, deliver):
        rolled = []
        for rank, shard in enumerate(shards):
            monkeypatch.setattr(mtp, "_exchange_cp_roll_boundaries", exchange(rank))
            rolled.append(mtp._roll_tensor_packed_seq(shard, -1, -1, packed_seq_params, _CPGroup(cp))[0])
    return rolled


def _unshard(shards, cu_seqlens, cp):
    """The global packed stream from the CP shards: rank r holds chunks r and 2cp-1-r of every segment."""
    out = []
    for i in range(len(cu_seqlens) - 1):
        chunk = (cu_seqlens[i + 1] - cu_seqlens[i]) // (2 * cp)
        start = cu_seqlens[i] // cp
        chunks = [None] * (2 * cp)
        for r, shard in enumerate(shards):
            chunks[r] = shard[0, start : start + chunk]
            chunks[2 * cp - 1 - r] = shard[0, start + chunk : start + 2 * chunk]
        out.extend(chunks)
    return torch.cat(out)


def _global_roll(stream, cu_seqlens):
    """Each segment rolled one token left, its final slot zeroed."""
    out = torch.zeros_like(stream)
    for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        out[a : b - 1] = stream[a + 1 : b]
    return out


CASES = [
    # cp, sequence lengths (one per row) or controller-packed rows, fp8 recipe (None: BF16)
    # The final segment's FP8 alignment tail is at least its own length: index_select ran out of bounds.
    pytest.param(16, [9000, 7001, 100], None, "mxfp8", id="cp16-mxfp8-short-last"),
    # A shorter tail: index_copy_ ran out of bounds.
    pytest.param(16, [9000, 7001, 333, 4100], None, "mxfp8", id="cp16-mxfp8"),
    pytest.param(8, [5000, 3001, 999], None, "mxfp8", id="cp8-mxfp8"),
    pytest.param(8, [4093, 77, 2500, 13], None, "mxfp8", id="cp8-mxfp8-uneven"),
    pytest.param(4, [301, 777, 50], None, "blockwise", id="cp4-blockwise"),
    pytest.param(4, None, [[301, 129], [777], [64, 5, 90]], None, id="cp4-bf16-controller-packed"),
    pytest.param(8, None, [[1000, 501], [3333, 17]], "mxfp8", id="cp8-mxfp8-controller-packed"),
]


@pytest.mark.parametrize("cp,lengths,sub_seq_lengths,fp8_recipe", CASES)
def test_mtp_position_ids_share_the_token_layout_and_roll_in_bounds(
    monkeypatch, cp, lengths, sub_seq_lengths, fp8_recipe
):
    batch = _batch(lengths, sub_seq_lengths, cp, fp8_recipe)
    token_shards, position_shards = [], []
    for rank in range(cp):
        tokens, packed_seq_params, positions = _pack(monkeypatch, cp, rank, batch, sub_seq_lengths, fp8_recipe)
        assert positions.shape == tokens.shape
        token_shards.append(tokens)
        position_shards.append(positions)
    cu = packed_seq_params.cu_seqlens_q_padded.tolist()
    # The position ids sit where packed_seq_params puts each segment: 0..n-1, then padding.
    real = [n for row in (sub_seq_lengths or [[n] for n in lengths]) for n in row]
    expected = torch.zeros(cu[-1], dtype=torch.long)
    for start, n in zip(cu[:-1], real):
        expected[start : start + n] = torch.arange(n)
    assert torch.equal(_unshard(position_shards, cu, cp), expected)
    # The MTP roll of both, as megatron-core does it under CP (the crash was its index_select / index_copy_ here).
    for shards in (token_shards, position_shards):
        rolled = _cp_roll(monkeypatch, shards, packed_seq_params, cp)
        assert torch.equal(_unshard(rolled, cu, cp), _global_roll(_unshard(shards, cu, cp), cu))
