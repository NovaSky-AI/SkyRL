"""Context-parallel (CP) layout helpers for Qwen4Exp's PLE and QSA.

megatron-core / Transformer Engine split every packed (THD) sequence of padded length ``L`` into
``2 * cp`` chunks of ``h = L / (2 * cp)`` tokens; CP rank ``r`` holds chunks ``r`` and
``2 * cp - 1 - r`` back to back, at local rows ``[cu[i] / cp, cu[i + 1] / cp)`` (``cu`` = the
global padded ``cu_seqlens``). PLE and the QSA indexer / attention look across that split: PLE
for a few tokens back (its n-gram shift and dilated conv), QSA over whole sequences. These
helpers map local rows to global stream positions and gather rows over the CP group, with
gradients flowing back to the rank that owns each row.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from torch import Tensor


def cp_global_positions(cu_seqlens_padded: Tensor, cp_rank: int, cp_size: int) -> Tensor:
    """[T / cp_size] global stream positions of CP rank ``cp_rank``'s local rows."""
    cu = cu_seqlens_padded.long()
    lengths = cu[1:] - cu[:-1]
    if cp_size == 1:
        return torch.arange(int(cu[-1]), device=cu.device)
    if bool((lengths % (2 * cp_size)).any()):
        raise ValueError(f"Packed sequence lengths {lengths.tolist()} are not multiples of 2 * CP = {2 * cp_size}.")
    local_lengths = lengths // cp_size
    seq = torch.repeat_interleave(torch.arange(lengths.numel(), device=cu.device), local_lengths)
    local_starts = cu[:-1] // cp_size
    u = torch.arange(int(cu[-1]) // cp_size, device=cu.device) - local_starts[seq]
    h = (lengths // (2 * cp_size))[seq]
    first = u < h
    return torch.where(first, cu[:-1][seq] + cp_rank * h + u, cu[1:][seq] - (cp_rank + 1) * h + (u - h))


@dataclass
class CPLayout:
    """The CP split of one packed micro-batch, shared by every PLE / QSA layer of the forward."""

    group: Optional[dist.ProcessGroup]
    size: int
    rank: int
    cu_seqlens_padded: Tensor  # global [num_seqs + 1]
    positions: Tensor  # [T_local] global stream positions of this rank's rows
    all_positions: Tensor  # [size * T_local] positions of every rank's rows, rank-major (gather order)
    inverse: Tensor  # [T] index into the rank-major gather of each global position

    @property
    def num_tokens(self) -> int:
        return int(self.cu_seqlens_padded[-1])

    @classmethod
    def build(cls, cu_seqlens_padded: Tensor, group: Optional[dist.ProcessGroup]) -> "CPLayout":
        size = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0
        all_positions = torch.cat([cp_global_positions(cu_seqlens_padded, r, size) for r in range(size)])
        inverse = torch.empty_like(all_positions)
        inverse[all_positions] = torch.arange(all_positions.numel(), device=all_positions.device)
        n = all_positions.numel() // size
        return cls(group, size, rank, cu_seqlens_padded, all_positions[rank * n : (rank + 1) * n], all_positions, inverse)

    def gather(self, x: Tensor) -> Tensor:
        """All of ``x`` ([T_local, ...]) in global stream order ([T, ...]).

        Backward reduce-scatters, so each rank gets the summed gradient of its own rows.
        """
        if self.size == 1:
            return x
        gathered = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True, group=self.group)
        return gathered[self.inverse]

    def gather_no_grad(self, x: Tensor) -> Tensor:
        if self.size == 1:
            return x
        out = x.new_empty((self.size * x.shape[0],) + tuple(x.shape[1:]))
        dist.all_gather_into_tensor(out, x.contiguous(), group=self.group)
        return out[self.inverse]

    def gather_blocks_no_grad(self, blocks: Tensor, block_size: int) -> Tensor:
        """Per-block rows ([T_local / block_size, ...], local block order) of every rank, in global
        block order ([T / block_size, ...]). Needs chunks of whole, aligned blocks."""
        if self.size == 1:
            return blocks
        out = blocks.new_empty((self.size * blocks.shape[0],) + tuple(blocks.shape[1:]))
        dist.all_gather_into_tensor(out, blocks.contiguous(), group=self.group)
        ordered = torch.empty_like(out)
        ordered[self.all_positions[::block_size] // block_size] = out
        return ordered

    def chunk_ends(self, positions: Tensor) -> Tensor:
        """End (exclusive) of the CP chunk holding each global position."""
        cu = self.cu_seqlens_padded.long()
        seq = torch.searchsorted(cu, positions, right=True) - 1
        h = (cu[seq + 1] - cu[seq]) // (2 * self.size)
        return cu[seq] + ((positions - cu[seq]) // h + 1) * h

    def halo(self, x: Tensor, width: int):
        """The rows of other ranks within ``width`` tokens before each local chunk.

        Returns ``(rows, positions)``: the last ``width`` rows of every chunk of every rank (a whole
        chunk if shorter), i.e. every row any chunk's ``width``-token lookback can reach, gathered
        over CP with gradients reduced back to their owners. Every rank holds the same number of
        such rows (all ranks hold one chunk of each length per sequence half), so a plain
        all-gather suffices.
        """
        keep = self.chunk_ends(self.positions) - self.positions <= width
        rows = x[keep]
        gathered = gather_from_sequence_parallel_region(rows, tensor_parallel_output_grad=True, group=self.group)
        all_keep = self.chunk_ends(self.all_positions) - self.all_positions <= width
        return gathered, self.all_positions[all_keep]
