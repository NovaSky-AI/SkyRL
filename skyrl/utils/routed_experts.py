import torch


def make_replay_padding_indices(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return dummy routes with ``topk`` distinct experts in every row."""
    if not shape or shape[-1] < 1:
        raise ValueError(f"Replay route padding requires a positive topk dimension, got {shape}")
    padding_row = torch.arange(shape[-1], dtype=dtype, device=device)
    return padding_row.expand(shape).clone()
