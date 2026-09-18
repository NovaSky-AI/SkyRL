"""Compatibility helper for exact full-vocabulary logprob comparisons."""

import torch


def check_full_logprobs(trainer: torch.Tensor, rollout: torch.Tensor, mask: torch.Tensor) -> None:
    """Require bitwise-identical float32 vocabulary rows at trainable token positions."""
    if trainer.shape != rollout.shape or trainer.ndim != 3 or mask.shape != trainer.shape[:2]:
        raise ValueError("Full logprob shapes do not match")
    if trainer.dtype != torch.float32 or rollout.dtype != torch.float32:
        raise ValueError("Full logprobs must be float32")
    selected = mask > 0
    left, right = trainer[selected].contiguous(), rollout[selected].contiguous()
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Full logprobs must be finite")
    different = left.view(torch.int32) != right.view(torch.int32)
    if different.any():
        first_row, first_token = different.nonzero(as_tuple=False)[0]
        trainer_bits = int(left.view(torch.int32)[first_row, first_token].item()) & 0xFFFFFFFF
        rollout_bits = int(right.view(torch.int32)[first_row, first_token].item()) & 0xFFFFFFFF
        raise ValueError(
            "Full logprob comparison failed: "
            f"{different.count_nonzero().item()} vocabulary entries differ; "
            f"first selected row={int(first_row.item())}, token={int(first_token.item())}, "
            f"trainer_bits=0x{trainer_bits:08x}, rollout_bits=0x{rollout_bits:08x}"
        )
