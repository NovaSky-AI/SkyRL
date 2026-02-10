"""Auto micro-batch sizing via GPU memory profiling.

Runs dummy forward + backward passes at increasing micro-batch sizes to find the
largest size that fits within GPU memory (with a safety margin).
"""

import gc

import torch
import torch.distributed as dist
from loguru import logger


def determine_micro_batch_size(
    model: torch.nn.Module,
    strategy,
    max_seq_len: int,
    mini_batch_size_per_gpu: int,
    safety_margin: float = 0.85,
    temperature: float = 1.0,
    compute_entropy: bool = True,
    entropy_requires_grad: bool = False,
) -> int:
    """Determine the largest micro-batch size that fits in GPU memory.

    Searches all divisors of mini_batch_size_per_gpu in ascending order.
    For each candidate a dummy forward + backward pass is executed and the
    actual memory consumed is measured via `torch.cuda.mem_get_info`.

    In distributed settings we `all_reduce(MAX)` the consumed memory so
    that every worker agrees on the result and no deadlocks occur.

    Args:
        model: The wrapped model (e.g. `HFModelWrapper`) already on GPU.
        strategy: `DistributedStrategy` instance (provides `.backward`).
        max_seq_len: Maximum sequence length to profile with
            (`max_prompt_length + max_generate_length`).
        mini_batch_size_per_gpu: Upper bound — the micro-batch size must
            evenly divide this value.
        safety_margin: Fraction of currently free GPU memory that the
            micro-batch is allowed to consume.  E.g. 0.85 means "use up
            to 85% of free memory". The remaining 15% is a buffer for
            memory fragmentation and other random allocations.
        temperature: Sampling temperature forwarded to the model.
        compute_entropy: Whether the model computes entropy in the forward pass
        entropy_requires_grad: Whether entropy computation requires grad.

    Returns:
        The largest micro-batch size (divisor of mini_batch_size_per_gpu)
        that fits safely.
    """
    device = torch.cuda.current_device()

    # Measure free memory after model is loaded but before any batch
    _cleanup_memory()
    torch.cuda.synchronize(device)
    free_baseline, _total = torch.cuda.mem_get_info(device)
    memory_budget = free_baseline * safety_margin
    logger.info(
        f"Auto micro-batch profiling: free baseline={free_baseline / 1e9:.2f} GB, "
        f"budget={memory_budget / 1e9:.2f} GB (safety_margin={safety_margin})"
    )

    candidates = sorted(
        d for d in range(1, mini_batch_size_per_gpu + 1)
        if mini_batch_size_per_gpu % d == 0
    )

    best_size = 1

    for candidate in candidates:
        _cleanup_memory()

        consumed = _profile_candidate(
            model=model,
            strategy=strategy,
            batch_size=candidate,
            seq_len=max_seq_len,
            device=device,
            temperature=temperature,
            compute_entropy=compute_entropy,
            entropy_requires_grad=entropy_requires_grad,
        )

        # When distributed, avoid deadlocks and take worst-case memory usage across all workers
        if dist.is_initialized() and dist.get_world_size() > 1:
            consumed_tensor = torch.tensor([consumed], dtype=torch.long, device=device)
            dist.all_reduce(consumed_tensor, op=dist.ReduceOp.MAX)
            consumed = consumed_tensor.item()

        if consumed <= memory_budget:
            best_size = candidate
            logger.info(
                f"Auto micro-batch profiling: size={candidate} fits "
                f"(consumed={consumed / 1e9:.2f} GB / budget={memory_budget / 1e9:.2f} GB)"
            )
        else:
            logger.info(
                f"Auto micro-batch profiling: size={candidate} exceeds budget "
                f"(consumed={consumed / 1e9:.2f} GB / budget={memory_budget / 1e9:.2f} GB). "
                f"Using size={best_size}."
            )
            break

    _cleanup_memory()

    logger.info(f"Auto micro-batch sizing complete: determined size = {best_size}")
    return best_size


def _profile_candidate(
    model: torch.nn.Module,
    strategy,
    batch_size: int,
    seq_len: int,
    device,
    temperature: float = 1.0,
    compute_entropy: bool = True,
    entropy_requires_grad: bool = False,
) -> int:
    """Run a dummy forward + backward pass and return memory consumed in bytes.

    Memory consumption is measured via `torch.cuda.mem_get_info` which
    reports driver-level free memory.  The difference `free_before −
    free_after` captures all GPU memory usage (PyTorch tensors, CUDA
    context growth, NCCL buffers, etc.), not just PyTorch allocations.
    """
    torch.cuda.synchronize(device)
    free_before, _total = torch.cuda.mem_get_info(device)

    # Dummy inputs
    sequences = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    num_actions = max(seq_len // 2, 1)

    model.train()

    try:
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            action_log_probs, output = model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=temperature,
                return_output=True,
                compute_entropy=compute_entropy,
                entropy_requires_grad=entropy_requires_grad,
            )
            loss = action_log_probs.sum()

        strategy.backward(loss, model, None)

        # Measure memory at the point of peak usage (after backward, before
        # freeing activations and gradients).
        torch.cuda.synchronize(device)
        free_after, _total = torch.cuda.mem_get_info(device)
    finally:
        model.zero_grad(set_to_none=True)

    consumed = max(free_before - free_after, 0)

    del sequences, attention_mask, action_log_probs, output, loss

    return consumed


def _cleanup_memory():
    """Free cached GPU memory between profiling runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
