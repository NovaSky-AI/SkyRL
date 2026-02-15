"""Auto micro-batch sizing via GPU memory profiling.

Profiles the model at several (batch_size, seq_len) combinations to estimate a token budget.
At runtime any micro-batch we check `batch_size x max_seq_len_in_batch ≤ C`
"""

import gc
from typing import List

import torch
import torch.distributed as dist
from loguru import logger


def determine_token_budget(
    model: torch.nn.Module,
    strategy,
    max_seq_len: int,
    safety_margin: float = 0.85,
    temperature: float = 1.0,
    compute_entropy: bool = True,
    entropy_requires_grad: bool = False,
) -> int:
    """Profile the model to estimate the token budget.

    token budget (C) is defined such that any micro-batch with
    `batch_size x max_seq_len_in_batch ≤ C` fits in GPU memory.

    Args:
        model: The wrapped model (e.g. `HFModelWrapper`) already on GPU.
        strategy: `DistributedStrategy` instance (provides `.backward`).
        max_seq_len: Maximum possible sequence length
            (`max_prompt_length + max_generate_length`).
        safety_margin: Fraction of free GPU memory that counts as "available".
            E.g. 0.85 means use up to 85% of free memory.
        temperature: Sampling temperature forwarded to the model.
        compute_entropy: Whether the model computes entropy in the forward pass.
        entropy_requires_grad: Whether entropy computation requires grad.

    Returns:
        The token budget C (an `int`). A value of 0 means profiling
        failed and the caller should fall back to a fixed micro-batch size.
    """
    device = torch.cuda.current_device()

    _cleanup_memory()
    torch.cuda.synchronize(device)
    free_baseline, _total = torch.cuda.mem_get_info(device)
    memory_budget = int(free_baseline * safety_margin)
    logger.info(
        f"Token-budget profiling: free baseline={free_baseline / 1e9:.2f} GB, "
        f"budget={memory_budget / 1e9:.2f} GB (safety_margin={safety_margin})"
    )

    profile_kwargs = dict(
        temperature=temperature,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
    )

    seq_lens = (torch.linspace(1, 0.25, 3) * max_seq_len).to(dtype=torch.long)
    logger.info(f"Profiling sequence lengths of: {list(seq_lens)}")
    boundary_products: List[int] = []

    for seq_len in seq_lens:
        max_bs = _find_max_batch_size(
            model=model,
            strategy=strategy,
            seq_len=seq_len,
            memory_budget=memory_budget,
            device=device,
            **profile_kwargs,
        )

        if max_bs <= 0:
            logger.warning(f"Token-budget profiling: even batch_size=1 at seq_len={seq_len} exceeds memory budget.")
            if seq_len == seq_lens[0]:
                # can't even fit 1 sample at the longest seq_len
                # return 0 so the caller falls back to fixed micro-batch size.
                logger.warning(f"Cannot even fit 1 sample at seq_len: {seq_len}. Returning 0.")
                return 0
            # skip this seq_len but continue with shorter ones
            continue

        product = max_bs * seq_len
        boundary_products.append(product)
        logger.info(f"Token-budget profiling: seq_len={seq_len} | max_bs={max_bs} -> product={product}")

    if not boundary_products:
        logger.warning("Token-budget profiling: no valid boundary found.  Returning 0.")
        return 0

    token_budget = min(boundary_products)

    # all workers agree now on budget
    if dist.is_initialized() and dist.get_world_size() > 1:
        budget_tensor = torch.tensor([token_budget], dtype=torch.long, device=device)
        dist.all_reduce(budget_tensor, op=dist.ReduceOp.MIN)
        token_budget = budget_tensor.item()

    _cleanup_memory()
    logger.info(f"Token-budget profiling complete: C={token_budget} " f"(boundary products={boundary_products})")
    return token_budget


def _would_oom(
    last_bs: int,
    last_consumed: int,
    candidate_bs: int,
    device,
) -> bool:
    """Estimate whether candidate_bs would OOM based on the last measurement.

    This function _over estimates_ GPU memory usage so as to avoid OOMing.

    Since `memory(batch_size) = base + k * batch_size`, and based is fixed overhead,
    We do:
    ```
    estimated = last_consumed x (candidate_bs / last_bs)
              = (base + k x last_bs) x (candidate_bs / last_bs)
              = base x (candidate_bs / last_bs) + k x candidate_bs
              >= base + k x candidate_bs
    ```
    """
    if last_bs <= 0 or last_consumed <= 0:
        return False
    estimated = last_consumed * (candidate_bs / last_bs)
    _cleanup_memory()
    torch.cuda.synchronize(device)
    free_now, _ = torch.cuda.mem_get_info(device)
    would_skip = estimated > free_now

    # if any worker would oom, all must skip
    if dist.is_initialized() and dist.get_world_size() > 1:
        skip_tensor = torch.tensor([int(would_skip)], dtype=torch.long, device=device)
        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
        would_skip = skip_tensor.item() > 0

    if would_skip:
        logger.debug(
            f"  skipping bs={candidate_bs} (estimated " f"{estimated / 1e9:.2f} GB > free {free_now / 1e9:.2f} GB)"
        )
    return would_skip


def _find_max_batch_size(
    model: torch.nn.Module,
    strategy,
    seq_len: int,
    memory_budget: int,
    device,
    **profile_kwargs,
) -> int:
    """Find the largest batch_size at seq_len that fits in memory_budget.

    Returns 0 if even bs=1 does not fit.
    """
    last_good = 0
    last_consumed = 0
    bs = 1

    while True:
        if bs > 1 and _would_oom(last_good, last_consumed, bs, device):
            break

        consumed = _profile_and_sync(
            model,
            strategy,
            bs,
            seq_len,
            device,
            **profile_kwargs,
        )
        logstr = f"exp-search: bs={bs}, seq_len={seq_len} " f"({consumed / 1e9:.2f} GB / {memory_budget / 1e9:.2f} GB)"
        if consumed <= memory_budget:
            last_good = bs
            last_consumed = consumed
            logger.debug(logstr)
            bs *= 2
        else:
            logger.debug(logstr)
            break

    if last_good <= 1:
        return last_good  # only bs=1 fits

    low = last_good
    high = bs  # bs is the first size that exceeded or was skipped

    while low < high:
        mid = (low + high + 1) // 2
        if _would_oom(last_good, last_consumed, mid, device):
            high = mid - 1
            continue

        consumed = _profile_and_sync(
            model,
            strategy,
            mid,
            seq_len,
            device,
            **profile_kwargs,
        )
        logstr = (
            f"  bin-search: bs={mid}, seq_len={seq_len} " f"({consumed / 1e9:.2f} GB / {memory_budget / 1e9:.2f} GB)"
        )
        if consumed <= memory_budget:
            low = mid
            last_good = mid
            last_consumed = consumed
            logger.debug(logstr)
        else:
            high = mid - 1
            logger.debug(logstr)

    return low


def _profile_and_sync(
    model: torch.nn.Module,
    strategy,
    batch_size: int,
    seq_len: int,
    device,
    **profile_kwargs,
) -> int:
    """Run a profiling pass.

    Returns max consumed memory over all workers in the distributed group
    """
    _cleanup_memory()

    consumed = _profile_candidate(
        model=model,
        strategy=strategy,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        **profile_kwargs,
    )

    if dist.is_initialized() and dist.get_world_size() > 1:
        consumed_tensor = torch.tensor([consumed], dtype=torch.long, device=device)
        dist.all_reduce(consumed_tensor, op=dist.ReduceOp.MAX)
        consumed = consumed_tensor.item()

    return consumed


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
    """Run a dummy forward + backward pass and return memory consumed in bytes."""
    torch.cuda.synchronize(device)
    free_before, _total = torch.cuda.mem_get_info(device)

    sequences = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    num_actions = max(seq_len // 2, 1)
    logger.debug(f"Profiling batch size {batch_size} | seq len {seq_len}")

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
