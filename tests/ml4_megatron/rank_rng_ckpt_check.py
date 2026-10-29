"""Every rank restores its own RNG streams from a Megatron dist checkpoint (multi-process check).

torchrun --nproc-per-node 4 tests/ml4_megatron/rank_rng_ckpt_check.py <ckpt_dir>

Saves with MegatronStrategy's RNG entries and the same save/load strategy wrappers, then perturbs every
generator, loads, and compares the next draws of torch CPU, CUDA, numpy, random and Megatron's CUDA RNG
tracker with the draws recorded right after the save. Exits non-zero on any rank mismatch.
"""

import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist


def draws():
    from megatron.core import tensor_parallel

    out = [torch.rand(4).tolist(), torch.rand(4, device="cuda").tolist(), np.random.rand(4).tolist(), random.random()]
    with tensor_parallel.get_cuda_rng_tracker().fork():
        out.append(torch.rand(4, device="cuda").tolist())
    return out


def main(ckpt_dir: str) -> int:
    from megatron.core import dist_checkpointing
    from megatron.core import parallel_state as mpu
    from megatron.core.dist_checkpointing.serialization import (
        get_default_save_sharded_strategy,
    )
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
        FullyParallelSaveStrategyWrapper,
    )
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistLoadShardedStrategy,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
        MegatronStrategy,
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(100 + rank)
    np.random.seed(200 + rank)
    random.seed(300 + rank)
    strat = MegatronStrategy.__new__(MegatronStrategy)

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()
    sd = {"rng": strat.get_rng_state(), "rng_per_rank": strat._rank_rng_object(strat._rank_rng_state())}
    save = FullyParallelSaveStrategyWrapper(
        get_default_save_sharded_strategy("torch_dist"), mpu.get_data_parallel_group()
    )
    dist_checkpointing.save(sharded_state_dict=sd, checkpoint_dir=ckpt_dir, sharded_strategy=save)
    want = draws()

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    model_parallel_cuda_manual_seed(999)
    torch.cuda.manual_seed(999)

    load_sd = {}
    if strat._checkpoint_has_rank_rng(ckpt_dir):
        load_sd["rng_per_rank"] = strat._rank_rng_object(None)
    load = FullyParallelLoadStrategyWrapper(TorchDistLoadShardedStrategy(), mpu.get_data_parallel_group())
    state = dist_checkpointing.load(sharded_state_dict=load_sd, checkpoint_dir=ckpt_dir, sharded_strategy=load)
    if "rng_per_rank" in state:
        strat._load_rank_rng_state(state["rng_per_rank"])
    else:
        strat.load_rng_state(state["rng"])
    got = draws()
    ok = got == want
    flags = [None] * dist.get_world_size()
    dist.all_gather_object(flags, ok)
    if rank == 0:
        print("per-rank RNG restored:", flags, "PASS" if all(flags) else "FAIL", flush=True)
    dist.destroy_process_group()
    return 0 if all(flags) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
