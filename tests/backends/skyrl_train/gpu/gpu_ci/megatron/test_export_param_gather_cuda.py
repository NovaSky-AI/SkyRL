"""Verify complete exported weights after precision-aware distributed Adam updates."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    importlib.import_module("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")

import pytest
import torch


def _distributed_main():
    """Check complete exported BF16 weights after a real precision-aware Adam step."""
    import json
    import os
    from datetime import timedelta
    from types import SimpleNamespace

    import torch
    from megatron.core import parallel_state
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.tensor_parallel.layers import (
        set_defaults_if_not_set_tensor_model_parallel_attributes,
    )
    from megatron.core.transformer import TransformerConfig

    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronWorker,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3))
    parallel_state.initialize_model_parallel()
    rank = torch.distributed.get_rank()

    def weights(model):
        return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

    for overlap in (False, True):
        module = (
            torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False), torch.nn.Linear(32, 32, bias=False))
            .cuda()
            .bfloat16()
        )
        with torch.no_grad():
            for param in module.parameters():
                param.fill_(0.25)
                set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        cfg = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=overlap,
            overlap_grad_reduce=True,
            bucket_size=1024,
            grad_reduce_in_fp32=False,
        )
        layout = DistributedOptimizer.compute_full_param_layout(list(module.parameters()), cfg.bucket_size, 2, cfg)
        model = DistributedDataParallel(
            TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=32),
            cfg,
            module,
            full_param_layout=layout,
        )
        optim_cfg = OptimizerConfig(
            lr=0.01,
            weight_decay=0.0,
            clip_grad=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            main_params_dtype=torch.float32,
            store_param_remainders=True,
            overlap_param_gather=overlap,
        )
        optimizer = get_megatron_optimizer(optim_cfg, [model])
        optimizer.zero_grad()
        model(torch.ones(4, 32, device="cuda", dtype=torch.bfloat16)).float().sum().backward()
        model.finish_grad_sync()
        success, norm, zeros = optimizer.step()
        assert success
        optimizer.zero_grad()
        model.zero_grad_buffer()
        before = weights(model)
        expected = torch.full_like(before, 0.24)
        stale = int((before != expected).sum())
        print(
            json.dumps(
                {
                    "rank": rank,
                    "overlap": overlap,
                    "stale_before_forced_gather": stale,
                    "numel": before.numel(),
                    "values": before.unique().float().tolist(),
                }
            ),
            flush=True,
        )
        assert (stale > 0) == overlap
        # Export through the worker; the stub strategy snapshots what the exporter reads.
        exported = []
        worker = SimpleNamespace(
            actor_module=[model],
            optimizer=optimizer,
            model=model,
            bridge=None,
            megatron_config=SimpleNamespace(
                hf_export_config=SimpleNamespace(distributed_save=False, save_every_n_ranks=1)
            ),
            strategy=SimpleNamespace(save_hf_model=lambda bridge, m, *args, **kwargs: exported.append(weights(m))),
        )
        MegatronWorker.save_hf_model(worker, "unused", tokenizer=None)
        (after,) = exported
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
        assert all(g.param_gather_handle is None for g in model.bucket_groups)
        with torch.no_grad():
            result = model(torch.ones(4, 32, device="cuda", dtype=torch.bfloat16))
        assert torch.isfinite(result).all()
        print("EXPORT_GATHER_PASS overlap=" + str(overlap) + " rank=" + str(rank), flush=True)
        if overlap:
            model.disable_forward_pre_hook()
        torch.distributed.barrier()
        del optimizer, model, module

    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.megatron
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_export_after_distributed_optimizer_step():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
        ],
        env={**os.environ, "NVTE_FLASH_ATTN": "0"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("EXPORT_GATHER_PASS") == 4


if __name__ == "__main__":
    _distributed_main()
