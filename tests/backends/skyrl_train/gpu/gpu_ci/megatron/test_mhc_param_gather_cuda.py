"""Two-rank regression for mHC functional reads and overlapped DDP gathers."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# The production image must load the worker before torch's shared libraries.
# The subprocess is also independent of pytest's already-imported parent state.
if __name__ == "__main__":
    importlib.import_module("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")

import pytest
import torch


def _distributed_main():
    """Isolate functional child-weight reads with the pinned Core DDP bucket lifecycle."""
    import json
    import os
    import traceback
    from datetime import timedelta

    import torch
    from megatron.core import parallel_state
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.transformer import TransformerConfig

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3))
    parallel_state.initialize_model_parallel()
    rank = torch.distributed.get_rank()

    from megatron.core.transformer.hyper_connection import HyperConnectionModule

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cfg = TransformerConfig(
                num_layers=3, num_attention_heads=1, hidden_size=128, enable_mhc_connections=True, use_fused_mhc=False
            )
            self.layers = torch.nn.ModuleList([HyperConnectionModule(cfg, i + 1) for i in range(3)])
            self.output = torch.nn.Linear(512, 128, bias=False)

        def forward(self, x):
            for layer in self.layers:
                y, h_res, h_post = layer(x)
                x = y.repeat(1, 1, 4) + h_res.square().mean((-1, -2)).unsqueeze(-1) + h_post.mean(-1, keepdim=True)
                x = x.tanh()
            return self.output(x)

    def emit(model, label):
        if rank:
            return
        names = {p: n for n, p in model.module.named_parameters()}
        print(
            json.dumps(
                {
                    "stage": label,
                    "buckets": [
                        {
                            "params": [names[p] for p in group.params],
                            "dispatched": group.param_gather_dispatched,
                            "handle": group.param_gather_handle is not None,
                        }
                        for group in model.bucket_groups
                    ],
                }
            ),
            flush=True,
        )

    from skyrl.backends.skyrl_train.patches.megatron.patch_mhc_param_gather import (
        patch_mhc_param_gather,
    )

    for fixed in (False, True):
        if fixed:
            patch_mhc_param_gather()
        torch.manual_seed(42)
        module = Model().cuda()
        config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=12000,
            grad_reduce_in_fp32=True,
        )
        layout = DistributedOptimizer.compute_full_param_layout(
            list(module.parameters()), config.bucket_size, 2, config
        )
        model = DistributedDataParallel(
            TransformerConfig(num_layers=1, num_attention_heads=1), config, module, full_param_layout=layout
        )
        x = torch.randn(4, 1, 512, device="cuda")
        label = f"fixed={fixed}"
        try:
            with torch.no_grad():
                model(x)
            emit(model, label + ":after_forward_only")
            loss = model(x).float().square().mean()
            assert torch.isfinite(loss), loss
            loss.backward()
            emit(model, label + ":before_finish_grad_sync")
            model.finish_grad_sync()
            emit(model, label + ":after_finish_grad_sync")
            model.zero_grad_buffer()
            with torch.no_grad():
                model(x)
            emit(model, label + ":second_forward_passed")
            assert fixed, "Unpatched mHC no longer reproduces: review/remove the adapter"
        except AssertionError:
            emit(model, label + ":assertion")
            traceback.print_exc()
            if fixed:
                raise
            assert traceback.extract_tb(sys.exc_info()[2])[-1].name == "start_param_sync"
            assert any(g.param_gather_handle is not None and not g.param_gather_dispatched for g in model.bucket_groups)
            print("UNFIXED_ASSERTION_REPRODUCED", flush=True)
        finally:
            with torch.no_grad():
                model.disable_forward_pre_hook()
            torch.distributed.barrier()
        del model, module

    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.megatron
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_mhc_functional_weight_gather_lifecycle():
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
    assert "UNFIXED_ASSERTION_REPRODUCED" in result.stdout
    assert "fixed=True:second_forward_passed" in result.stdout


if __name__ == "__main__":
    _distributed_main()
