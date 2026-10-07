"""Variable packed microbatch counts with native schedules and distributed Adam."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

if __name__ == "__main__":
    importlib.import_module("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")


def _distributed_main():
    import json
    from datetime import timedelta
    from types import SimpleNamespace

    from megatron.core import parallel_state
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.enums import ModelType
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
    from megatron.core.tensor_parallel.layers import (
        set_defaults_if_not_set_tensor_model_parallel_attributes,
    )
    from megatron.core.transformer import TransformerConfig

    from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
        MegatronModelWrapper,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3))
    parallel_state.initialize_model_parallel()
    rank = torch.distributed.get_rank()

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model_type = ModelType.encoder_or_decoder
            self.layers = torch.nn.ModuleList([torch.nn.Linear(128, 128, bias=False) for _ in range(3)])

        def set_input_tensor(self, tensor):
            pass

        def forward(self, x):
            for layer in self.layers:
                x = layer(x).tanh()
            return x.float().square().mean()

    def run_case(fixed, overlap):
        torch.manual_seed(42)
        config = TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=128)
        module = Model(config).cuda().bfloat16()
        for p in module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(p)
        cfg = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=False,
            overlap_grad_reduce=overlap,
            bucket_size=12000,
            grad_reduce_in_fp32=True,
        )
        layout = DistributedOptimizer.compute_full_param_layout(list(module.parameters()), cfg.bucket_size, 2, cfg)
        model = DistributedDataParallel(config, cfg, module, full_param_layout=layout)
        opt_cfg = OptimizerConfig(
            lr=0.01,
            weight_decay=0.0,
            clip_grad=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            main_params_dtype=torch.float32,
            store_param_remainders=True,
            overlap_param_gather=False,
        )
        optimizer = get_megatron_optimizer(opt_cfg, [model])
        wrapper = MegatronModelWrapper(SimpleNamespace(remove_microbatch_padding=False), [model], optimizer)
        assert callable(config.no_sync_func) if overlap else config.no_sync_func is None
        if not fixed:
            config.no_sync_func = None  # Reproduce the omitted schedule hook.
        metrics = []
        initial = torch.cat([p.detach().flatten() for p in module.parameters()]).clone()

        def fwd(iterator, mdl):
            loss = mdl(next(iterator))
            return loss, lambda value: (value, {"loss": value.detach()})

        try:
            for step, count in enumerate((4, 4, 3, 5)):
                optimizer.zero_grad()
                model.zero_grad_buffer()
                for g in optimizer.param_groups:
                    g["lr"] = 0.0 if step == 0 else 0.01
                generator = torch.Generator(device="cuda").manual_seed(77 + step + rank)
                batches = [
                    torch.randn(4, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
                    for _ in range(count)
                ]
                result = forward_backward_no_pipelining(
                    forward_step_func=fwd,
                    data_iterator=iter(batches),
                    model=[model],
                    num_microbatches=count,
                    seq_length=4,
                    micro_batch_size=1,
                    forward_only=False,
                )
                counts = [
                    {
                        "actual": list(g.per_param_grad_ready_counts.values()),
                        "golden": list(g.golden_per_param_grad_ready_counts.values()),
                    }
                    for g in model.bucket_groups
                ]
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "fixed": fixed,
                            "overlap": overlap,
                            "step": step + 1,
                            "microbatches": count,
                            "counts": counts,
                        }
                    ),
                    flush=True,
                )
                try:
                    wrapper.run_pending_grad_sync()
                except AssertionError as error:
                    assert (
                        not fixed and overlap and step == 2 and "Communication call has not been issued" in str(error)
                    ), str(error)
                    print("NATIVE_VARIABLE_MICROBATCH_ASSERTION_REPRODUCED rank=" + str(rank), flush=True)
                    return None
                success, norm, _ = optimizer.step()
                assert success and torch.isfinite(torch.as_tensor(norm)) and norm > 0
                metrics.append([sum(float(row["loss"]) for row in result) / count, float(norm)])
            values = torch.cat([p.detach().flatten() for p in module.parameters()]).clone()
            assert not torch.equal(initial, values)
            if overlap:
                # A second forward_backward before optim_step would accumulate into in-flight buffers.
                wrapper._defer_finalize_model_grads(None)
                with pytest.raises(RuntimeError, match="one forward_backward call"):
                    wrapper._defer_finalize_model_grads(None)
                wrapper._pending_grad_sync = None
            return torch.tensor(metrics), values
        finally:
            torch.distributed.barrier()

    assert run_case(False, True) is None
    control = run_case(True, False)
    fixed = run_case(True, True)
    for got, expected in zip(fixed, control):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    print("VARIABLE_MICROBATCH_ADAM_PARITY_PASSED rank=" + str(rank), flush=True)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.megatron
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_variable_microbatch_distributed_adam():
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
        timeout=360,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("NATIVE_VARIABLE_MICROBATCH_ASSERTION_REPRODUCED") == 2
    assert result.stdout.count("VARIABLE_MICROBATCH_ADAM_PARITY_PASSED") == 2


if __name__ == "__main__":
    _distributed_main()
