"""Mixed-dtype Adam checkpoint identity and continuation on two CUDA ranks."""

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    importlib.import_module("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")

import pytest
import torch


def _distributed_main(checkpoint_root):
    from datetime import timedelta

    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.serialization import (
        get_default_load_sharded_strategy,
        get_default_save_sharded_strategy,
    )
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
        FullyParallelSaveStrategyWrapper,
    )
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

    from skyrl.backends.skyrl_train.patches.megatron.patch_optimizer_group_order import (
        patch_optimizer_group_order,
    )
    from skyrl.backends.skyrl_train.workers.megatron.param_sync import (
        sync_params_for_export,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3))
    parallel_state.initialize_model_parallel()
    rank = torch.distributed.get_rank()
    patch_optimizer_group_order()

    class Model(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            shapes = [(32, 32)] * 2 if kind == "equal" else [(32, 32), (24, 16), (16, 32)]
            dtypes = (
                [torch.bfloat16, torch.float32] if kind == "equal" else [torch.bfloat16, torch.float32, torch.bfloat16]
            )
            if kind in ("bf16", "fp32"):
                dtypes = [torch.bfloat16 if kind == "bf16" else torch.float32] * 3
            for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
                self.register_parameter(
                    f"weight_{i}", torch.nn.Parameter(torch.full(shape, 0.25 + 0.125 * i, dtype=dtype, device="cuda"))
                )

        def forward(self, scale):
            return sum((i + 1) * p.float().sum() for i, p in enumerate(self.parameters())) * scale

    def build(kind, precision_aware):
        module = Model(kind)
        for param in module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            bucket_size=4096,
            grad_reduce_in_fp32=False,
        )
        layout = DistributedOptimizer.compute_full_param_layout(list(module.parameters()), 4096, 2, ddp_config)
        model = DistributedDataParallel(
            TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=32),
            ddp_config,
            module,
            full_param_layout=layout,
        )
        kwargs = {
            "lr": 0.01,
            "weight_decay": 0.0,
            "clip_grad": 0.0,
            "bf16": True,
            "params_dtype": torch.bfloat16,
            "use_distributed_optimizer": True,
            "overlap_param_gather": True,
            "use_precision_aware_optimizer": precision_aware,
        }
        if precision_aware:
            kwargs.update(
                exp_avg_dtype=torch.bfloat16,
                exp_avg_sq_dtype=torch.bfloat16,
                main_params_dtype=torch.float32,
                store_param_remainders=True,
            )
        optimizer = get_megatron_optimizer(OptimizerConfig(**kwargs), [model])
        return model, optimizer

    def step(model, optimizer, scale):
        optimizer.zero_grad()
        model(torch.tensor(scale, device="cuda")).backward()
        model.finish_grad_sync()
        assert optimizer.step()[0]
        optimizer.zero_grad()
        model.zero_grad_buffer()
        sync_params_for_export([model], optimizer)

    def model_state(model):
        return {
            name: ShardedTensor.from_rank_offsets(name, p, replica_id=rank)
            for name, p in model.module.named_parameters()
        }

    def state(model, optimizer, loading=False):
        tensors = model_state(model)
        return {
            "model": tensors,
            "optimizer": optimizer.sharded_state_dict(
                tensors,
                is_loading=loading,
                metadata={"distrib_optim_sharding_type": "dp_reshardable"},
            ),
        }

    def snapshot(model, optimizer):
        names = {p: name for name, p in model.module.named_parameters()}
        result = {"weights": {name: p.detach().clone() for name, p in model.module.named_parameters()}}
        for child_index, child in enumerate(getattr(optimizer, "chained_optimizers", [optimizer])):
            for param in child.model_param_group_index_map:
                states = child._get_main_param_and_optimizer_states(param)
                # Prove this is the correct parameter, not merely a matching shape.
                interval = child._get_model_param_range_map(param)["param"]
                expected_shard = param.detach().flatten()[interval.start : interval.end]
                if child.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                    group, index = child.model_param_group_index_map[param]
                    shard = child.optimizer.param_groups[group]["params"][index]
                    assert shard.data_ptr() == expected_shard.data_ptr()
                    assert shard.shape == expected_shard.shape and shard.dtype == expected_shard.dtype
                    # TE may encode the master as int16 BF16 remainders. Compare
                    # those exact state bits across reload, not as float values.
                else:
                    torch.testing.assert_close(states["param"].to(param.dtype), expected_shard, rtol=0, atol=0)
                for key, tensor in states.items():
                    result[f"{child_index}/{names[param]}/{key}"] = tensor.detach().clone()
        return result

    def compare(actual, expected):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], dict):
                compare(actual[key], expected[key])
            else:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)

    for kind in ("mixed", "equal", "bf16", "fp32"):
        for precision_aware in (False, True):
            path = Path(checkpoint_root) / f"{kind}-{precision_aware}"
            if rank == 0:
                path.mkdir()
            torch.distributed.barrier()
            model, optimizer = build(kind, precision_aware)
            step(model, optimizer, 1.0)
            saved = snapshot(model, optimizer)
            save_strategy = FullyParallelSaveStrategyWrapper(
                get_default_save_sharded_strategy("torch_dist"),
                parallel_state.get_data_parallel_group(),
            )
            dist_checkpointing.save(state(model, optimizer), str(path), sharded_strategy=save_strategy)
            step(model, optimizer, -0.5)
            expected = snapshot(model, optimizer)
            model.disable_forward_pre_hook()
            del model, optimizer
            restored, resumed_optimizer = build(kind, precision_aware)
            load_strategy = FullyParallelLoadStrategyWrapper(
                get_default_load_sharded_strategy(str(path)),
                parallel_state.get_data_parallel_group(),
            )
            loaded = dist_checkpointing.load(
                state(restored, resumed_optimizer, loading=True), str(path), sharded_strategy=load_strategy
            )
            restored.module.load_state_dict(loaded["model"])
            resumed_optimizer.load_state_dict(loaded["optimizer"])
            compare(snapshot(restored, resumed_optimizer), saved)
            step(restored, resumed_optimizer, -0.5)
            compare(snapshot(restored, resumed_optimizer), expected)
            restored.disable_forward_pre_hook()
            torch.distributed.barrier()
            print(f"OPTIMIZER_DCP_ROUNDTRIP_PASS rank={rank} kind={kind} precision_aware={precision_aware}", flush=True)
            del restored, resumed_optimizer
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.megatron
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_mixed_dtype_optimizer_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(Path(__file__).resolve()),
                directory,
            ],
            env={**os.environ, "NVTE_FLASH_ATTN": "0"},
            capture_output=True,
            text=True,
            timeout=720,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("OPTIMIZER_DCP_ROUNDTRIP_PASS") == 16


if __name__ == "__main__":
    _distributed_main(sys.argv[1])
