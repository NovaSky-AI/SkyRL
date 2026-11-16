"""
The token-balanced DP split (``trainer.balance_dp_tokens``) gives the same update as the default split.

One fixed batch goes through a fresh policy twice: with the default split, and with the
token-balanced split. The two runs must agree on the summed policy loss, the per-sample
log-probs (in batch order), the gradient norm of the optimizer step, and the per-sample log-probs
after that step. Samples have different lengths and different advantages, so a wrong sample order
or a split-dependent loss scale changes these values.

Run with:
uv run --isolated --extra dev --extra fsdp -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_balanced_dp_split.py -m "not megatron"
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_balanced_dp_split.py -m megatron
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
SEQ_LENS = [60, 58, 55, 50, 40, 30, 22, 20, 12, 10, 8, 8, 6, 6, 5, 5]
MAX_TOKENS = 64
NUM_ACTIONS = 4


def _batch() -> TrainingInputBatch:
    torch.manual_seed(42)
    n, max_len = len(SEQ_LENS), max(SEQ_LENS)
    sequences = torch.zeros((n, max_len), dtype=int)
    attention_mask = torch.zeros((n, max_len), dtype=int)
    for i, seq_len in enumerate(SEQ_LENS):
        sequences[i, :seq_len] = torch.randint(1, 100, (seq_len,))
        attention_mask[i, :seq_len] = 1
    # A different advantage per sample, already scaled over the whole batch (as the trainer does).
    advantages = torch.linspace(-1.0, 1.0, n).unsqueeze(1).repeat(1, NUM_ACTIONS) / (n * NUM_ACTIONS)
    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": 0.4 * torch.ones((n, NUM_ACTIONS)),
            "base_action_log_probs": 0.3 * torch.ones((n, NUM_ACTIONS)),
            "values": 0.5 * torch.ones((n, NUM_ACTIONS)),
            "returns": 0.5 * torch.ones((n, NUM_ACTIONS)),
            "advantages": advantages,
            "loss_mask": torch.ones((n, NUM_ACTIONS), dtype=int),
            "response_mask": torch.ones((n, NUM_ACTIONS), dtype=int),
            "rollout_logprobs": 0.2 * torch.ones((n, NUM_ACTIONS)),
        }
    )
    data.metadata = {"response_length": NUM_ACTIONS, "global_step": 0}
    return data


def _config(strategy: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.logger = "console"
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.max_tokens_per_microbatch = MAX_TOKENS
    cfg.trainer.algorithm.use_kl_loss = False
    if strategy == "megatron":
        cfg.trainer.strategy = "megatron"
        cfg.trainer.remove_microbatch_padding = False
        cfg.trainer.placement.policy_num_gpus_per_node = 4  # TP1, PP1 -> DP 4
        cfg.trainer.placement.colocate_all = False
        cfg.trainer.placement.colocate_policy_ref = False
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    else:
        cfg.trainer.strategy = "fsdp2"
        cfg.trainer.placement.policy_num_gpus_per_node = 2  # DP 2
        cfg.generator.inference_engine.tensor_parallel_size = 2
        validate_cfg(cfg)
    return cfg


def _run(strategy: str, balance_max_tokens: int) -> dict:
    cfg = _config(strategy)
    group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )

    def call(method):
        refs = group.async_run_ray_method("mesh", method, data=_batch(), dp_balance_max_tokens=balance_max_tokens)
        out = WorkerOutput.cat(group.actor_infos, ray.get(refs), sample_order=getattr(refs, "sample_order", None))
        return out, getattr(refs, "sample_order", None)

    train_out, order = call("forward_backward")
    grad_norm = ray.get(group.async_run_ray_method("pass_through", "optim_step"))[0]
    after_out, _ = call("forward")
    return {
        "order": order,
        "policy_loss": train_out.metrics["policy_loss"],
        "num_microbatches": train_out.metrics.get("num_microbatches"),
        "num_padding_microbatches": train_out.metrics.get("num_padding_microbatches"),
        "logprobs": loss_fn_outputs_to_tensor(train_out.loss_fn_outputs, key="logprobs").float(),
        "grad_norm": grad_norm,
        "logprobs_after": loss_fn_outputs_to_tensor(after_out.loss_fn_outputs, key="logprobs").float(),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp2", pytest.param("megatron", marks=pytest.mark.megatron)])
async def test_balanced_split_gives_the_same_update(ray_init_fixture, strategy):
    default = _run(strategy, balance_max_tokens=0)
    ray.shutdown()
    ray_init_for_tests()
    balanced = _run(strategy, balance_max_tokens=MAX_TOKENS)

    assert default["order"] is None
    assert balanced["order"] is not None and balanced["order"] != sorted(balanced["order"])
    assert balanced["num_padding_microbatches"] == 0

    step_change = (default["logprobs_after"] - default["logprobs"]).abs().mean().item()
    report = {
        "policy_loss": (default["policy_loss"], balanced["policy_loss"]),
        "grad_norm": (default["grad_norm"], balanced["grad_norm"]),
        "num_microbatches": (default["num_microbatches"], balanced["num_microbatches"]),
        "num_padding_microbatches": (default["num_padding_microbatches"], balanced["num_padding_microbatches"]),
        "logprobs max diff": (default["logprobs"] - balanced["logprobs"]).abs().max().item(),
        "logprobs_after max diff": (default["logprobs_after"] - balanced["logprobs_after"]).abs().max().item(),
        "mean |logprobs_after - logprobs| (size of the step)": step_change,
    }
    for key, value in report.items():
        print(f"  {strategy} {key}: {value}")

    assert report["logprobs max diff"] < 2e-3
    assert abs(default["policy_loss"] - balanced["policy_loss"]) < 1e-4 + 1e-2 * abs(default["policy_loss"])
    assert default["grad_norm"] is not None and balanced["grad_norm"] is not None
    assert abs(default["grad_norm"] - balanced["grad_norm"]) < 2e-2 * abs(default["grad_norm"])
    assert report["logprobs_after max diff"] < 5e-3
