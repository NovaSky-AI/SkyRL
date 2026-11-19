"""
The token-balanced DP split (``trainer.balance_dp_tokens``) gives the same update as the default split.

One fixed batch goes through a fresh policy twice: with the default split, and with the
token-balanced split. Each run is a test of its own (a fresh Ray cluster from the fixture), and a
third test compares them. The two runs must agree on the per-sample log-probs (in batch order), the
gradient norm of the optimizer step, and the per-sample log-probs after that step. Samples have
different lengths and different advantages, so a wrong sample order or a split-dependent loss
scale changes these values.

The old log-probs come from a first forward pass with the split of the run, as in the trainer. The
PPO ratio therefore starts at 1 in both runs, and the gradient is not clipped to zero.

Tolerances. The two splits give a rank different samples, so the ranks form different
micro-batches. On FSDP the forward is bit-exact across groupings. On Megatron (BF16) a different
grouping alone moves a per-token log-prob by up to ~0.13, also with the default split on a
reordered batch. A swapped sample would differ by whole nats (random tokens: |log-prob| ~7).
Adam's first step is about ``lr * sign(grad)``, so it amplifies summation-order noise: the
log-probs after the step differ by up to ~0.1 between two default-split runs with different caps.
The after-step bounds are therefore loose, and they only catch a gross error.

Run with:
uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_balanced_dp_split.py -m "not megatron"
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_balanced_dp_split.py -m megatron
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
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "Qwen/Qwen3-0.6B"
SEQ_LENS = [60, 58, 55, 50, 40, 30, 22, 20, 12, 10, 8, 8, 6, 6, 5, 5]
MAX_TOKENS = 64
NUM_ACTIONS = 4


# Per backend: log-prob max / mean difference, relative gradient-norm difference, and
# after-step log-prob max / mean difference.
TOLERANCES = {
    "fsdp2": {"logprobs_max": 1e-3, "logprobs_mean": 1e-4, "grad_norm": 1e-3, "after_max": 0.3, "after_mean": 0.1},
    "megatron": {"logprobs_max": 0.3, "logprobs_mean": 0.03, "grad_norm": 1e-2, "after_max": 0.3, "after_mean": 0.1},
}


def _batch(old_logprobs=None) -> TrainingInputBatch:
    torch.manual_seed(42)
    n, max_len = len(SEQ_LENS), max(SEQ_LENS)
    sequences = torch.zeros((n, max_len), dtype=int)
    attention_mask = torch.zeros((n, max_len), dtype=int)
    # Left-padded, so the last NUM_ACTIONS positions (the response) are real tokens in every sample.
    for i, seq_len in enumerate(SEQ_LENS):
        sequences[i, max_len - seq_len :] = torch.randint(1, 100, (seq_len,))
        attention_mask[i, max_len - seq_len :] = 1
    # A different advantage per sample, already scaled over the whole batch (as the trainer does).
    advantages = torch.linspace(-1.0, 1.0, n).unsqueeze(1).repeat(1, NUM_ACTIONS) / (n * NUM_ACTIONS)
    # Old and rollout log-probs equal to the policy's own (ratio 1), so PPO clipping does not zero the gradient.
    old = old_logprobs if old_logprobs is not None else 0.4 * torch.ones((n, NUM_ACTIONS))
    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": old.clone(),
            "base_action_log_probs": 0.3 * torch.ones((n, NUM_ACTIONS)),
            "values": 0.5 * torch.ones((n, NUM_ACTIONS)),
            "returns": 0.5 * torch.ones((n, NUM_ACTIONS)),
            "advantages": advantages,
            "loss_mask": torch.ones((n, NUM_ACTIONS), dtype=int),
            "response_mask": torch.ones((n, NUM_ACTIONS), dtype=int),
            "rollout_logprobs": old.clone(),
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
        cfg.trainer.remove_microbatch_padding = True  # packed layout, as the recipes run
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


def _run(strategy: str, balance_tokens: bool) -> dict:
    cfg = _config(strategy)
    group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )

    def call(method, old_logprobs=None, balance=balance_tokens):
        refs = group.async_run_ray_method("mesh", method, data=_batch(old_logprobs), dp_balance_tokens=balance)
        out = WorkerOutput.cat(group.actor_infos, ray.get(refs), sample_order=getattr(refs, "sample_order", None))
        return out, getattr(refs, "sample_order", None)

    # Each arm computes the old log-probs with its own split, as the trainer does. The PPO ratio is
    # then exactly 1 in both arms. With old log-probs from the other split, the BF16 regrouping noise
    # of Megatron moves the ratio away from 1, and the gradient norm then differs by ~2% (also
    # between two default-split runs on reordered batches).
    before_out, _ = call("forward")
    old_logprobs = loss_fn_outputs_to_tensor(before_out.loss_fn_outputs, key="logprobs").float()
    train_out, order = call("forward_backward", old_logprobs)
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


STRATEGIES = ["fsdp2", pytest.param("megatron", marks=pytest.mark.megatron)]
# Results of the two arms, by (strategy, arm). Each arm is a test of its own, so each one gets a
# fresh Ray cluster and fresh GPUs from ``ray_init_fixture``. The comparison test reads both.
_RESULTS: dict = {}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_default_split_arm(ray_init_fixture, strategy):
    result = _run(strategy, balance_tokens=False)
    assert result["order"] is None
    _RESULTS[(strategy, "default")] = result


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_balanced_split_arm(ray_init_fixture, strategy):
    result = _run(strategy, balance_tokens=True)
    assert result["order"] is not None and result["order"] != sorted(result["order"])
    _RESULTS[(strategy, "balanced")] = result


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_balanced_split_gives_the_same_update(strategy):
    if (strategy, "default") not in _RESULTS or (strategy, "balanced") not in _RESULTS:
        pytest.fail("this test needs the two arm tests of the same strategy to run first and pass")
    default, balanced = _RESULTS[(strategy, "default")], _RESULTS[(strategy, "balanced")]

    logprobs_diff = (default["logprobs"] - balanced["logprobs"]).abs()
    after_diff = (default["logprobs_after"] - balanced["logprobs_after"]).abs()
    step_change = (default["logprobs_after"] - default["logprobs"]).abs().mean().item()
    grad_norm_diff = abs(default["grad_norm"] - balanced["grad_norm"]) / abs(default["grad_norm"])
    report = {
        "policy_loss": (default["policy_loss"], balanced["policy_loss"]),
        "grad_norm": (default["grad_norm"], balanced["grad_norm"]),
        "grad_norm relative diff": grad_norm_diff,
        "num_microbatches": (default["num_microbatches"], balanced["num_microbatches"]),
        "num_padding_microbatches": (default["num_padding_microbatches"], balanced["num_padding_microbatches"]),
        "logprobs max diff": logprobs_diff.max().item(),
        "logprobs mean diff": logprobs_diff.mean().item(),
        "logprobs mean |value|": default["logprobs"].abs().mean().item(),
        "logprobs_after max diff": after_diff.max().item(),
        "logprobs_after mean diff": after_diff.mean().item(),
        "mean |logprobs_after - logprobs| (size of the step)": step_change,
    }
    for key, value in report.items():
        print(f"  {strategy} {key}: {value}")

    tol = TOLERANCES[strategy]
    assert default["grad_norm"] > 0 and step_change > 0, "the batch must produce an update"
    assert report["logprobs max diff"] < tol["logprobs_max"]
    assert report["logprobs mean diff"] < tol["logprobs_mean"]
    assert grad_norm_diff < tol["grad_norm"]
    assert report["logprobs_after max diff"] < tol["after_max"]
    assert report["logprobs_after mean diff"] < tol["after_mean"]
