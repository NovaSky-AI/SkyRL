"""CPU regression tests for policy worker loss scaling."""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytest import approx

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig


class _NoopDeviceMesh:
    def get_group(self, name):
        return None


def _make_policy_worker(cfg, rank=0, dp_size=1, strategy=None, model=None, scheduler=None):
    worker = PolicyWorkerBase(
        cfg=cfg.trainer,
        world_size=dp_size,
        rank=rank,
        local_rank=rank,
        master_addr="localhost",
        master_port=12345 + rank,
        sequence_parallel_size=1,
    )
    worker.strategy = strategy if strategy is not None else MagicMock()
    worker.device_mesh = _NoopDeviceMesh()
    worker.mesh_rank = SimpleNamespace(dp_size=dp_size)
    worker.model = model if model is not None else MagicMock()
    worker.optimizer = None
    worker.scheduler = scheduler if scheduler is not None else MagicMock(get_last_lr=MagicMock(return_value=[0.0]))
    return worker


def _make_loss_scaling_batch(loss_values, loss_fn, dp_size=1):
    loss_values = torch.tensor(loss_values, dtype=torch.float32)
    batch_size = loss_values.numel()
    sequences = torch.zeros((batch_size, 2), dtype=torch.long)
    sequences[:, 0] = torch.arange(batch_size)

    if loss_fn == "cross_entropy":
        logprobs_by_sample = -loss_values.view(batch_size, 1)
        old_action_log_probs = torch.zeros((batch_size, 1), dtype=torch.float32)
        advantages = torch.zeros((batch_size, 1), dtype=torch.float32)
        total_nonpad_tokens = batch_size * dp_size
        loss_mask_scale = 1 / total_nonpad_tokens
    elif loss_fn == "dual_clip":
        logprobs_by_sample = torch.zeros((batch_size, 1), dtype=torch.float32)
        old_action_log_probs = torch.zeros((batch_size, 1), dtype=torch.float32)
        advantages = -loss_values.view(batch_size, 1)
        loss_mask_scale = 1.0
    else:
        raise ValueError(f"Unsupported loss_fn {loss_fn}")

    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
            "action_log_probs": old_action_log_probs,
            "base_action_log_probs": torch.zeros((batch_size, 1), dtype=torch.float32),
            "values": torch.zeros((batch_size, 1), dtype=torch.float32),
            "returns": torch.zeros((batch_size, 1), dtype=torch.float32),
            "advantages": advantages,
            "loss_mask": torch.full((batch_size, 1), loss_mask_scale, dtype=torch.float32),
            "response_mask": torch.ones((batch_size, 1), dtype=torch.float32),
            "rollout_logprobs": None,
        }
    )
    batch.metadata = {"response_length": 1}
    return batch, logprobs_by_sample.tolist()


def _make_loss_scaling_cfg(loss_fn, micro_batch_size):
    cfg = SkyRLTrainConfig()
    cfg.trainer.micro_train_batch_size_per_gpu = micro_batch_size
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.algorithm.policy_loss_type = loss_fn
    cfg.trainer.algorithm.temperature = 1.0
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.generator.sampling_params.temperature = 1.0
    return cfg


def _patch_worker_cuda_for_cpu():
    stack = ExitStack()
    stack.enter_context(patch("torch.cuda.current_device", return_value=torch.device("cpu")))
    stack.enter_context(patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()))
    return stack


def _all_reduce_payload(strategy, op, key):
    for args, kwargs in strategy.all_reduce.call_args_list:
        metrics = args[0]
        call_op = kwargs.get("op")
        if call_op == op and key in metrics:
            return metrics
    raise AssertionError(f"No all_reduce call found for op={op!r}, key={key!r}")


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "dual_clip"])
def test_policy_forward_backward_loss_scaling_with_mocked_ranks(loss_fn):
    """FSDP forward_backward scales local microbatch losses before DP reduction."""
    dp_size = 2
    micro_batch_size = 2
    rank_loss_values = {
        0: [2.0, 4.0, 6.0, 8.0],
        1: [10.0, 12.0, 14.0, 16.0],
    }

    if loss_fn == "cross_entropy":
        global_num_tokens = sum(len(v) for v in rank_loss_values.values())
        expected_backward_by_rank = {
            rank: [
                sum(values[i : i + micro_batch_size]) * dp_size / global_num_tokens
                for i in range(0, len(values), micro_batch_size)
            ]
            for rank, values in rank_loss_values.items()
        }
        expected_local_metric = {rank: sum(values) / global_num_tokens for rank, values in rank_loss_values.items()}
        expected_global_metric = sum(expected_local_metric.values())
    else:
        expected_backward_by_rank = {
            rank: [sum(values[i : i + micro_batch_size]) * dp_size for i in range(0, len(values), micro_batch_size)]
            for rank, values in rank_loss_values.items()
        }
        expected_local_metric = {rank: sum(values) for rank, values in rank_loss_values.items()}
        expected_global_metric = sum(expected_local_metric.values())

    assert expected_local_metric[0] != expected_local_metric[1]

    for rank, loss_values in rank_loss_values.items():
        cfg = _make_loss_scaling_cfg(loss_fn, micro_batch_size)
        batch, logprobs_by_sample = _make_loss_scaling_batch(loss_values, loss_fn=loss_fn, dp_size=dp_size)

        backward_losses = []
        strategy = MagicMock()

        def mock_backward(loss, model, optimizer):
            backward_losses.append(float(loss.detach().cpu().item()))

        def mock_all_reduce(metrics, op, group=None):
            if op == "sum":
                return {
                    k: expected_global_metric if k in {"loss", "policy_loss", "final_loss"} else v
                    for k, v in metrics.items()
                }
            return dict(metrics)

        strategy.backward.side_effect = mock_backward
        strategy.all_reduce.side_effect = mock_all_reduce

        def model_forward(sequences, num_actions, **kwargs):
            sample_ids = sequences[:, 0].long().tolist()
            action_log_probs = torch.tensor(
                [logprobs_by_sample[i] for i in sample_ids],
                dtype=torch.float32,
                requires_grad=True,
            )
            entropy = torch.zeros((len(sample_ids), num_actions + 1), dtype=torch.float32)
            return action_log_probs, {"entropy": entropy}

        model = MagicMock()
        model.side_effect = model_forward
        worker = _make_policy_worker(cfg, rank=rank, dp_size=dp_size, strategy=strategy, model=model)

        with _patch_worker_cuda_for_cpu():
            result = worker.forward_backward(batch, loss_fn=loss_fn)

        assert backward_losses == approx(expected_backward_by_rank[rank])
        assert model.call_count == 2

        if loss_fn == "cross_entropy":
            sum_payload = _all_reduce_payload(strategy, "sum", "loss")
            assert sum_payload["loss"] == approx(expected_local_metric[rank])
            assert result.metrics["loss"] == approx(expected_global_metric)
        else:
            sum_payload = _all_reduce_payload(strategy, "sum", "policy_loss")
            assert sum_payload["policy_loss"] == approx(expected_local_metric[rank])
            assert sum_payload["final_loss"] == approx(expected_local_metric[rank])
            assert result.metrics["policy_loss"] == approx(expected_global_metric)
            assert result.metrics["final_loss"] == approx(expected_global_metric)
