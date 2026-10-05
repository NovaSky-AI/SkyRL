"""Trajectory-level GSPO on step-wise rows matches sequence-level GSPO on the merged conversation (FSDP).

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_trajectory_gspo.py -v
"""

import math

import pytest
import ray
import torch
from ray.util.placement_group import placement_group

from skyrl.backends.skyrl_train.distributed.dispatch import loss_fn_outputs_to_tensor
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "Qwen/Qwen3-0.6B"
EPS_CLIP = 0.28

# One append-only conversation per trajectory: (prompt, [(response, observation), ...]).
CONVERSATIONS = [
    (
        [101, 102, 103, 104, 105, 106],
        [([201, 202, 203, 204], [301, 302, 303]), ([211, 212, 213, 214], [311, 312]), ([221, 222, 223, 224], [])],
    ),
    ([151, 152, 153, 154, 155], [([251, 252, 253, 254], [])]),
]
ADVANTAGES = [1.0, -0.5]
# log(pi_theta / pi_old) on each turn's response tokens. Turn 0 of trajectory 0 clips on its own,
# but trajectory 0 as a whole does not.
TURN_LOG_RATIOS = [[0.5, -0.4, 0.0], [0.1]]


def _expected_loss(groups):
    """Summed GSPO loss over ``(num_tokens, log_weight, advantage)`` groups of tokens sharing one weight."""
    total = 0.0
    for num_tokens, log_weight, advantage in groups:
        ratio = math.exp(log_weight)
        clipped = min(max(ratio, 1 - EPS_CLIP), 1 + EPS_CLIP)
        total -= num_tokens * min(ratio * advantage, clipped * advantage)
    return total


def _turn_groups():
    """``(num_tokens, log_ratio, advantage)`` for each turn's response, in row order."""
    return [
        (len(response), log_ratio, advantage)
        for (_, turns), advantage, log_ratios in zip(CONVERSATIONS, ADVANTAGES, TURN_LOG_RATIOS)
        for (response, _), log_ratio in zip(turns, log_ratios)
    ]


def _trajectory_groups():
    """``(num_tokens, log_weight, advantage)`` for each trajectory, its log weight averaged over all its turns."""
    groups = []
    for (_, turns), advantage, log_ratios in zip(CONVERSATIONS, ADVANTAGES, TURN_LOG_RATIOS):
        num_tokens = sum(len(response) for response, _ in turns)
        log_ratio_sum = sum(len(response) * log_ratio for (response, _), log_ratio in zip(turns, log_ratios))
        groups.append((num_tokens, log_ratio_sum / num_tokens, advantage))
    return groups


def _step_rows():
    """One row per turn: ``(prompt, response, loss_mask, log_ratio, advantage)``."""
    rows = []
    for (prompt, turns), advantage, log_ratios in zip(CONVERSATIONS, ADVANTAGES, TURN_LOG_RATIOS):
        context = list(prompt)
        for (response, observation), log_ratio in zip(turns, log_ratios):
            rows.append((list(context), response, [1] * len(response), [log_ratio] * len(response), advantage))
            context += response + observation
    return rows


def _merged_rows():
    """One row per trajectory, with observation tokens masked out of the loss."""
    rows = []
    for (prompt, turns), advantage, log_ratios in zip(CONVERSATIONS, ADVANTAGES, TURN_LOG_RATIOS):
        response, loss_mask, row_log_ratios = [], [], []
        for (turn_response, observation), log_ratio in zip(turns, log_ratios):
            response += turn_response + observation
            loss_mask += [1] * len(turn_response) + [0] * len(observation)
            row_log_ratios += [log_ratio] * len(turn_response) + [0.0] * len(observation)
        rows.append((list(prompt), response, loss_mask, row_log_ratios, advantage))
    return rows


def _right_aligned(values, width):
    out = torch.zeros(len(values), width)
    for i, row in enumerate(values):
        out[i, width - len(row) :] = torch.tensor(row, dtype=torch.float32)
    return out


def _batch(rows, trajectory_index=None) -> tuple[TrainingInputBatch, torch.Tensor]:
    """Returns the batch (without old log-probs) and the target log-ratios, both in the trainer's layout."""
    prompts, responses, loss_masks, log_ratios, advantages = zip(*rows)
    sequences, attention_mask, response_mask, _, loss_mask, _, _, _ = convert_prompts_responses_to_batch_tensors(
        0,
        list(prompts),
        list(responses),
        [[0.0] * len(response) for response in responses],
        list(loss_masks),
    )
    num_actions = response_mask.shape[1]
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "loss_mask": loss_mask.float(),
            "advantages": torch.tensor(advantages).unsqueeze(-1) * loss_mask.float(),
        }
    )
    if trajectory_index is not None:
        batch["trajectory_index"] = torch.tensor(trajectory_index)
    batch.metadata = {"response_length": num_actions}
    return batch, _right_aligned(log_ratios, num_actions)


def _policy_cfg(gspo_ratio_level: str, num_gpus: int) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.logger = "console"
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.policy_loss_type = "gspo"
    cfg.trainer.algorithm.loss_reduction = "sequence_mean"
    cfg.trainer.algorithm.gspo_ratio_level = gspo_ratio_level
    cfg.trainer.algorithm.eps_clip_low = EPS_CLIP
    cfg.trainer.algorithm.eps_clip_high = EPS_CLIP
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.temperature = 1.0
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.step_wise_trajectories = True
    validate_cfg(cfg)
    return cfg


def _forward_backward_metrics(cfg: SkyRLTrainConfig, batches) -> list:
    """Runs ``forward_backward`` on each ``(batch, log_ratios)``, with old log-probs set to give those log-ratios.

    No optimizer step runs, so every batch sees the initial parameters.
    """
    num_gpus = cfg.trainer.placement.policy_num_gpus_per_node
    raw_pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
    actor_group = None
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=raw_pg,
            colocate_all=False,
            num_gpus_per_node=num_gpus,
            cfg=cfg,
            num_gpus_per_actor=1.0,
        )
        dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
        metrics = []
        for batch, log_ratios in batches:
            current_log_probs = loss_fn_outputs_to_tensor(dispatch.forward("policy", batch).loss_fn_outputs)
            batch["action_log_probs"] = current_log_probs - log_ratios
            metrics.append(dispatch.forward_backward("policy", batch).metrics)
        return metrics
    finally:
        if actor_group is not None:
            for actor_info in actor_group.actor_infos:
                ray.kill(actor_info.handle, no_restart=True)
        ray.util.remove_placement_group(raw_pg)


@pytest.mark.parametrize("num_gpus", [1, 2], ids=["dp1", "dp2"])
def test_trajectory_gspo_matches_sequence_gspo_on_merged_rows(ray_init_fixture, num_gpus):
    """With two DP ranks, trajectory 0's turns are split across ranks (rows [0, 1] and [2, 3])."""
    (trajectory_metrics,) = _forward_backward_metrics(
        _policy_cfg("trajectory", num_gpus), [_batch(_step_rows(), trajectory_index=[0, 0, 0, 1])]
    )
    merged_metrics, per_turn_metrics = _forward_backward_metrics(
        _policy_cfg("sequence", num_gpus), [_batch(_merged_rows()), _batch(_step_rows())]
    )

    expected = _expected_loss(_trajectory_groups())
    per_turn_expected = _expected_loss(_turn_groups())
    assert per_turn_expected != pytest.approx(expected, rel=1e-2)

    assert trajectory_metrics["policy_loss"] == pytest.approx(expected, rel=2e-3)
    assert merged_metrics["policy_loss"] == pytest.approx(expected, rel=2e-3)
    assert per_turn_metrics["policy_loss"] == pytest.approx(per_turn_expected, rel=2e-3)

    # One row per microbatch, so the metric is the mean over turns of |trajectory - turn| log weight.
    turn_trajectory_log_weights = [
        log_weight for (_, log_weight, _), turns in zip(_trajectory_groups(), TURN_LOG_RATIOS) for _ in turns
    ]
    abs_diffs = [abs(w - log_ratio) for w, (_, log_ratio, _) in zip(turn_trajectory_log_weights, _turn_groups())]
    assert trajectory_metrics["loss_metrics/gspo_traj_step_log_weight_abs_diff"] == pytest.approx(
        sum(abs_diffs) / len(abs_diffs), abs=1e-2
    )
    assert "loss_metrics/gspo_traj_step_log_weight_abs_diff" not in per_turn_metrics
