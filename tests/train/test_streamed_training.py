"""Streamed fully-async training: planning a mini-batch of stored rollouts from their summaries.

uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_streamed_training.py
"""

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from loguru import logger

from skyrl.backends.skyrl_train.rollout_shard import build_training_batch
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    LossReduction,
)
from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import (
    RowCollation,
    compute_prompt_boundaries,
    rows_to_training_input,
)
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _check_dumps,
    _row_prompts,
    streaming_bypassed_overrides,
)
from skyrl.train.generators.base import (
    RolloutMetadataView,
    StoredRolloutGroup,
    TrajectoryID,
)
from skyrl.train.generators.utils import concatenate_generator_outputs
from skyrl.train.streamed_training import unsupported_streaming_settings
from skyrl.train.trainer import RayPPOTrainer, advantage_metrics
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.async_utils import BackgroundFailure

REDUCTIONS = [
    LossReduction.TOKEN_MEAN,
    LossReduction.SEQUENCE_MEAN,
    LossReduction.SEQ_MEAN_TOKEN_SUM_NORM,
    LossReduction.PROMPT_MEAN,
]


def test_the_default_config_lists_what_streaming_needs():
    cfg = SkyRLTrainConfig()
    assert "token-based microbatching (trainer.max_tokens_per_microbatch > 0)" in unsupported_streaming_settings(
        cfg, has_critic=False
    )
    cfg.trainer.max_tokens_per_microbatch = 4096
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.generator.step_wise_trajectories = True
    assert unsupported_streaming_settings(cfg, has_critic=False) == []
    assert unsupported_streaming_settings(cfg, has_critic=True) == ["no critic"]
    # Plain policy gradient sums per-row terms, so it streams (score centering's other anchor).
    cfg.trainer.algorithm.policy_loss_type = "reinforce"
    assert unsupported_streaming_settings(cfg, has_critic=False) == []
    cfg.generator.vision_language_generator = True
    assert unsupported_streaming_settings(cfg, has_critic=False) == ["no vision-language generator"]
    cfg.generator.vision_language_generator = False
    # The MTP draft loss scores every attended token, including those of trajectories a stream leaves out.
    cfg.trainer.policy.megatron_config.mtp_num_layers = 1
    assert unsupported_streaming_settings(cfg, has_critic=False) == [
        "no MTP draft heads (policy.megatron_config.mtp_num_layers)"
    ]


def _output(instance, rewards, rows_per_trajectory, loss_tokens=None):
    """One group: trajectory ``j`` has ``rows_per_trajectory[j]`` step-wise rows and reward ``rewards[j]``."""
    prompts, responses, masks, row_rewards, last, ids = [], [], [], [], [], []
    for j, (reward, rows) in enumerate(zip(rewards, rows_per_trajectory)):
        for row in range(rows):
            length = 2 + (j + row) % 3
            trained = length if loss_tokens is None else loss_tokens[j]
            # A distinct first token, so every row's tokens identify it.
            prompts.append(np.asarray([1000 * ord(instance) + 10 * j + row, *range(1, 1 + row)], dtype=np.int32))
            responses.append(np.arange(10, 10 + length, dtype=np.int32))
            masks.append(np.asarray([1] * trained + [0] * (length - trained), dtype=np.int8))
            row_rewards.append(reward)
            last.append(row == rows - 1)
            ids.append(TrajectoryID(instance, j))
    return {
        "prompt_token_ids": prompts,
        "response_ids": responses,
        "loss_masks": masks,
        "rollout_logprobs": [np.full(len(r), -0.5, dtype=np.float32) for r in responses],
        "rewards": row_rewards,
        "is_last_step": last,
        "trajectory_ids": ids,
        "stop_reasons": ["complete"] * len(responses),
        "rollout_metrics": {"generate/custom_max": 1.0},
    }


def _groups(outputs):
    return [
        GeneratedOutputGroup(
            generator_output=StoredRolloutGroup.from_output(o),
            uid=o["trajectory_ids"][0].instance_id,
            global_step_when_scheduled=0,
        )
        for o in outputs
    ]


def _cfg(loss_reduction=LossReduction.TOKEN_MEAN, **algorithm):
    cfg = SkyRLTrainConfig()
    cfg.generator.step_wise_trajectories = True
    cfg.trainer.max_tokens_per_microbatch = 4096
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.loss_reduction = loss_reduction
    cfg.trainer.algorithm.max_seq_len = 64
    for key, value in algorithm.items():
        setattr(cfg.trainer.algorithm, key, value)
    return cfg


def _trainer(cfg, dp_size):
    trainer = SimpleNamespace(
        cfg=cfg,
        tokenizer=SimpleNamespace(pad_token_id=0),
        dispatch=SimpleNamespace(dp_size=lambda model: dp_size),
        all_metrics={},
    )
    trainer._row_collation = lambda: RayPPOTrainer._row_collation(trainer)
    trainer._skip_policy_forward = lambda has: RayPPOTrainer._skip_policy_forward(trainer, has)
    return trainer


def _in_memory_advantages(cfg, outputs, trained_rows, normalization=None):
    """The unstreamed trainer's token advantages for the trained rows, from its own methods.

    Without ``normalization``, the batch counts its own, as the unstreamed trainer does.
    """
    rows = {
        key: [v for o in outputs for v in o[key]]
        for key in ("prompt_token_ids", "response_ids", "loss_masks", "rewards")
    }
    batch = rows_to_training_input(
        {k: rows[k] for k in ("prompt_token_ids", "response_ids", "loss_masks")},
        RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=False),
    )
    rewards = torch.zeros_like(batch["loss_mask"])
    rewards[:, -1] = torch.tensor(rows["rewards"])
    batch["rewards"] = rewards
    batch["values"] = None
    batch.metadata = {
        "uids": [o["trajectory_ids"][0].instance_id for o in outputs for _ in o["rewards"]],
        "is_last_step": [v for o in outputs for v in o["is_last_step"]],
        "avg_response_length": 0.0,
    }
    if normalization is None:
        normalization = MinibatchLossNormalization.from_batch(batch)
    stand_in = SimpleNamespace(cfg=cfg, all_metrics={})
    batch = RayPPOTrainer.compute_advantages_and_returns(stand_in, batch)
    selected = torch.tensor(trained_rows)
    sub = type(batch)({key: batch[key][selected] for key in ("sequences", "advantages", "loss_mask", "response_mask")})
    sub.metadata = {}
    uids = [batch.metadata["uids"][i] for i in trained_rows]
    out = RayPPOTrainer._normalize_advantages(
        stand_in, sub, [(0, len(trained_rows))], [normalization], compute_prompt_boundaries(uids)
    )
    return out["advantages"], sub["response_mask"]


def _assert_rows_train_the_unstreamed_advantages(plan, cfg, outputs, normalization=None) -> int:
    """Each row a shard trains carries the advantage the unstreamed trainer gives it; returns how many."""
    by_row = {}
    for batch in [build_training_batch(shard) for shard in plan.shards]:
        for row in range(batch.batch_size):
            key = tuple(batch["sequences"][row][batch["attention_mask"][row].bool()].tolist())
            by_row[key] = batch["advantages"][row][batch["response_mask"][row].bool()]
    prompts = [p for o in outputs for p in o["prompt_token_ids"]]
    responses = [r for o in outputs for r in o["response_ids"]]
    expected, response_mask = _in_memory_advantages(cfg, outputs, list(range(len(responses))), normalization)
    compared = 0
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        key = tuple(prompt.tolist() + response.tolist())
        if key in by_row:
            torch.testing.assert_close(by_row[key], expected[i][response_mask[i].bool()])
            compared += 1
    assert compared == len(by_row)
    return compared


@pytest.mark.parametrize("loss_reduction", REDUCTIONS)
@pytest.mark.parametrize("batch_normalize", [False, True])
def test_a_plan_trains_each_row_with_the_unstreamed_advantage(loss_reduction, batch_normalize):
    outputs = [
        _output("a", [1.0, 0.0, 1.0], [2, 1, 1]),
        _output("b", [0.5, 0.25], [1, 3]),
    ]
    cfg = _cfg(loss_reduction, advantage_batch_normalize=batch_normalize)
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=2), _groups(outputs))

    batches = [build_training_batch(shard) for shard in plan.shards]
    assert not any(shard.padding for shard in plan.shards)
    assert sum(b.batch_size for b in batches) == 8
    # The step's denominators are those of all its rows, wherever they train.
    shard_counts = [MinibatchLossNormalization.from_batch(b) for b in batches]
    assert plan.normalization == MinibatchLossNormalization(
        num_sequences=sum(n.num_sequences for n in shard_counts),
        num_loss_tokens=sum(n.num_loss_tokens for n in shard_counts),
        num_model_tokens=sum(n.num_model_tokens for n in shard_counts),
        num_segments=sum(n.num_segments for n in shard_counts),
        min_segment_length=min(n.min_segment_length for n in shard_counts),
    )
    # Rows trained on any rank carry the advantage the unstreamed trainer gives them.
    assert _assert_rows_train_the_unstreamed_advantages(plan, cfg, outputs, plan.normalization) == 8


@pytest.mark.parametrize("loss_reduction", REDUCTIONS)
def test_rows_that_train_nothing_change_no_advantage_streamed_or_not(loss_reduction):
    outputs = [
        # Trajectory 1 trains nothing (a failed rollout, say).
        _output("a", [1.0, 0.0, 0.5], [2, 1, 1], loss_tokens=[2, 0, 1]),
        # A group that trains nothing at all, as if loss-masked by the zero-variance filter.
        _output("b", [0.5, 0.25], [1, 1], loss_tokens=[0, 0]),
        _output("c", [0.0, 1.0], [1, 2]),
    ]
    cfg = _cfg(loss_reduction)
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=2), _groups(outputs))

    # The unstreamed trainer collates every row and counts its own denominators; only the six rows
    # with loss tokens train.
    assert _assert_rows_train_the_unstreamed_advantages(plan, cfg, outputs) == 6


@pytest.mark.parametrize("loss_reduction", REDUCTIONS)
def test_a_step_without_loss_tokens_in_a_trained_trajectory_counts_as_the_unstreamed_batch_counts_it(
    loss_reduction,
):
    outputs = [_output("a", [1.0, 0.0], [3, 1]), _output("b", [0.5, 0.25], [2, 2])]
    # The middle step of trajectory a/0 trains nothing (a turn its harness masked, say).
    outputs[0]["loss_masks"][1] = np.zeros_like(outputs[0]["loss_masks"][1])
    cfg = _cfg(loss_reduction)
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=2), _groups(outputs))

    # Every trajectory trains, so the unstreamed batch is the streamed rows; a row without loss
    # tokens is not a trained sequence in either.
    rows = {k: [v for o in outputs for v in o[k]] for k in ("prompt_token_ids", "response_ids", "loss_masks")}
    collation = RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=False)
    assert plan.normalization == MinibatchLossNormalization.from_batch(rows_to_training_input(rows, collation))
    assert plan.normalization.num_sequences == 7
    assert _assert_rows_train_the_unstreamed_advantages(plan, cfg, outputs) == 8


def test_untrainable_trajectories_and_filtered_groups_do_not_train_or_count():
    outputs = [
        # Trajectory 1 has no loss tokens.
        _output("a", [1.0, 0.0], [1, 1], loss_tokens=[2, 0]),
        # Zero variance: filtered.
        _output("b", [1.0, 1.0], [1, 1]),
    ]
    cfg = _cfg(zero_variance_filter=True)
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=2), _groups(outputs))

    assert plan.normalization.num_sequences == 1
    assert plan.metrics["reward/num_zero_variance_filtered"] == 1
    # One trajectory trains; the other rank pads, so it still runs the step's collectives.
    assert [shard.padding for shard in plan.shards] == [False, True]
    assert plan.metrics["async/num_padding_dp_shards"] == 1
    padding = build_training_batch(plan.shards[1])
    assert not padding.real_sample_mask.any()
    assert plan.compute_old_logprobs is False  # rollout_is reads the rollout logprobs


def test_a_mini_batch_of_groups_storing_different_fields_fails_before_any_rank_trains():
    with_logprobs = _output("a", [1.0, 0.0], [1, 1])
    without = {key: value for key, value in _output("b", [1.0, 0.0], [1, 1]).items() if key != "rollout_logprobs"}
    with pytest.raises(ValueError, match="different payload keys"):
        FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(
            _trainer(_cfg(), dp_size=1), _groups([with_logprobs, without])
        )


def test_a_shard_carries_each_rows_lengths_for_its_workers_to_plan_from():
    outputs = [_output("a", [1.0, 0.0], [2, 1]), _output("b", [0.5, 0.25], [1, 3])]
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(_cfg(), dp_size=2), _groups(outputs))

    for shard in plan.shards:
        batch = build_training_batch(shard)
        lengths = batch["attention_mask"].sum(-1).tolist()
        assert [p + r for p, r in zip(shard.prompt_lengths, shard.response_lengths)] == lengths
        assert sum(shard.payload_rows) == batch.batch_size


def test_a_rank_pads_with_a_trained_trajectory_not_a_failed_placeholder():
    # Trajectory 0 is the smallest but trains nothing (a failed attempt's placeholder, say).
    outputs = [_output("a", [1.0, 0.0], [1, 1], loss_tokens=[0, 3])]
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(_cfg(), dp_size=2), _groups(outputs))

    assert [shard.padding for shard in plan.shards] == [False, True]
    assert plan.shards[1].payload_refs == plan.shards[0].payload_refs


def _with_routes(output):
    routes = [np.zeros((len(p) + len(r), 1, 2), dtype=np.uint8) for p, r in _rows(output)]
    return {**output, "rollout_expert_indices": routes}


def _rows(output):
    return list(zip(output["prompt_token_ids"], output["response_ids"]))


def test_with_r3_a_trained_trajectory_without_routes_fails_to_collate():
    cfg = _cfg()
    cfg.generator.inference_engine.enable_return_routed_experts = True
    trainer = _trainer(cfg, dp_size=1)
    routed = _with_routes(_output("a", [1.0, 0.0], [1, 1]))
    # A group that trains nothing is left out of the plan, so it needs no routes.
    untrained = _output("b", [1.0, 0.0], [1, 1], loss_tokens=[0, 0])
    (shard,) = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(trainer, _groups([routed, untrained])).shards
    assert build_training_batch(shard)["rollout_expert_indices"] is not None

    (shard,) = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(
        trainer, _groups([_output("b", [1.0, 0.0], [1, 1])])
    ).shards
    with pytest.raises(ValueError, match="carry no rollout_expert_indices"):
        build_training_batch(shard)


def _with_support(output, scored):
    supports = []
    for response in output["response_ids"]:
        n = len(response)
        logprobs = np.full(2 * n, -0.7, dtype=np.float32) if scored else None
        supports.append(
            CompactSampleSupport.from_token_ids(
                np.arange(2 * n, dtype=np.int32), np.full(n, 2), top_k=4, logprobs=logprobs
            )
        )
    return {**output, "rollout_sample_support": supports}


def _built_shard(cfg, output):
    (shard,) = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=1), _groups([output])).shards
    return build_training_batch(shard)


def test_with_score_centering_unscored_support_fails_to_collate():
    cfg = _cfg(enable_sample_support_replay=True, enable_score_centering=True)
    _built_shard(cfg, _with_support(_output("a", [1.0, 0.0], [1, 1]), scored=True))

    with pytest.raises(ValueError, match="no sampler member logprobs"):
        _built_shard(cfg, _with_support(_output("b", [1.0, 0.0], [1, 1]), scored=False))


def test_with_sample_support_replay_a_loss_token_without_support_fails_to_collate():
    cfg = _cfg(enable_sample_support_replay=True)
    _built_shard(cfg, _with_support(_output("a", [1.0, 0.0], [1, 1]), scored=False))

    partial = _with_support(_output("b", [1.0, 0.0], [1, 1]), scored=False)
    rows = partial["rollout_sample_support"][0]
    # The first response token's support row is empty.
    partial["rollout_sample_support"][0] = CompactSampleSupport.from_token_ids(
        np.arange(2 * (rows.num_rows - 1), dtype=np.int32), np.asarray([0] + [2] * (rows.num_rows - 1)), top_k=4
    )
    with pytest.raises(ValueError, match="loss tokens without captured sample support"):
        _built_shard(cfg, partial)


def test_token_level_rewards_plan_as_their_sums():
    outputs = [_output("a", [1.0, 0.0, 0.5], [1, 2, 1]), _output("b", [0.5, 0.25], [1, 1])]
    token_level = [dict(o) for o in outputs]
    for o in token_level:
        o["rewards"] = [[0.0] * (len(r) - 1) + [reward] for r, reward in zip(o["response_ids"], o["rewards"])]
    trainer = _trainer(_cfg(), dp_size=2)

    expected = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(trainer, _groups(outputs))
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(trainer, _groups(token_level))

    assert plan.normalization == expected.normalization
    assert [s.row_advantages for s in plan.shards] == [s.row_advantages for s in expected.shards]


def test_per_row_advantage_metrics_equal_the_per_token_ones():
    rewards = torch.tensor([1.0, 0.0, 0.5])
    advantages = torch.tensor([0.7, -1.3, 0.2])
    token_counts = torch.tensor([3.0, 1.0, 0.0])

    weighted = advantage_metrics(rewards, advantages, token_counts=token_counts)
    per_token = advantage_metrics(rewards, advantages.repeat_interleave(token_counts.long()))

    assert weighted == pytest.approx(per_token)


@pytest.mark.parametrize("stored", [False, True])
def test_both_paths_log_the_concatenated_outputs_metrics(stored):
    outputs = [_output("a", [1.0, 0.0], [1, 2]), _output("b", [0.5, 0.5], [1, 1])]
    groups = (
        _groups(outputs)
        if stored
        else [
            GeneratedOutputGroup(
                generator_output=o, uid=o["trajectory_ids"][0].instance_id, global_step_when_scheduled=0
            )
            for o in outputs
        ]
    )
    trainer = SimpleNamespace(
        cfg=_cfg(),
        global_step=1,
        max_staleness_steps=1,
        all_metrics={},
        _reprefix_metrics=FullyAsyncRayPPOTrainer._reprefix_metrics,
    )
    trainer._staleness_metrics = lambda gs: FullyAsyncRayPPOTrainer._staleness_metrics(trainer, gs)

    FullyAsyncRayPPOTrainer._log_group_metrics(trainer, groups[:1], groups[1:])

    def concatenated(selected):
        return concatenate_generator_outputs([_as_lists(o) for o in selected], step_wise=True)["rollout_metrics"]

    for prefix_metrics in (
        concatenated(outputs),
        FullyAsyncRayPPOTrainer._reprefix_metrics(concatenated(outputs[:1]), "kept"),
        FullyAsyncRayPPOTrainer._reprefix_metrics(concatenated(outputs[1:]), "dropped"),
    ):
        for key, value in prefix_metrics.items():
            assert trainer.all_metrics[key] == pytest.approx(value), key
    assert trainer.all_metrics["reward_dropped/avg_raw_reward"] == pytest.approx(0.5)


def test_each_row_is_logged_with_its_own_trajectorys_prompt():
    group = _groups([_output("a", [1.0, 0.0], [1, 2])])[0]
    group.prompts = ["first", "second"]
    assert _row_prompts([group]) == ["first", "second", "second"]


def test_rows_without_trajectory_ids_are_logged_with_their_prompts():
    # SkyRLGymGenerator outputs no trajectory ids unless step-wise: one row per trajectory.
    output = {"rewards": [1.0, 0.0], "trajectory_ids": None}
    group = GeneratedOutputGroup(generator_output=output, uid="a", global_step_when_scheduled=0, prompts=["p0", "p1"])
    assert _row_prompts([group]) == ["p0", "p1"]


def test_a_streamed_steps_metrics_come_from_the_metadata():
    outputs = [_output("a", [1.0, 0.0], [1, 2])]
    outputs[0]["rollout_stats"] = [{"turns": 1}, {"turns": 2}, {"turns": 3}]
    groups = _groups(outputs)
    seen = {}

    def log_rollout_metrics(view, uids, **kwargs):
        seen["view"], seen["uids"] = view, uids

    trainer = SimpleNamespace(
        cfg=_cfg(),
        global_step=1,
        max_staleness_steps=1,
        all_metrics={},
        log_rollout_metrics=log_rollout_metrics,
        _reprefix_metrics=FullyAsyncRayPPOTrainer._reprefix_metrics,
    )
    trainer._staleness_metrics = lambda gs: FullyAsyncRayPPOTrainer._staleness_metrics(trainer, gs)
    trainer._log_group_metrics = lambda gs, dropped: FullyAsyncRayPPOTrainer._log_group_metrics(trainer, gs, dropped)
    rows = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(_cfg(), dp_size=1), groups).rows
    FullyAsyncRayPPOTrainer._log_streamed_rollout_metrics(trainer, groups, [], rows)

    assert isinstance(seen["view"], RolloutMetadataView) and seen["uids"] == rows.uids == ["a"] * 3
    assert seen["view"]["rollout_stats"] == outputs[0]["rollout_stats"]
    assert trainer.all_metrics["generate/avg_num_tokens"] == pytest.approx(
        np.mean([len(r) for r in outputs[0]["response_ids"]])
    )
    assert trainer.all_metrics["generate/custom_max"] == 1.0


def test_overrides_a_streamed_step_skips_are_refused_unless_they_only_observe():
    class Reshapes(FullyAsyncRayPPOTrainer):
        def compute_advantages_and_returns(self, data):
            return data

    class Observes(FullyAsyncRayPPOTrainer):
        STREAMED_HOOKS = ("convert_to_training_input",)

        def convert_to_training_input(self, generator_output, uids):
            return generator_output

    class Both(Observes, Reshapes):
        pass

    class ShapesRewards(FullyAsyncRayPPOTrainer):
        # DAPO's overlong punishment, say: a streamed step plans from the stored rewards.
        def postprocess_generator_output(self, generator_output, uids, *args, **kwargs):
            return generator_output, uids

    class LogsMetrics(FullyAsyncRayPPOTrainer):
        def log_rollout_metrics(self, generator_output, uids, *args, **kwargs):
            pass

    assert streaming_bypassed_overrides(FullyAsyncRayPPOTrainer) == []
    assert streaming_bypassed_overrides(Observes) == []
    assert streaming_bypassed_overrides(LogsMetrics) == []
    assert [o.rsplit(".", 2)[-2:] for o in streaming_bypassed_overrides(Both)] == [
        ["Reshapes", "compute_advantages_and_returns"]
    ]
    assert [o.rsplit(".", 2)[-2:] for o in streaming_bypassed_overrides(ShapesRewards)] == [
        ["ShapesRewards", "postprocess_generator_output"]
    ]


def test_a_streamed_steps_logging_reads_only_the_logged_rows_beside_the_training():
    outputs = [_output("a", [1.0, 0.0], [1, 2]), _output("b", [0.5, 0.25], [1, 1])]
    groups = _groups(outputs)
    for group, output in zip(groups, outputs):
        # One per trajectory; each step-wise row is logged with its trajectory's.
        group.prompts = [
            f"prompt {group.uid}{j}" for j in range(len({t.repetition_id for t in output["trajectory_ids"]}))
        ]
    cfg = _cfg()
    cfg.trainer.num_logger_train_samples = 2
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=1), groups)
    view = plan.rows.view
    logged, observed, threads = {}, [], []

    class Logger:
        def select_sample_indices(self, *, num_samples, rewards, total):
            assert (num_samples, total) == (2, 5)
            return [1, 4]

        def log(self, **kwargs):
            threads.append(threading.current_thread())
            logged.update(kwargs)

    trainer = SimpleNamespace(
        cfg=cfg,
        trajectory_logger=Logger(),
        tracker=None,
        global_step=3,
        tokenizer=SimpleNamespace(decode=str),
        _observer=ThreadPoolExecutor(max_workers=1),
        log_streamed_rollouts=observed.append,
    )
    trainer._logged_rows = lambda rewards, n: FullyAsyncRayPPOTrainer._logged_rows(trainer, rewards, n)
    trainer._log_train_examples = lambda *args: FullyAsyncRayPPOTrainer._log_train_examples(trainer, *args)
    FullyAsyncRayPPOTrainer._observe_streamed_step(trainer, groups, plan.rows).result()

    responses = [r for o in outputs for r in o["response_ids"]]
    masks = [m for o in outputs for m in o["loss_masks"]]
    assert logged["generator_output"]["response_ids"] == [responses[1].tolist(), responses[4].tolist()]
    assert logged["generator_output"]["loss_masks"] == [masks[1].tolist(), masks[4].tolist()]
    assert logged["generator_output"]["rewards"] == [view["rewards"][1], view["rewards"][4]]
    assert logged["prompts"] == ["prompt a1", "prompt b1"] and logged["global_step"] == 3
    assert threads[0] is not threading.current_thread()
    (rows,) = observed
    assert rows.uids == ["a"] * 3 + ["b"] * 2 and len(rows.advantages) == 5
    assert rows.view["rewards"] == [r for o in outputs for r in o["rewards"]]
    assert rows.response_lengths == [len(r) for r in responses]
    assert rows.loss_tokens == rows.trained_loss_tokens == [int(m.sum()) for m in masks]


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, object_store_memory=200 * 1024**2)
    yield


@pytest.mark.asyncio
async def test_collection_propagates_a_background_generation_failure():
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.mini_batch_size = 1
    trainer.sample_full_batch = False
    trainer._phase_gauge = SimpleNamespace(timed_phase=lambda *args: nullcontext())
    buffer, done, failure = asyncio.Queue(), asyncio.Event(), BackgroundFailure()
    pending = asyncio.create_task(trainer._collect_generation_mini_batch(buffer, done, failure))
    await asyncio.sleep(0)
    failure.record(RuntimeError("generator failed while collecting"), "generator")
    with pytest.raises(RuntimeError, match="generator failed while collecting"):
        await asyncio.wait_for(pending, timeout=1)


def _with_mixed_support(output):
    """Support rows of 4 (full), 2 and 0 (uncaptured) members across each response."""
    supports = []
    for response in output["response_ids"]:
        lengths = np.asarray([(4, 2, 0)[t % 3] for t in range(len(response))])
        supports.append(
            CompactSampleSupport.from_token_ids(np.arange(int(lengths.sum()), dtype=np.int32), lengths, top_k=4)
        )
    return {**output, "rollout_sample_support": supports}


def _as_lists(output):
    return {
        k: [x.tolist() if isinstance(x, np.ndarray) else x for x in v] if isinstance(v, list) else v
        for k, v in output.items()
    }


def test_a_plan_logs_and_filters_as_the_unstreamed_trainer_does():
    outputs = [
        # Equal final rewards, but the step-wise rows' rewards vary: the unstreamed filter keeps it.
        _output("a", [1.0, 1.0], [2, 1]),
        # Every row's reward is equal: filtered.
        _output("b", [0.5, 0.5], [1, 2]),
        _output("c", [0.0, 1.0], [1, 1], loss_tokens=[2, 0]),
    ]
    outputs[0]["rewards"][0] = 0.0
    outputs = [_with_mixed_support(o) for o in outputs]
    cfg = _cfg(zero_variance_filter=True)
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 3
    cfg.generator.n_samples_per_prompt = 2
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(_trainer(cfg, dp_size=2), _groups(outputs))

    # The unstreamed step: postprocess (the zero-variance filter), then collate.
    unstreamed = _trainer(cfg, dp_size=2)
    unstreamed.dispatch.get_lcm_dp_size = lambda: 2
    unstreamed.log_rollout_metrics = lambda *args, **kwargs: None
    output = concatenate_generator_outputs([_as_lists(o) for o in outputs], step_wise=True)
    uids = [o["trajectory_ids"][0].instance_id for o in outputs for _ in o["rewards"]]
    output, uids = RayPPOTrainer.postprocess_generator_output(unstreamed, output, uids)
    batch = RayPPOTrainer.convert_to_training_input(unstreamed, output, uids)

    for key in (
        "reward/num_zero_variance_filtered",
        "generate/batch_num_seq",
        "generate/batch_padded_seq_len",
        "generate/sample_support_size_mean",
        "generate/sample_support_full_fraction",
    ):
        assert plan.metrics[key] == pytest.approx(unstreamed.all_metrics[key]), key
    assert plan.metrics["reward/num_zero_variance_filtered"] == 1
    # Each row trains the loss tokens the unstreamed batch's loss mask keeps.
    num_rows = len(uids)
    assert plan.rows.trained_loss_tokens == batch["loss_mask"][:num_rows].sum(-1).int().tolist()
    assert plan.rows.trained_loss_tokens == [2, 3, 3, 0, 0, 0, 2, 0]


def test_a_plan_keeps_the_advantages_before_z_scoring_for_its_observers():
    outputs = [_output("a", [1.0, 0.0, 1.0], [2, 1, 1]), _output("b", [0.5, 0.25], [1, 3])]
    plans = [
        FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(
            _trainer(_cfg(advantage_batch_normalize=normalize), dp_size=2), _groups(outputs)
        )
        for normalize in (False, True)
    ]

    # The z-score changes what trains, not what observers (and the batch dump) see.
    assert plans[0].rows.advantages == plans[1].rows.advantages
    assert plans[0].unscaled_advantages == plans[1].unscaled_advantages
    assert [s.row_advantages for s in plans[0].shards] != [s.row_advantages for s in plans[1].shards]
    # As ``compute_advantages_and_returns`` gives them: GRPO (not divided by the group's std), before
    # normalization.
    assert plans[0].rows.advantages[:4] == pytest.approx([1 / 3, 1 / 3, -2 / 3, 1 / 3], abs=1e-4)
    by_rows = {row for shard in plans[0].unscaled_advantages for row in shard}
    assert by_rows <= set(plans[0].rows.advantages)


def _training_stand_in(cfg, plan, worker_metrics):
    groups = _groups([_output("a", [1.0, 0.0], [1, 1])])
    trainer = SimpleNamespace(
        cfg=cfg,
        all_metrics={},
        all_timings={},
        dispatch=SimpleNamespace(
            forward_backward_rollout_shards=lambda *args, **kwargs: SimpleNamespace(metrics=dict(worker_metrics)),
            optim_step=lambda model: 2.0,
            empty_cache=lambda: None,
        ),
        _plan_streamed_mini_batch=lambda groups: plan,
        _log_streamed_rollout_metrics=lambda groups, dropped, rows: None,
    )
    done = Future()
    done.set_result(None)
    trainer._observe_streamed_step = lambda groups, rows: done
    return trainer, groups


@pytest.mark.parametrize("compute_old_logprobs", [False, True])
def test_a_streamed_step_times_its_parts_and_logs_the_forward_pass_diff(compute_old_logprobs):
    from dataclasses import replace

    cfg = _cfg()
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(
        _trainer(cfg, dp_size=1), _groups([_output("a", [1.0, 0.0], [1, 1])])
    )
    plan = replace(plan, compute_old_logprobs=compute_old_logprobs)
    diff = {
        "minibatch_rollout_logprobs_abs_diff_mean": 0.1,
        "minibatch_rollout_logprobs_abs_diff_sq_mean": 0.02,
        "minibatch_rollout_logprobs_abs_diff_max": 0.5,
        "minibatch_rollout_logprobs_abs_diff_min": 0.0,
    }
    trainer, groups = _training_stand_in(cfg, plan, {"policy_loss": 1.0, **diff})

    status, rows = FullyAsyncRayPPOTrainer._train_streamed_step(trainer, groups, [])

    assert {
        "plan_streamed_mini_batch",
        "policy_train",
        "policy_forward_backward",
        "policy_optim_step",
    } <= trainer.all_timings.keys()
    assert all(t.payload is None for group in groups for t in group.generator_output.trajectories)
    assert rows is plan.rows and status["grad_norm"] == 2.0
    aliases = {
        key: trainer.all_metrics[key]
        for key in trainer.all_metrics
        if key.startswith("policy/rollout_train_logprobs_abs_diff_")
    }
    if not compute_old_logprobs:
        # As the unstreamed trainer logs it only when it runs the old-logprob forward pass.
        assert aliases == {}
        return
    assert aliases == pytest.approx(
        {
            "policy/rollout_train_logprobs_abs_diff_mean": 0.1,
            "policy/rollout_train_logprobs_abs_diff_max": 0.5,
            "policy/rollout_train_logprobs_abs_diff_min": 0.0,
            "policy/rollout_train_logprobs_abs_diff_std": trainer.all_metrics[
                "policy/minibatch_rollout_logprobs_abs_diff_std"
            ],
        }
    )


def test_a_failed_batch_dump_warns_without_failing_the_step():
    @ray.remote
    def fails():
        raise OSError("disk full")

    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        _check_dumps([fails.remote(), ray.put(None)])
    finally:
        logger.remove(sink)
    assert len(messages) == 1 and "disk full" in messages[0]


def test_only_the_fully_async_trainer_streams_stored_groups():
    cfg = _cfg()
    cfg.trainer.fully_async.store_groups_in_ray = True
    with pytest.raises(ValueError, match="needs the fully-async trainer, not RayPPOTrainer"):
        RayPPOTrainer(cfg, None, None, None, None, None)
    with pytest.raises(ValueError, match="requires trainer.fully_async.enabled"):
        validate_cfg(cfg)
