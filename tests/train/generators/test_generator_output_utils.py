"""
uv run --extra dev --isolated pytest tests/train/generators/test_generator_output_utils.py
"""

from unittest.mock import patch

import numpy as np
import pytest

from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    merge_stepwise_output,
)
from skyrl.train.utils.utils import validate_cfg
from tests.train.util import example_dummy_config


def test_generator_output_concatenation():
    # First ensure that the GeneratorOutput fields are what we expect
    expected_fields = [
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
        "rollout_logprobs",
        "rollout_expert_indices",
        # optional but present in the signature
        "trajectory_ids",
        "is_last_step",
        "pixel_values",
        "image_grid_thw",
    ]
    assert set(GeneratorOutput.__annotations__.keys()) == set(expected_fields), (
        "GeneratorOutput fields are not what we expect. "
        "Please update the test and `concatenate_generator_outputs()` to reflect the new fields."
        "It is needed to help Trainer.eval() record the full GeneratorOutput information."
    )

    generator_output_1: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.1, 0.2], [0.3, 0.4]],
    }

    generator_output_2: GeneratorOutput = {
        "prompt_token_ids": [[5, 6, 7], [8]],
        "response_ids": [[5, 6, 7], [8]],
        "rewards": [2.0, 3.0],
        "loss_masks": [[1, 1, 1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.5, 0.6, 0.7], [0.8]],
    }

    generator_outputs = [generator_output_1, generator_output_2]
    concatenated_output = concatenate_generator_outputs(generator_outputs)

    assert concatenated_output["prompt_token_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["response_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["rewards"] == [1.0, 2.0, 2.0, 3.0]
    assert concatenated_output["loss_masks"] == [[1, 1], [1, 1], [1, 1, 1], [1]]
    assert concatenated_output["stop_reasons"] == ["stop", "stop", "stop", "stop"]
    assert concatenated_output["rollout_logprobs"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7], [0.8]]

    # Validate rollout metrics
    expected_rollout_metrics = {
        "generate/min_num_tokens": 1,
        "generate/max_num_tokens": 3,
        "generate/avg_num_tokens": 2.0,
        "generate/std_num_tokens": np.std([2, 2, 3, 1]).item(),
        "generate/avg_tokens_non_zero_rewards": 2.0,
        "generate/avg_tokens_zero_rewards": 0,
    }
    assert concatenated_output["rollout_metrics"].keys() == expected_rollout_metrics.keys()
    for key, value in expected_rollout_metrics.items():
        np.testing.assert_allclose(concatenated_output["rollout_metrics"][key], value)


def test_get_metrics_from_generator_output():
    # Per trajectory rewards, where rewards are List[float]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 1.5
    assert metrics["pass_at_n"] == 1.0
    assert metrics["mean_positive_reward"] == 1.5

    # Per token rewards, where rewards are List[List[float]], so for pass_at_n we use the last
    # token's reward to signify the trajectory's reward
    generator_output["rewards"] = [[1.0, 0.0], [0.0, 1.0]]
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 1.0
    assert metrics["pass_at_n"] == 0.5
    assert metrics["mean_positive_reward"] == 1.0

    # Mixed rewards with some negative rewards
    generator_output["rewards"] = [-1.0, 2.0]
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 0.5
    assert metrics["pass_at_n"] == 0.5
    assert metrics["mean_positive_reward"] == 1.0

    # Mixed per-token rewards with negatives - per-token rewards
    generator_output["rewards"] = [[1.0, -1.0], [-0.5, 0.5]]
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 0.0
    assert metrics["pass_at_n"] == 0.5
    assert metrics["mean_positive_reward"] == 0.75


# ───────────────────────────────────────────────────────────────────
# merge_stepwise_output (prefix-aware merging) tests
# ───────────────────────────────────────────────────────────────────


def _make_tid(instance_id: str, rep: int = 0) -> TrajectoryID:
    return TrajectoryID(instance_id=instance_id, repetition_id=rep)


class TestMergeStepwiseOutput:
    """CPU-only tests for ``merge_stepwise_output`` (prefix-aware merging)."""

    # ─── Core merging cases ────────────────────────────────────────

    def test_case1_response_only_assistant(self):
        """Case 1: response only contains assistant-generated tokens.

        Turn 1: prompt=[O1], response=[A1]
        Turn 2: prompt=[O1, A1, O2], response=[A2]

        Merged: prompt=[O1], response=[A1, O2, A2]
        obs_delta = [O2]
        """
        tid = _make_tid("traj_1")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[10], [10, 20, 30]],
            "response_ids": [[20], [40, 41]],
            "rewards": [[1.0], [0.0, 5.0]],
            "loss_masks": [[1], [1, 1]],
            "stop_reasons": ["continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": [[-0.5], [-0.3, -0.4]],
            "trajectory_ids": [tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, True],
        }

        merged = merge_stepwise_output(gen_out)

        assert len(merged["response_ids"]) == 1
        assert merged["prompt_token_ids"] == [[10]]
        # response = A1 + obs_delta(O2) + A2(two tokens)
        assert merged["response_ids"] == [[20, 30, 40, 41]]
        # loss_mask: A1=1, O2=0, A2_tok1=1, A2_tok2=1
        assert merged["loss_masks"] == [[1, 0, 1, 1]]
        # logprobs: A1=-0.5, O2=0.0, A2_tok1=-0.3, A2_tok2=-0.4
        assert merged["rollout_logprobs"] == [[-0.5, 0.0, -0.3, -0.4]]
        # rewards: A1=1.0, O2=0.0, A2_tok1=0.0, A2_tok2=5.0
        assert merged["rewards"] == [[1.0, 0.0, 0.0, 5.0]]
        assert merged["stop_reasons"] == ["eos"]
        assert merged["trajectory_ids"] == [tid]
        assert merged["is_last_step"] == [True]

    def test_case2_response_contains_observation(self):
        """Case 2: response contains both assistant and observation tokens.

        Turn 1: prompt=[O1], response=[A1, O2]
        Turn 2: prompt=[O1, A1, O2], response=[A2]

        Merged: prompt=[O1], response=[A1, O2, A2]
        obs_delta = [] (empty, since prompt+response of turn 1 == prompt of turn 2)
        """
        tid = _make_tid("traj_2")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[10], [10, 20, 30]],
            "response_ids": [[20, 30], [40]],
            "rewards": [[0.0, 0.0], [7.0]],
            "loss_masks": [[1, 0], [1]],
            "stop_reasons": ["continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": [[-0.1, -0.2], [-0.3]],
            "trajectory_ids": [tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, True],
        }

        merged = merge_stepwise_output(gen_out)

        assert len(merged["response_ids"]) == 1
        assert merged["prompt_token_ids"] == [[10]]
        # response = [A1, O2] + [] + [A2] = [A1, O2, A2]
        assert merged["response_ids"] == [[20, 30, 40]]
        assert merged["loss_masks"] == [[1, 0, 1]]
        assert merged["rollout_logprobs"] == [[-0.1, -0.2, -0.3]]
        assert merged["rewards"] == [[0.0, 0.0, 7.0]]
        assert merged["stop_reasons"] == ["eos"]
        assert merged["is_last_step"] == [True]

    def test_case3_combination(self):
        """Case 3: response has obs tokens AND there's extra obs_delta in prompt.

        Turn 1: prompt=[O1], response=[A1, O2]
        Turn 2: prompt=[O1, A1, O2, O2_5], response=[A2]

        Merged: prompt=[O1], response=[A1, O2, O2_5, A2]
        obs_delta = [O2_5]
        """
        tid = _make_tid("traj_3")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[10], [10, 20, 30, 35]],
            "response_ids": [[20, 30], [40]],
            "rewards": [[0.0, 0.0], [9.0]],
            "loss_masks": [[1, 0], [1]],
            "stop_reasons": ["continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": [[-0.1, -0.2], [-0.5]],
            "trajectory_ids": [tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, True],
        }

        merged = merge_stepwise_output(gen_out)

        assert len(merged["response_ids"]) == 1
        assert merged["prompt_token_ids"] == [[10]]
        # response = [A1, O2] + obs_delta[O2_5] + [A2]
        assert merged["response_ids"] == [[20, 30, 35, 40]]
        assert merged["loss_masks"] == [[1, 0, 0, 1]]
        assert merged["rollout_logprobs"] == [[-0.1, -0.2, 0.0, -0.5]]
        assert merged["rewards"] == [[0.0, 0.0, 0.0, 9.0]]
        assert merged["is_last_step"] == [True]

    # ─── Multi-turn and multi-trajectory ───────────────────────────

    def test_three_turns(self):
        """Three-turn trajectory merging (Case 1 pattern repeated).

        Turn 1: prompt=[1], response=[2]
        Turn 2: prompt=[1,2,3], response=[4]
        Turn 3: prompt=[1,2,3,4,5], response=[6]

        Merged: prompt=[1], response=[2, 3, 4, 5, 6]
        """
        tid = _make_tid("traj_multi")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[1], [1, 2, 3], [1, 2, 3, 4, 5]],
            "response_ids": [[2], [4], [6]],
            "rewards": [[0.0], [0.0], [10.0]],
            "loss_masks": [[1], [1], [1]],
            "stop_reasons": ["continue", "continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": [[-1.0], [-2.0], [-3.0]],
            "trajectory_ids": [tid, tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, False, True],
        }

        merged = merge_stepwise_output(gen_out)

        assert len(merged["response_ids"]) == 1
        assert merged["prompt_token_ids"] == [[1]]
        # resp[0]=2, obs_delta=3, resp[1]=4, obs_delta=5, resp[2]=6
        assert merged["response_ids"] == [[2, 3, 4, 5, 6]]
        assert merged["loss_masks"] == [[1, 0, 1, 0, 1]]
        assert merged["rollout_logprobs"] == [[-1.0, 0.0, -2.0, 0.0, -3.0]]
        assert merged["rewards"] == [[0.0, 0.0, 0.0, 0.0, 10.0]]
        assert merged["is_last_step"] == [True]
        assert merged["stop_reasons"] == ["eos"]

    def test_multiple_trajectories(self):
        """Two separate trajectories in the same batch, each with 2 turns."""
        tid_a = _make_tid("A")
        tid_b = _make_tid("B")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [
                [10],  # A turn 1
                [10, 20, 30],  # A turn 2
                [100],  # B turn 1
                [100, 200, 300],  # B turn 2
            ],
            "response_ids": [
                [20],  # A turn 1
                [40],  # A turn 2
                [200],  # B turn 1
                [400],  # B turn 2
            ],
            "rewards": [[0.0], [1.0], [0.0], [2.0]],
            "loss_masks": [[1], [1], [1], [1]],
            "stop_reasons": ["continue", "eos", "continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": [tid_a, tid_a, tid_b, tid_b],
            "rollout_expert_indices": None,
            "is_last_step": [False, True, False, True],
        }

        merged = merge_stepwise_output(gen_out)

        assert len(merged["response_ids"]) == 2
        # Trajectory A merged
        assert merged["prompt_token_ids"][0] == [10]
        assert merged["response_ids"][0] == [20, 30, 40]
        assert merged["loss_masks"][0] == [1, 0, 1]
        # Trajectory B merged
        assert merged["prompt_token_ids"][1] == [100]
        assert merged["response_ids"][1] == [200, 300, 400]
        assert merged["loss_masks"][1] == [1, 0, 1]
        assert merged["is_last_step"] == [True, True]
        assert merged["stop_reasons"] == ["eos", "eos"]

    def test_mixed_trajectories_and_turns(self):
        """Mix of single-turn and multi-turn trajectories in one batch."""
        tid_a = _make_tid("A")
        tid_b = _make_tid("B")  # single turn
        tid_c = _make_tid("C")

        gen_out: GeneratorOutput = {
            "prompt_token_ids": [
                [1],  # A turn 1
                [1, 2, 3],  # A turn 2
                [1, 2, 3, 4, 5],  # A turn 3
                [50],  # B single turn
                [60],  # C turn 1
                [60, 70, 80],  # C turn 2
            ],
            "response_ids": [
                [2],  # A
                [4],  # A
                [6],  # A
                [51],  # B
                [70],  # C
                [90],  # C
            ],
            "rewards": [[0.0], [0.0], [10.0], [3.0], [0.0], [7.0]],
            "loss_masks": [[1], [1], [1], [1], [1], [1]],
            "stop_reasons": ["c", "c", "eos", "eos", "c", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": [[-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0]],
            "trajectory_ids": [tid_a, tid_a, tid_a, tid_b, tid_c, tid_c],
            "rollout_expert_indices": None,
            "is_last_step": [False, False, True, True, False, True],
        }

        merged = merge_stepwise_output(gen_out)

        # 3 entries: A merged, B single, C merged
        assert len(merged["response_ids"]) == 3

        # A: prompt=[1], response=[2,3,4,5,6]
        assert merged["prompt_token_ids"][0] == [1]
        assert merged["response_ids"][0] == [2, 3, 4, 5, 6]
        assert merged["loss_masks"][0] == [1, 0, 1, 0, 1]
        assert merged["rollout_logprobs"][0] == [-1.0, 0.0, -2.0, 0.0, -3.0]
        assert merged["rewards"][0] == [0.0, 0.0, 0.0, 0.0, 10.0]

        # B: unchanged
        assert merged["prompt_token_ids"][1] == [50]
        assert merged["response_ids"][1] == [51]

        # C: prompt=[60], response=[70,80,90]
        assert merged["prompt_token_ids"][2] == [60]
        assert merged["response_ids"][2] == [70, 80, 90]
        assert merged["loss_masks"][2] == [1, 0, 1]

        assert merged["is_last_step"] == [True, True, True]
        assert merged["stop_reasons"] == ["eos", "eos", "eos"]

    # ─── Edge cases / optional fields ──────────────────────────────

    def test_single_turn_passthrough(self):
        """Single-turn trajectory is passed through unchanged."""
        tid = _make_tid("single")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[1, 2, 3]],
            "response_ids": [[4, 5]],
            "rewards": [[0.5, 0.6]],
            "loss_masks": [[1, 1]],
            "stop_reasons": ["eos"],
            "rollout_metrics": {"some_metric": 1.0},
            "rollout_logprobs": [[-0.1, -0.2]],
            "trajectory_ids": [tid],
            "rollout_expert_indices": None,
            "is_last_step": [True],
        }

        merged = merge_stepwise_output(gen_out)

        assert merged["prompt_token_ids"] == [[1, 2, 3]]
        assert merged["response_ids"] == [[4, 5]]
        assert merged["loss_masks"] == [[1, 1]]
        assert merged["rollout_logprobs"] == [[-0.1, -0.2]]
        assert merged["rewards"] == [[0.5, 0.6]]
        assert merged["is_last_step"] == [True]
        # rollout_metrics is re-aggregated by concatenate_generator_outputs
        assert merged["rollout_metrics"] is not None

    def test_per_trajectory_scalar_rewards_and_overlong_filtering(self):
        """
        Per-trajectory scalar rewards (List[float]) are handled correctly.

        Also check that overlong filtering is applied correctly. Overlong filtering will have loss
        masks [0] for assistant responses, which merging should preserve.
        """
        for apply_overlong_filtering in [True, False]:
            tid = _make_tid("scalar_rew")
            if apply_overlong_filtering:
                loss_masks = [[0], [0]]
            else:
                loss_masks = [[1], [1]]
            gen_out: GeneratorOutput = {
                "prompt_token_ids": [[10], [10, 20, 30]],
                "response_ids": [[20], [40]],
                "rewards": [0.0, 5.0],  # scalar per turn
                "loss_masks": loss_masks,
                "stop_reasons": None,
                "rollout_metrics": None,
                "rollout_logprobs": None,
                "trajectory_ids": [tid, tid],
                "rollout_expert_indices": None,
                "is_last_step": [False, True],
            }

            merged = merge_stepwise_output(gen_out)

            assert len(merged["response_ids"]) == 1
            assert merged["response_ids"] == [[20, 30, 40]]
            # Scalar reward: use the last turn's value
            assert merged["rewards"] == [5.0]
            assert merged["rollout_logprobs"] is None
            if apply_overlong_filtering:
                assert merged["loss_masks"] == [[0, 0, 0]]
            else:
                assert merged["loss_masks"] == [[1, 0, 1]]

    def test_no_logprobs_no_stop_reasons(self):
        """Works correctly when rollout_logprobs and stop_reasons are None."""
        tid = _make_tid("no_lp")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[1], [1, 2, 3]],
            "response_ids": [[2], [4]],
            "rewards": [[0.0], [1.0]],
            "loss_masks": [[1], [1]],
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": [tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, True],
        }

        merged = merge_stepwise_output(gen_out)

        assert merged["rollout_logprobs"] is None
        assert merged["stop_reasons"] is None
        assert merged["response_ids"] == [[2, 3, 4]]
        assert merged["loss_masks"] == [[1, 0, 1]]

    def test_prefix_mismatch_no_merge(self):
        """If prefix condition fails, turns are kept separate."""
        tid = _make_tid("no_merge")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[10], [99, 88]],  # second prompt doesn't share prefix
            "response_ids": [[20], [40]],
            "rewards": [[0.0], [1.0]],
            "loss_masks": [[1], [1]],
            "stop_reasons": ["continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": [tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, True],
        }

        merged = merge_stepwise_output(gen_out)

        # No merging happened, output has same number of entries
        assert len(merged["response_ids"]) == 2
        assert merged["prompt_token_ids"] == [[10], [99, 88]]
        assert merged["response_ids"] == [[20], [40]]
        assert merged["is_last_step"] == [False, True]

    def test_prefix_of_prompt_plus_response_but_not_prompt_alone_no_merge(self):
        """prompt[i]+response[i] is a prefix of prompt[i+1]+response[i+1] but NOT
        of prompt[i+1] alone. This must NOT merge.

        Turn 1: prompt=[10], response=[20, 30]
          → full = [10, 20, 30]
        Turn 2: prompt=[10, 20], response=[30, 40]
          → prompt[1]+response[1] = [10, 20, 30, 40]

        full ([10,20,30]) IS a prefix of prompt[1]+response[1] ([10,20,30,40]),
        but is NOT a prefix of prompt[1] ([10,20]) since full is longer.
        This would imply response tokens from turn 1 overlap with response tokens
        of turn 2, which is malformed — we intentionally refuse to merge.
        """
        tid = _make_tid("overlap")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[10], [10, 20]],
            "response_ids": [[20, 30], [30, 40]],
            "rewards": [[0.0, 0.0], [1.0, 2.0]],
            "loss_masks": [[1, 0], [1, 1]],
            "stop_reasons": ["continue", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": [tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, True],
        }

        merged = merge_stepwise_output(gen_out)

        # No merging: kept as 2 separate entries
        assert len(merged["response_ids"]) == 2
        assert merged["prompt_token_ids"] == [[10], [10, 20]]
        assert merged["response_ids"] == [[20, 30], [30, 40]]
        assert merged["is_last_step"] == [False, True]

    def test_partial_merge_within_trajectory(self):
        """4 turns where prefix breaks mid-trajectory, producing 2 merged sequences.

        Turn 1: prompt=[1],       response=[2]         → prefix OK for turn 2
        Turn 2: prompt=[1,2,3],   response=[4]         → prefix BREAKS for turn 3
                                                          (re-tokenization changed [2,3] to [23])
        Turn 3: prompt=[1,23,4,5], response=[6]        → prefix OK for turn 4
        Turn 4: prompt=[1,23,4,5,6,7], response=[8]

        Turns 1+2 merge into one sequence, turns 3+4 merge into another.
        Result: 4 turns → 2 sequences.
        """
        tid = _make_tid("partial")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [
                [1],  # turn 1
                [1, 2, 3],  # turn 2: prompt[0]+resp[0]=[1,2] is prefix of [1,2,3] ✓
                [1, 23, 4, 5],  # turn 3: prompt[1]+resp[1]=[1,2,3,4] is NOT prefix of [1,23,4,5] ✗
                [1, 23, 4, 5, 6, 7],  # turn 4: prompt[2]+resp[2]=[1,23,4,5,6] is prefix ✓
            ],
            "response_ids": [
                [2],  # turn 1
                [4],  # turn 2
                [6],  # turn 3
                [8],  # turn 4
            ],
            "rewards": [[0.0], [0.0], [0.0], [10.0]],
            "loss_masks": [[1], [1], [1], [1]],
            "stop_reasons": ["c", "c", "c", "eos"],
            "rollout_metrics": None,
            "rollout_logprobs": [[-1.0], [-2.0], [-3.0], [-4.0]],
            "trajectory_ids": [tid, tid, tid, tid],
            "rollout_expert_indices": None,
            "is_last_step": [False, False, False, True],
        }

        merged = merge_stepwise_output(gen_out)

        # 4 turns → 2 merged sequences
        assert len(merged["response_ids"]) == 2

        # First merged group: turns 1+2
        # prompt=[1], response=[2] + obs_delta=[3] + [4] = [2, 3, 4]
        assert merged["prompt_token_ids"][0] == [1]
        assert merged["response_ids"][0] == [2, 3, 4]
        assert merged["loss_masks"][0] == [1, 0, 1]
        assert merged["rollout_logprobs"][0] == [-1.0, 0.0, -2.0]
        assert merged["rewards"][0] == [0.0, 0.0, 0.0]
        assert merged["is_last_step"][0] is False

        # Second merged group: turns 3+4
        # prompt=[1,23,4,5], response=[6] + obs_delta=[7] + [8] = [6, 7, 8]
        assert merged["prompt_token_ids"][1] == [1, 23, 4, 5]
        assert merged["response_ids"][1] == [6, 7, 8]
        assert merged["loss_masks"][1] == [1, 0, 1]
        assert merged["rollout_logprobs"][1] == [-3.0, 0.0, -4.0]
        assert merged["rewards"][1] == [0.0, 0.0, 10.0]
        assert merged["is_last_step"][1] is True

        assert merged["stop_reasons"] == ["c", "eos"]
        assert merged["trajectory_ids"] == [tid, tid]

    # ─── Assertion / validation on GeneratorOutput shape ───────────

    def test_asserts_trajectory_ids_required(self):
        """Raises when trajectory_ids is None."""
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "rewards": [[1.0]],
            "loss_masks": [[1]],
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": None,
            "rollout_expert_indices": None,
            "is_last_step": [True],
        }
        with pytest.raises(AssertionError, match="trajectory_ids"):
            merge_stepwise_output(gen_out)

    def test_asserts_is_last_step_required(self):
        """Raises when is_last_step is None."""
        tid = _make_tid("x")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "rewards": [[1.0]],
            "loss_masks": [[1]],
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": [tid],
            "rollout_expert_indices": None,
            "is_last_step": None,
        }
        with pytest.raises(TypeError):
            merge_stepwise_output(gen_out)

    def test_asserts_no_expert_indices(self):
        """Raises when rollout_expert_indices is present."""
        tid = _make_tid("x")
        gen_out: GeneratorOutput = {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "rewards": [[1.0]],
            "loss_masks": [[1]],
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": [tid],
            "rollout_expert_indices": [[[[1, 2]]]],
            "is_last_step": [True],
        }
        with pytest.raises(AssertionError, match="rollout_expert_indices not supported"):
            merge_stepwise_output(gen_out)

    # ─── Config validation ─────────────────────────────────────────

    @patch("skyrl.train.utils.utils.validate_batch_sizes", new=lambda cfg: None)
    @patch("skyrl.train.utils.utils.validate_generator_cfg", new=lambda cfg: None)
    def test_validate_cfg_merge_stepwise_requires_step_wise(self):
        """`merge_stepwise_output=True` without `step_wise_trajectories=True` must fail validation.

        Prefix-aware merging operates on step-wise-only fields (trajectory_ids, is_last_step), so
        enabling it without step-wise training would crash later with a confusing assertion inside
        `merge_stepwise_output`. `validate_cfg` should reject the combination up front.
        """
        cfg = example_dummy_config()
        cfg.generator.merge_stepwise_output = True
        cfg.generator.step_wise_trajectories = False
        with pytest.raises(ValueError, match="merge_stepwise_output.*requires.*step_wise_trajectories"):
            validate_cfg(cfg)
