"""
Tests for Tinker training utilities.

Run with:
    python -m pytest skyrl-train/integrations/fleet/tests/test_tinker_training.py -v
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    truncate_sequence,
    truncate_auxiliary_data,
    apply_overlong_filtering_simple,
    prepare_training_sequence,
)


class TestTruncateSequence:
    """Tests for sequence truncation logic."""

    def test_no_truncation_within_limit(self):
        """Sequences within max_sequence_length should not be modified."""
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5]
        max_sequence_length = 10

        full_seq, truncated_resp, resp_len = truncate_sequence(prompt_ids, response_ids, max_sequence_length)

        assert full_seq == [1, 2, 3, 4, 5]
        assert truncated_resp == [4, 5]
        assert resp_len == 2

    def test_truncate_overlong_sequence(self):
        """Sequences exceeding max_sequence_length should be truncated."""
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5, 6, 7, 8, 9, 10]
        max_sequence_length = 8

        full_seq, truncated_resp, resp_len = truncate_sequence(prompt_ids, response_ids, max_sequence_length)

        assert len(full_seq) == max_sequence_length
        assert full_seq == [1, 2, 3, 4, 5, 6, 7, 8]
        assert truncated_resp == [4, 5, 6, 7, 8]
        assert resp_len == 5

    def test_truncation_preserves_prompt(self):
        """Truncation should preserve prompt and truncate response."""
        prompt_ids = [1, 2, 3, 4, 5]  # 5 tokens
        response_ids = [6, 7, 8, 9, 10]  # 5 tokens
        max_sequence_length = 7

        full_seq, truncated_resp, resp_len = truncate_sequence(prompt_ids, response_ids, max_sequence_length)

        # Prompt should be fully preserved
        assert full_seq[:5] == prompt_ids
        # Response should be truncated
        assert resp_len == 2  # Only 2 response tokens fit
        assert truncated_resp == [6, 7]

    def test_exact_limit(self):
        """Sequences exactly at limit should not be modified."""
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5, 6, 7]
        max_sequence_length = 7

        full_seq, truncated_resp, resp_len = truncate_sequence(prompt_ids, response_ids, max_sequence_length)

        assert len(full_seq) == 7
        assert full_seq == [1, 2, 3, 4, 5, 6, 7]
        assert truncated_resp == response_ids
        assert resp_len == 4

    def test_empty_response(self):
        """Empty response should work correctly."""
        prompt_ids = [1, 2, 3]
        response_ids = []
        max_sequence_length = 10

        full_seq, truncated_resp, resp_len = truncate_sequence(prompt_ids, response_ids, max_sequence_length)

        assert full_seq == [1, 2, 3]
        assert truncated_resp == []
        assert resp_len == 0


class TestTruncateAuxiliaryData:
    """Tests for auxiliary data truncation."""

    def test_truncate_to_shorter_length(self):
        """Data should be truncated to target length."""
        data = [1, 2, 3, 4, 5]
        result = truncate_auxiliary_data(data, 3)
        assert result == [1, 2, 3]

    def test_no_truncation_needed(self):
        """Data within limit should not be modified."""
        data = [1, 2, 3]
        result = truncate_auxiliary_data(data, 5)
        assert result == [1, 2, 3]

    def test_exact_length(self):
        """Data exactly at limit should not be modified."""
        data = [1, 2, 3]
        result = truncate_auxiliary_data(data, 3)
        assert result == [1, 2, 3]

    def test_truncate_floats(self):
        """Float data (like logprobs) should work."""
        data = [-0.1, -0.2, -0.3, -0.4, -0.5]
        result = truncate_auxiliary_data(data, 3)
        assert result == [-0.1, -0.2, -0.3]

    def test_empty_data(self):
        """Empty data should work correctly."""
        data = []
        result = truncate_auxiliary_data(data, 3)
        assert result == []


class TestOverlongFiltering:
    """Tests for DAPO overlong filtering."""

    def test_zeros_truncated_responses(self):
        """Responses not ending with EOS should have loss mask zeroed."""
        loss_masks = [[1, 1, 1], [1, 1, 1]]
        response_ids = [[1, 2, 3], [4, 5, 99]]  # First doesn't end with EOS, second does
        eos_token_id = 99

        result = apply_overlong_filtering_simple(loss_masks, response_ids, eos_token_id)

        assert result[0] == [0, 0, 0], "Truncated response should have zeroed mask"
        assert result[1] == [1, 1, 1], "Complete response should keep original mask"

    def test_preserves_length(self):
        """Output masks should have same length as input masks."""
        loss_masks = [[1, 0, 1, 1], [1, 1]]
        response_ids = [[1, 2, 3, 4], [5, 6]]
        eos_token_id = 99

        result = apply_overlong_filtering_simple(loss_masks, response_ids, eos_token_id)

        assert len(result[0]) == len(loss_masks[0])
        assert len(result[1]) == len(loss_masks[1])

    def test_empty_response(self):
        """Empty responses should have loss mask zeroed."""
        loss_masks = [[1, 1, 1]]
        response_ids = [[]]
        eos_token_id = 99

        result = apply_overlong_filtering_simple(loss_masks, response_ids, eos_token_id)

        assert result[0] == [0, 0, 0], "Empty response should have zeroed mask"

    def test_all_complete(self):
        """All responses ending with EOS should preserve masks."""
        loss_masks = [[1, 1], [1, 0, 1]]
        response_ids = [[1, 99], [2, 3, 99]]
        eos_token_id = 99

        result = apply_overlong_filtering_simple(loss_masks, response_ids, eos_token_id)

        assert result[0] == [1, 1]
        assert result[1] == [1, 0, 1]

    def test_all_truncated(self):
        """All responses not ending with EOS should have zeroed masks."""
        loss_masks = [[1, 1], [1, 0, 1]]
        response_ids = [[1, 2], [3, 4, 5]]
        eos_token_id = 99

        result = apply_overlong_filtering_simple(loss_masks, response_ids, eos_token_id)

        assert result[0] == [0, 0]
        assert result[1] == [0, 0, 0]


class TestPrepareTrainingSequence:
    """Tests for combined preparation function."""

    def test_no_truncation(self):
        """Sequences within limit should not be truncated."""
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5]
        logprobs = [-0.1, -0.2]
        loss_mask = [1, 1]
        max_sequence_length = 10

        full_seq, trunc_logprobs, trunc_mask, was_truncated = prepare_training_sequence(
            prompt_ids, response_ids, logprobs, loss_mask, max_sequence_length
        )

        assert full_seq == [1, 2, 3, 4, 5]
        assert trunc_logprobs == [-0.1, -0.2]
        assert trunc_mask == [1, 1]
        assert was_truncated is False

    def test_with_truncation(self):
        """Overlong sequences should be truncated with flag set."""
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5, 6, 7, 8]
        logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]
        loss_mask = [1, 1, 1, 1, 1]
        max_sequence_length = 6

        full_seq, trunc_logprobs, trunc_mask, was_truncated = prepare_training_sequence(
            prompt_ids, response_ids, logprobs, loss_mask, max_sequence_length
        )

        assert len(full_seq) == 6
        assert full_seq == [1, 2, 3, 4, 5, 6]
        assert trunc_logprobs == [-0.1, -0.2, -0.3]
        assert trunc_mask == [1, 1, 1]
        assert was_truncated is True

    def test_preserves_mask_values(self):
        """Truncation should preserve original mask values."""
        prompt_ids = [1, 2]
        response_ids = [3, 4, 5, 6]
        logprobs = [-0.1, -0.2, -0.3, -0.4]
        loss_mask = [1, 0, 1, 0]  # Mixed mask
        max_sequence_length = 4

        full_seq, trunc_logprobs, trunc_mask, was_truncated = prepare_training_sequence(
            prompt_ids, response_ids, logprobs, loss_mask, max_sequence_length
        )

        assert trunc_mask == [1, 0]  # First 2 values preserved
        assert was_truncated is True


class TestCombinedFlow:
    """
    Tests for combined DAPO filtering + truncation flow.

    These tests validate the pattern used by prepare_training_data() in
    main_fleet_tinker.py, which:
    1. Applies DAPO overlong filtering (zeros mask if no EOS)
    2. Truncates sequences to max_sequence_length
    3. Builds training datums (Tinker-specific, not tested here)
    """

    def test_dapo_then_truncate(self):
        """DAPO filtering should be applied, then truncation."""
        # Simulate the flow in prepare_training_data()
        response_ids = [1, 2, 3, 4, 5]  # Doesn't end with EOS
        loss_mask = [1, 1, 1, 1, 1]
        eos_token_id = 99

        # Step 1: Apply DAPO overlong filtering
        filtered_masks = apply_overlong_filtering_simple([loss_mask], [response_ids], eos_token_id)
        filtered_mask = filtered_masks[0]

        # Should be zeroed because response doesn't end with EOS
        assert filtered_mask == [0, 0, 0, 0, 0]

        # Step 2: Prepare training sequence with truncation
        prompt_ids = [10, 11, 12]
        max_sequence_length = 6

        full_seq, _, trunc_mask, _ = prepare_training_sequence(
            prompt_ids, response_ids, [0.0] * 5, filtered_mask, max_sequence_length
        )

        # Both should be truncated
        assert len(full_seq) == max_sequence_length
        assert len(trunc_mask) == 3  # 6 - 3 = 3 response tokens
        assert trunc_mask == [0, 0, 0]  # Still zeroed

    def test_complete_response_then_truncate(self):
        """Complete response (ends with EOS) but still needs truncation."""
        response_ids = [1, 2, 3, 4, 99]  # Ends with EOS
        loss_mask = [1, 1, 1, 1, 1]
        eos_token_id = 99

        # Step 1: Apply DAPO overlong filtering
        filtered_masks = apply_overlong_filtering_simple([loss_mask], [response_ids], eos_token_id)
        filtered_mask = filtered_masks[0]

        # Should NOT be zeroed because response ends with EOS
        assert filtered_mask == [1, 1, 1, 1, 1]

        # Step 2: Prepare training sequence with truncation
        prompt_ids = [10, 11, 12]
        max_sequence_length = 6

        full_seq, _, trunc_mask, was_truncated = prepare_training_sequence(
            prompt_ids, response_ids, [0.0] * 5, filtered_mask, max_sequence_length
        )

        # Truncated but mask values preserved
        assert len(full_seq) == max_sequence_length
        assert len(trunc_mask) == 3
        assert trunc_mask == [1, 1, 1]  # Original values preserved
        assert was_truncated is True

    def test_batch_processing(self):
        """Test processing multiple rollouts (as prepare_training_data does)."""
        eos_token_id = 99

        # Simulate batch of rollouts with mixed completion status
        rollouts = [
            {
                "prompt_ids": [1, 2, 3],
                "response_ids": [4, 5, 6],  # No EOS - should be zeroed
                "logprobs": [-0.1, -0.2, -0.3],
                "loss_mask": [1, 1, 1],
            },
            {
                "prompt_ids": [10, 11],
                "response_ids": [12, 13, 99],  # Ends with EOS - should be preserved
                "logprobs": [-0.4, -0.5, -0.6],
                "loss_mask": [1, 1, 1],
            },
            {
                "prompt_ids": [20, 21, 22],
                "response_ids": [23, 24, 25, 26, 27],  # No EOS, needs truncation
                "logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
                "loss_mask": [1, 1, 1, 1, 1],
            },
        ]
        max_sequence_length = 6

        # Step 1: Apply DAPO filtering to all rollouts
        all_response_ids = [r["response_ids"] for r in rollouts]
        all_loss_masks = [r["loss_mask"] for r in rollouts]
        filtered_masks = apply_overlong_filtering_simple(all_loss_masks, all_response_ids, eos_token_id)

        # Rollout 0: No EOS -> zeroed
        assert filtered_masks[0] == [0, 0, 0]
        # Rollout 1: Has EOS -> preserved
        assert filtered_masks[1] == [1, 1, 1]
        # Rollout 2: No EOS -> zeroed
        assert filtered_masks[2] == [0, 0, 0, 0, 0]

        # Step 2: Prepare each sequence with truncation
        results = []
        for idx, rollout in enumerate(rollouts):
            full_seq, trunc_logprobs, trunc_mask, was_truncated = prepare_training_sequence(
                rollout["prompt_ids"],
                rollout["response_ids"],
                rollout["logprobs"],
                filtered_masks[idx],
                max_sequence_length,
            )
            results.append(
                {
                    "full_seq": full_seq,
                    "trunc_mask": trunc_mask,
                    "was_truncated": was_truncated,
                }
            )

        # Rollout 0: 3+3=6, no truncation needed, mask zeroed
        assert len(results[0]["full_seq"]) == 6
        assert results[0]["trunc_mask"] == [0, 0, 0]
        assert results[0]["was_truncated"] is False

        # Rollout 1: 2+3=5, no truncation needed, mask preserved
        assert len(results[1]["full_seq"]) == 5
        assert results[1]["trunc_mask"] == [1, 1, 1]
        assert results[1]["was_truncated"] is False

        # Rollout 2: 3+5=8 > 6, truncated to 6, mask zeroed
        assert len(results[2]["full_seq"]) == 6
        assert len(results[2]["trunc_mask"]) == 3  # 6 - 3 = 3 response tokens
        assert results[2]["trunc_mask"] == [0, 0, 0]
        assert results[2]["was_truncated"] is True
