from skyrl_train.utils.trainer_utils import (
    run_on_each_node,
    cleanup_old_checkpoints,
    validate_consistency_for_latest_checkpoint,
    sanitize_data_source,
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
)
from typing import Union
import ray
import os
import tempfile
import pytest

from unittest.mock import Mock, patch, mock_open
import json

BasicType = Union[int, float, str, bool, type(None)]


def test_run_on_node_local_rank_0():
    def fn(x):
        return x + 1

    all_nodes = [node for node in ray.nodes() if node.get("CPU", 0) > 0]
    # repeat the node ids 4 times to test that the function is called only once per node
    node_ids = [all_nodes[i]["NodeID"] for i in range(len(all_nodes))] * 4
    ret = run_on_each_node(node_ids, fn, 1)
    assert ret == [2] * len(all_nodes)


def setup_mock_ckpts(tmpdir, checkpoint_steps):
    """
    Sets up dummy checkpoint directories.
    """
    # Create dummy checkpoint directories
    for step in checkpoint_steps:
        os.makedirs(os.path.join(tmpdir, f"global_step_{step}"))
    return


def test_cleanup_old_checkpoints():
    """
    Verify that _cleanup_old_checkpoints correctly removes old checkpoints
    while keeping the most recent ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=2, current_global_step=11)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        expected_remaining = ["global_step_10", "global_step_11"]

        assert len(remaining_dirs) == 2, "Incorrect number of checkpoints remaining"
        assert remaining_dirs == expected_remaining, "Did not keep the correct (most recent) checkpoints"

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute - remove all checkpoints
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=0, current_global_step=11)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 0, "Cleanup should have removed all checkpoints"

    # 3. Test cleanup with `current_global_step` less than the highest global step in the folder
    # This means that the folder contains checkpoints from a previous run.
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=2, current_global_step=2)

        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 4, "Cleanup should not have removed any checkpoints"


def test_cleanup_does_not_run_when_not_needed():
    """
    Verify that cleanup does not remove any checkpoints if the total number
    is less than or equal to max_ckpts_to_keep.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=5, current_global_step=4)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 4, "Cleanup should not have removed any checkpoints"


def test_cleanup_with_negative_max_checkpoints():
    """
    Verify that cleanup is disabled when max_ckpts_to_keep is -1
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=-1, current_global_step=5)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 5, "Cleanup should be disabled when max_ckpts_to_keep is -1"


def test_validate_consistency_for_latest_checkpoint():
    """
    Verify that `validate_consistency_for_latest_checkpoint` correctly validates the checkpoint folder.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("5")

        latest_ckpt_path = os.path.join(tmpdir, "global_step_5")
        ckpt_iteration = 5

        # 2. Execute
        validate_consistency_for_latest_checkpoint(
            tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=1
        )


def test_validate_consistency_for_latest_checkpoint_with_inconsistent_folder():
    """
    Verify that `validate_consistency_for_latest_checkpoint` correctly validates the checkpoint folder.
    """
    # Example 1: `latest_ckpt_global_step.txt` points to a lower global step than the highest global step in the folder
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # change the latest checkpoint file to point to a lower global step
        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("3")

        latest_ckpt_path = os.path.join(tmpdir, "global_step_3")
        ckpt_iteration = 3
        save_interval = 1

        # 2. Execute
        with pytest.raises(ValueError, match="Inconsistent checkpoint folder"):
            validate_consistency_for_latest_checkpoint(
                tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=save_interval
            )

    # Example 2: `latest_ckpt_global_step.txt` points to a lower global step but it's within the save interval
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 3, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # change the latest checkpoint file to point to a lower global step
        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("3")

        save_interval = 2
        latest_ckpt_path = os.path.join(tmpdir, "global_step_3")
        ckpt_iteration = 3

        # 2. Execute
        validate_consistency_for_latest_checkpoint(
            tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=save_interval
        )


def test_sanitize_data_source_none():
    """Test sanitize_data_source with None input."""
    result = sanitize_data_source(None)
    assert result == "unknown"


def test_sanitize_data_source_slash_replacement():
    """Test sanitize_data_source replaces slashes with underscores."""
    result = sanitize_data_source("dataset/with/slashes")
    assert result == "dataset_with_slashes"


def test_sanitize_data_source_normal_string():
    """Test sanitize_data_source with normal string."""
    result = sanitize_data_source("normal_dataset")
    assert result == "normal_dataset"


def test_calculate_per_dataset_metrics_single_source():
    """Test calculate_per_dataset_metrics with single data source."""
    # Create test data
    generator_outputs = {
        "rewards": [0.5, 0.7, 0.9],
        "prompt_token_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "response_ids": [[10, 11], [12, 13], [14, 15]],
    }
    uids = ["uid1", "uid2", "uid3"]
    data_sources = ["dataset1", "dataset1", "dataset1"]

    result = calculate_per_dataset_metrics(generator_outputs, uids, data_sources, 2)

    # Verify results - actual computed values
    # Mean reward: (0.5 + 0.7 + 0.9) / 3 = 0.7
    # Pass@N: all rewards > 0, all unique uids, so 3/3 = 1.0
    assert "eval/dataset1/avg_score" in result
    assert "eval/dataset1/pass_at_2" in result
    assert result["eval/dataset1/avg_score"] == pytest.approx(0.7)
    assert result["eval/dataset1/pass_at_2"] == 1.0


def test_calculate_per_dataset_metrics_multiple_sources():
    """Test calculate_per_dataset_metrics with multiple data sources including None."""
    # Create test data with mixed sources
    generator_outputs = {
        "rewards": [0.5, 0.7, 0.9, 0.4],
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "response_ids": [[10, 11], [12, 13], [14, 15], [16, 17]],
    }
    uids = ["uid1", "uid2", "uid3", "uid4"]
    data_sources = ["dataset1", None, "dataset1", None]

    result = calculate_per_dataset_metrics(generator_outputs, uids, data_sources, 2)

    # Verify results for both datasets - actual computed values
    # dataset1: indices 0, 2 -> rewards [0.5, 0.9] -> mean = 0.7, pass@n = 2/2 = 1.0
    # unknown (None): indices 1, 3 -> rewards [0.7, 0.4] -> mean = 0.55, pass@n = 2/2 = 1.0
    assert "eval/dataset1/avg_score" in result
    assert "eval/dataset1/pass_at_2" in result
    assert "eval/unknown/avg_score" in result
    assert "eval/unknown/pass_at_2" in result

    assert result["eval/dataset1/avg_score"] == pytest.approx(0.7)
    assert result["eval/dataset1/pass_at_2"] == 1.0
    assert result["eval/unknown/avg_score"] == pytest.approx(0.55)
    assert result["eval/unknown/pass_at_2"] == 1.0


@patch("builtins.open", new_callable=mock_open)
def test_dump_per_dataset_eval_results_comprehensive(mock_file):
    """Test dump_per_dataset_eval_results comprehensive functionality."""
    # Mock dump directory path
    mock_dump_dir = Mock()
    mock_dump_dir.__truediv__ = Mock(side_effect=lambda x: f"mock_path/{x}")

    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.decode.side_effect = lambda x: f"decoded_{x}"

    # Create test data
    generator_outputs = {
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6]],
        "response_ids": [[10, 11], [12, 13], [14, 15]],
        "rewards": [0.5, 0.7, 0.9],
        "stop_reasons": ["stop1", "stop2", "stop3"],
    }
    data_sources = ["dataset1", None, "dataset1"]
    all_envs = ["env1", "env2", "env3"]
    env_extras = [{"extra1": "val1"}, {"extra2": "val2"}, {"extra3": "val3"}]
    eval_metrics = {"eval/dataset1/avg_score": 0.8, "eval/unknown/avg_score": 0.6}

    # Call the function
    dump_per_dataset_eval_results(
        mock_dump_dir, mock_tokenizer, generator_outputs, data_sources, all_envs, env_extras, eval_metrics
    )

    # Verify tokenizer was called for decoding
    assert mock_tokenizer.decode.call_count == 6  # 3 prompts + 3 responses

    # Verify files were opened (2 per-dataset files + 1 aggregated file)
    assert mock_file.call_count == 3

    # Verify file writes occurred
    handle = mock_file.return_value
    assert handle.write.call_count > 0

    # Verify JSON structure by checking some write calls contain expected data
    write_calls = [call[0][0] for call in handle.write.call_args_list]
    json_writes = [call for call in write_calls if call.strip() and not call.startswith("Dumped")]

    # At least one JSON line should contain our test data
    assert len(json_writes) > 0

    # Parse one of the JSON writes to verify structure
    for write_call in json_writes:
        try:
            data = json.loads(write_call.strip())
            if "input_prompt" in data:
                # This is a per-dataset entry
                assert "output_response" in data
                assert "score" in data
                assert "data_source" in data
                break
        except json.JSONDecodeError:
            continue
