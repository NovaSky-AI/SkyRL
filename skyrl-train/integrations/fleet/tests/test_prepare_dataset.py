"""Tests for prepare_dataset.py - stratified eval split functionality."""

import json
import os
import tempfile

import pytest

from integrations.fleet.prepare_dataset import (
    HELD_OUT_ENVS,
    MIN_EVAL_SAMPLES,
    _task_to_record,
    hash_to_split,
    load_tasks_from_json,
    prepare_fleet_dataset,
)


class TestHashToSplit:
    """Tests for deterministic hash-based split assignment."""

    def test_deterministic_same_key(self):
        """Same task_key always produces same split."""
        key = "github_task_123"
        results = [hash_to_split(key) for _ in range(100)]
        assert len(set(results)) == 1, "Same key should always produce same split"

    def test_deterministic_across_calls(self):
        """Split assignment is reproducible."""
        keys = ["task_a", "task_b", "task_c", "github_456", "booking_789"]
        first_run = [hash_to_split(k) for k in keys]
        second_run = [hash_to_split(k) for k in keys]
        assert first_run == second_run

    def test_respects_eval_ratio(self):
        """Approximately eval_ratio of tasks go to eval split."""
        # Use many keys to get statistical significance
        keys = [f"task_{i}" for i in range(10000)]
        splits = [hash_to_split(k, eval_ratio=0.1) for k in keys]
        eval_count = splits.count("eval")
        # Should be roughly 10% (allow some variance)
        assert 800 < eval_count < 1200, f"Expected ~1000 eval, got {eval_count}"

    def test_different_eval_ratios(self):
        """Different eval_ratio produces different distribution."""
        keys = [f"task_{i}" for i in range(1000)]
        splits_10 = [hash_to_split(k, eval_ratio=0.1) for k in keys]
        splits_20 = [hash_to_split(k, eval_ratio=0.2) for k in keys]
        eval_10 = splits_10.count("eval")
        eval_20 = splits_20.count("eval")
        assert eval_20 > eval_10, "Higher eval_ratio should produce more eval samples"

    def test_returns_valid_split(self):
        """Only returns 'train' or 'eval'."""
        keys = [f"key_{i}" for i in range(100)]
        splits = [hash_to_split(k) for k in keys]
        assert all(s in ("train", "eval") for s in splits)


class TestTaskToRecord:
    """Tests for _task_to_record conversion."""

    def test_valid_task(self):
        """Converts valid task to record format."""
        task = {
            "key": "github_123",
            "prompt": "Create a new repository",
            "env_key": "github",
        }
        record = _task_to_record(task, "github")
        assert record is not None
        assert record["prompt"] == [{"role": "user", "content": "Create a new repository"}]
        assert record["env_class"] == "fleet_task"
        assert record["task_key"] == "github_123"
        assert record["data_source"] == "github"

    def test_task_key_field(self):
        """Handles 'task_key' field name."""
        task = {
            "task_key": "booking_456",
            "prompt": "Search for hotels",
        }
        record = _task_to_record(task, "booking")
        assert record["task_key"] == "booking_456"

    def test_missing_key_returns_none(self):
        """Returns None if task_key is missing."""
        task = {"prompt": "Some prompt"}
        record = _task_to_record(task, "test")
        assert record is None

    def test_missing_prompt_returns_none(self):
        """Returns None if prompt is missing."""
        task = {"key": "task_123"}
        record = _task_to_record(task, "test")
        assert record is None

    def test_empty_prompt_returns_none(self):
        """Returns None if prompt is empty."""
        task = {"key": "task_123", "prompt": ""}
        record = _task_to_record(task, "test")
        assert record is None


class TestLoadTasksFromJson:
    """Tests for JSON loading with different formats."""

    def test_array_format(self):
        """Loads tasks from array format JSON."""
        tasks_data = [
            {"key": "task_1", "prompt": "Do something"},
            {"key": "task_2", "prompt": "Do another thing"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tasks_data, f)
            f.flush()
            tasks = load_tasks_from_json(f.name)
        os.unlink(f.name)
        assert len(tasks) == 2
        assert tasks[0]["key"] == "task_1"

    def test_object_with_tasks_key(self):
        """Loads tasks from object format with 'tasks' key."""
        tasks_data = {
            "tasks": [
                {"key": "task_1", "prompt": "Do something"},
                {"key": "task_2", "prompt": "Do another thing"},
            ],
            "metadata": {"version": "1.0"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tasks_data, f)
            f.flush()
            tasks = load_tasks_from_json(f.name)
        os.unlink(f.name)
        assert len(tasks) == 2

    def test_invalid_format_raises(self):
        """Raises ValueError for invalid JSON format."""
        tasks_data = {"invalid": "format"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tasks_data, f)
            f.flush()
            with pytest.raises(ValueError, match="Invalid JSON format"):
                load_tasks_from_json(f.name)
        os.unlink(f.name)


class TestHeldOutEnvs:
    """Tests for held-out eval environment configuration."""

    def test_tool_use_held_out(self):
        """Outlook is held out for tool_use."""
        assert "outlook" in HELD_OUT_ENVS["tool_use"]

    def test_computer_use_held_out(self):
        """Instacart is held out for computer_use."""
        assert "instacart" in HELD_OUT_ENVS["computer_use"]

    def test_min_eval_samples_threshold(self):
        """MIN_EVAL_SAMPLES is set to 10."""
        assert MIN_EVAL_SAMPLES == 10


class TestPrepareFleetDataset:
    """Integration tests for prepare_fleet_dataset."""

    def _create_test_tasks(self, env_counts: dict, modality: str = "tool_use"):
        """Create test tasks for multiple environments."""
        tasks = []
        for env_key, count in env_counts.items():
            for i in range(count):
                tasks.append(
                    {
                        "key": f"{env_key}_task_{i}",
                        "prompt": f"Task {i} for {env_key}",
                        "env_key": env_key,
                        "task_modality": modality,
                    }
                )
        return tasks

    def test_held_out_env_goes_to_eval(self):
        """Held-out environments go entirely to eval split."""
        tasks = self._create_test_tasks(
            {"github": 100, "outlook": 24},  # outlook is held out for tool_use
            modality="tool_use",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            prepare_fleet_dataset(
                tasks_json=json_path,
                output_dir=tmpdir,
                modality="tool_use",
            )

            # Check validation.parquet contains outlook tasks (held-out go to eval)
            val_path = os.path.join(tmpdir, "validation.parquet")
            assert os.path.exists(val_path)

            import pyarrow.parquet as pq

            val_df = pq.read_table(val_path).to_pandas()
            # Should have outlook (24) + github eval (~12)
            outlook_tasks = [k for k in val_df["task_key"] if "outlook" in k]
            assert len(outlook_tasks) == 24

    def test_small_env_all_to_train(self):
        """Environments with < MIN_EVAL_SAMPLES go entirely to train."""
        # 5 tasks = less than MIN_EVAL_SAMPLES (10) expected in eval
        tasks = self._create_test_tasks({"small_env": 5}, modality="tool_use")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            prepare_fleet_dataset(
                tasks_json=json_path,
                output_dir=tmpdir,
                modality="tool_use",
            )

            train_path = os.path.join(tmpdir, "train.parquet")
            val_path = os.path.join(tmpdir, "validation.parquet")

            import pyarrow.parquet as pq

            train_df = pq.read_table(train_path).to_pandas()
            assert len(train_df) == 5

            # validation.parquet should not exist or be empty
            assert not os.path.exists(val_path)

    def test_stratified_split(self):
        """Large environments get stratified train/eval split."""
        # 200 tasks = ~20 expected in eval (above MIN_EVAL_SAMPLES)
        tasks = self._create_test_tasks({"github": 200}, modality="tool_use")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            prepare_fleet_dataset(
                tasks_json=json_path,
                output_dir=tmpdir,
                modality="tool_use",
                eval_ratio=0.1,
            )

            import pyarrow.parquet as pq

            train_df = pq.read_table(os.path.join(tmpdir, "train.parquet")).to_pandas()
            val_df = pq.read_table(os.path.join(tmpdir, "validation.parquet")).to_pandas()

            total = len(train_df) + len(val_df)
            assert total == 200

            # Check roughly 10% in eval (allow some variance due to hash)
            eval_ratio = len(val_df) / total
            assert 0.05 < eval_ratio < 0.15

    def test_modality_filter(self):
        """Only tasks matching modality are included."""
        tasks = self._create_test_tasks({"github": 50}, modality="tool_use") + self._create_test_tasks(
            {"amazon": 50}, modality="computer_use"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            prepare_fleet_dataset(
                tasks_json=json_path,
                output_dir=tmpdir,
                modality="tool_use",
            )

            import pyarrow.parquet as pq

            train_df = pq.read_table(os.path.join(tmpdir, "train.parquet")).to_pandas()
            # Only github (tool_use) tasks, not amazon (computer_use)
            assert all("github" in k for k in train_df["task_key"])

    def test_env_filter(self):
        """env_filter limits to specific environment."""
        tasks = self._create_test_tasks(
            {"github": 100, "booking": 100},
            modality="tool_use",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            prepare_fleet_dataset(
                tasks_json=json_path,
                output_dir=tmpdir,
                modality="tool_use",
                env_filter="github",
            )

            import pyarrow.parquet as pq

            train_df = pq.read_table(os.path.join(tmpdir, "train.parquet")).to_pandas()
            # Only github tasks
            assert all("github" in k for k in train_df["task_key"])

    def test_deterministic_split_across_runs(self):
        """Same tasks produce same split across multiple runs."""
        tasks = self._create_test_tasks({"github": 200}, modality="tool_use")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            # Run twice with different output dirs
            out1 = os.path.join(tmpdir, "run1")
            out2 = os.path.join(tmpdir, "run2")

            prepare_fleet_dataset(tasks_json=json_path, output_dir=out1, modality="tool_use")
            prepare_fleet_dataset(tasks_json=json_path, output_dir=out2, modality="tool_use")

            import pyarrow.parquet as pq

            train1 = set(pq.read_table(os.path.join(out1, "train.parquet")).to_pandas()["task_key"])
            train2 = set(pq.read_table(os.path.join(out2, "train.parquet")).to_pandas()["task_key"])
            val1 = set(pq.read_table(os.path.join(out1, "validation.parquet")).to_pandas()["task_key"])
            val2 = set(pq.read_table(os.path.join(out2, "validation.parquet")).to_pandas()["task_key"])

            assert train1 == train2, "Train splits should be identical"
            assert val1 == val2, "Validation splits should be identical"

    def test_computer_use_held_out(self):
        """Instacart is held out for computer_use modality."""
        tasks = self._create_test_tasks(
            {"amazon": 100, "instacart": 32},
            modality="computer_use",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tasks.json")
            with open(json_path, "w") as f:
                json.dump(tasks, f)

            prepare_fleet_dataset(
                tasks_json=json_path,
                output_dir=tmpdir,
                modality="computer_use",
            )

            import pyarrow.parquet as pq

            val_df = pq.read_table(os.path.join(tmpdir, "validation.parquet")).to_pandas()
            # Should have instacart (32) + amazon eval (~14)
            instacart_tasks = [k for k in val_df["task_key"] if "instacart" in k]
            assert len(instacart_tasks) == 32
