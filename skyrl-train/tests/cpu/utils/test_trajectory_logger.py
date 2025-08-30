"""
Unit tests for trajectory logging functionality.

This module tests the TrajectoryLogger system including:
- Base TrajectoryLogger abstract class utilities
- WandbTableTrajectoryLogger with proper mocking
- CSVTrajectoryLogger file operations
- Trajectory dataclass functionality

Run with: uv run --extra dev --isolated pytest tests/cpu/utils/test_trajectory_logger.py
"""

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig
import pandas as pd
import pytest

from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.generators.trajectory_logger import (
    CSVTrajectoryLogger,
    Trajectory,
    TrajectoryLogger,
    WandbTableTrajectoryLogger,
    create_trajectory_logger_from_config,
)


# Test Fixtures
@pytest.fixture
def sample_trajectory() -> Trajectory:
    """Create a single sample trajectory for testing."""
    prompt: ConversationType = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    
    chat_history: ConversationType = prompt + [
        {"role": "assistant", "content": "2 + 2 equals 4."}
    ]
    
    return Trajectory(
        prompt=prompt,
        chat_history=chat_history,
        response="The answer is 4.",
        reward=1.0,
        stop_reason="stop",
        env_class="gsm8k",
        env_extras={"difficulty": "easy", "index": 0},
        loss_mask=[1, 1, 1, 1, 1, 1],
        metadata={"trajectory_id": "test_0"}
    )


@pytest.fixture
def sample_trajectories(sample_trajectory) -> List[Trajectory]:
    """Create multiple sample trajectories with varying properties."""
    trajectories = []
    for i in range(3):
        traj = Trajectory(
            prompt=sample_trajectory.prompt,
            chat_history=sample_trajectory.chat_history,
            response=f"Response {i}: The answer is {4 + i}.",
            reward=float(i) / 2,  # 0.0, 0.5, 1.0
            stop_reason="stop" if i < 2 else "length",
            env_class="gsm8k",
            env_extras={"difficulty": "easy", "index": i},
            loss_mask=[1, 1, 1, 1, 1, 1],
            metadata={"trajectory_id": f"test_{i}"}
        )
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def concrete_logger():
    """Create a concrete implementation of TrajectoryLogger for testing abstract methods."""
    class ConcreteLogger(TrajectoryLogger):
        def log(self, trajectories, step, prefix="train"):
            """Concrete implementation for testing."""
            pass
    return ConcreteLogger()


# Test Classes


class TestTrajectoryLogger:
    """Test the base TrajectoryLogger abstract class utility methods."""
    
    def test_format_conversation(self, concrete_logger):
        """Test conversation formatting produces readable chat format."""
        conversation: ConversationType = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        formatted = concrete_logger._format_conversation(conversation)
        assert "[user]: Hello" in formatted
        assert "[assistant]: Hi there!" in formatted
    
    def test_to_dataframe_conversion(self, concrete_logger, sample_trajectories):
        """Test conversion of trajectories to pandas DataFrame."""
        df = concrete_logger.to_dataframe(sample_trajectories)
        
        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_trajectories)
        
        # Check required columns exist
        expected_columns = {"prompt", "response", "reward", "stop_reason", "env_class", "chat_turns"}
        assert expected_columns.issubset(set(df.columns))
        
        # Validate specific data values
        assert df["reward"].iloc[0] == 0.0
        assert df["reward"].iloc[1] == 0.5
        assert df["stop_reason"].iloc[2] == "length"


class TestWandbTableTrajectoryLogger:
    """Test WandB table trajectory logger with proper API mocking."""
    
    def test_initialization_success(self):
        """Test successful WandB logger initialization with all parameters."""
        mock_wandb_module = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(
                max_trajectories=5,
                log_full_history=False
            )
            
            # Verify initialization parameters
            assert logger.max_trajectories == 5
            assert logger.log_full_history is False
            assert logger.wandb == mock_wandb_module
    
    def test_log_prompt_response_mode(self, sample_trajectories):
        """Test logging in prompt/response mode with trajectory limit verification."""
        mock_wandb_module = MagicMock()
        mock_table = MagicMock()
        mock_wandb_module.Table.return_value = mock_table
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(
                max_trajectories=2,  # Limit to first 2 trajectories
                log_full_history=False
            )
            
            logger.log(sample_trajectories, step=100, prefix="test")
            
            # Verify table creation with correct column structure
            mock_wandb_module.Table.assert_called_once()
            table_kwargs = mock_wandb_module.Table.call_args[1]
            columns = table_kwargs["columns"]
            
            # Should have separate prompt/response columns
            assert "prompt" in columns
            assert "response" in columns
            assert "reward" in columns
            assert "full_conversation" not in columns
            
            # Verify trajectory limiting (max_trajectories=2)
            assert mock_table.add_data.call_count == 2
            
            # Verify wandb.log call with correct metadata
            mock_wandb_module.log.assert_called_once()
            log_args, log_kwargs = mock_wandb_module.log.call_args
            log_data = log_args[0]
            
            assert "test/trajectories" in log_data
            assert log_kwargs["step"] == 100
    
    def test_log_full_history_mode(self, sample_trajectories):
        """Test logging in full conversation history mode."""
        mock_wandb_module = MagicMock()
        mock_table = MagicMock()
        mock_wandb_module.Table.return_value = mock_table
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(
                log_full_history=True
            )
            
            logger.log(sample_trajectories, step=200, prefix="eval")
            
            # Verify table structure for full history mode
            table_kwargs = mock_wandb_module.Table.call_args[1]
            columns = table_kwargs["columns"]
            
            # Should have full conversation column, not separate prompt/response
            assert "full_conversation" in columns
            assert "prompt" not in columns
            
            # Verify correct prefix in log data
            log_args = mock_wandb_module.log.call_args[0]
            log_data = log_args[0]
            assert "eval/trajectories" in log_data


class TestCSVTrajectoryLogger:
    """Test CSV file trajectory logger functionality."""
    
    def test_initialization(self):
        """Test CSV logger initialization and directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVTrajectoryLogger(
                output_dir=tmpdir,
                max_trajectories=5
            )
            
            # Verify initialization parameters
            assert logger.output_dir == tmpdir
            assert logger.max_trajectories == 5
            assert os.path.exists(tmpdir)
    
    def test_log_trajectories_to_csv(self, sample_trajectories):
        """Test logging trajectories to CSV file with proper content validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVTrajectoryLogger(
                output_dir=tmpdir,
                max_trajectories=2
            )
            
            logger.log(sample_trajectories, step=100, prefix="train")
            
            # Verify CSV file creation
            expected_filename = "train_trajectories_step_100.csv"
            csv_file = os.path.join(tmpdir, expected_filename)
            assert os.path.exists(csv_file)
            
            # Verify CSV content and structure
            df = pd.read_csv(csv_file)
            assert len(df) == 2  # Limited by max_trajectories
            assert df["step"].iloc[0] == 100
            assert df["prefix"].iloc[0] == "train"
            
            # Verify required columns are present
            expected_columns = {"prompt", "response", "reward", "step", "prefix"}
            assert expected_columns.issubset(set(df.columns))
            
            # Verify logger state tracking
            assert logger.logged_count == 2


class TestTrajectoryLoggerFactory:
    """Test the create_trajectory_logger_from_config factory function."""
    
    def test_create_wandb_logger_from_config(self):
        """Test creating WandB logger from configuration."""
        
        config = DictConfig({
            "enabled": True,
            "type": "wandb",
            "max_trajectories": 5,
            "log_full_history": False
        })
        
        with patch.dict('sys.modules', {'wandb': MagicMock()}):
            logger = create_trajectory_logger_from_config(config)
            
            assert isinstance(logger, WandbTableTrajectoryLogger)
            assert logger.max_trajectories == 5
            assert logger.log_full_history is False
    
    def test_create_csv_logger_from_config(self):
        """Test creating CSV logger from configuration."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DictConfig({
                "enabled": True,
                "type": "csv",
                "output_dir": tmpdir,
                "max_trajectories": 20
            })
            
            logger = create_trajectory_logger_from_config(config)
            
            assert isinstance(logger, CSVTrajectoryLogger)
            assert logger.output_dir == tmpdir
            assert logger.max_trajectories == 20
    
    def test_disabled_config_returns_none(self):
        """Test that disabled configuration returns None."""
        
        config = DictConfig({
            "enabled": False,
            "type": "wandb"
        })
        
        logger = create_trajectory_logger_from_config(config)
        assert logger is None
    
    def test_unknown_logger_type_raises_error(self):
        """Test that unknown logger type raises ValueError."""
        
        config = DictConfig({
            "enabled": True,
            "type": "unknown_type"
        })
        
        with pytest.raises(ValueError, match="Unknown trajectory logger type"):
            create_trajectory_logger_from_config(config)