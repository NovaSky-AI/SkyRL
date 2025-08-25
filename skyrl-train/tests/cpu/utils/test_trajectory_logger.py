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
from skyrl_train.utils.trajectory_logger import (
    CSVTrajectoryLogger,
    Trajectory,
    TrajectoryLogger,
    WandbTableTrajectoryLogger,
    create_trajectory_logger_from_config,
)


# Test Fixtures
@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with standard decode/encode behavior."""
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded text"
    tokenizer.encode.return_value = [1, 2, 3, 4]
    tokenizer.eos_token_id = 4
    return tokenizer


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
        prompt_tokens=[1, 2, 3, 4, 5],
        response_tokens=[10, 11, 12, 13, 14, 15],
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
            prompt_tokens=[1, 2, 3, 4, 5],
            response_tokens=[10, 11, 12, 13, 14, 15],
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
    
    def test_truncate_text_no_truncation(self, concrete_logger):
        """Test text truncation when text is within limits."""
        short_text = "Hello world"
        result = concrete_logger._truncate_text(short_text, 20)
        assert result == short_text
    
    def test_truncate_text_with_truncation(self, concrete_logger):
        """Test text truncation when text exceeds limits."""
        long_text = "a" * 100
        truncated = concrete_logger._truncate_text(long_text, 10)
        assert len(truncated) == 10
        assert truncated.endswith("...")
    
    def test_detokenize_with_tokenizer(self, concrete_logger, mock_tokenizer):
        """Test detokenization using provided tokenizer."""
        logger = type(concrete_logger)(tokenizer=mock_tokenizer)
        result = logger._detokenize([1, 2, 3])
        
        assert result == "decoded text"
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)
    
    def test_detokenize_without_tokenizer(self, concrete_logger):
        """Test detokenization fallback when no tokenizer provided."""
        logger = type(concrete_logger)(tokenizer=None)
        result = logger._detokenize([1, 2, 3])
        assert result == "<3 tokens>"
    
    def test_to_dataframe_conversion(self, concrete_logger, sample_trajectories):
        """Test conversion of trajectories to pandas DataFrame."""
        df = concrete_logger.to_dataframe(sample_trajectories)
        
        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_trajectories)
        
        # Check required columns exist
        expected_columns = {"prompt", "response", "reward", "stop_reason", "env_class", "chat_turns", "response_length"}
        assert expected_columns.issubset(set(df.columns))
        
        # Validate specific data values
        assert df["reward"].iloc[0] == 0.0
        assert df["reward"].iloc[1] == 0.5
        assert df["stop_reason"].iloc[2] == "length"


class TestWandbTableTrajectoryLogger:
    """Test WandB table trajectory logger with proper API mocking."""
    
    def test_initialization_success(self, mock_tokenizer):
        """Test successful WandB logger initialization with all parameters."""
        mock_wandb_module = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(
                tokenizer=mock_tokenizer,
                max_trajectories=5,
                max_text_length=100,
                log_full_history=True
            )
            
            # Verify initialization parameters
            assert logger.tokenizer == mock_tokenizer
            assert logger.max_trajectories == 5
            assert logger.max_text_length == 100
            assert logger.log_full_history is True
            assert logger.wandb == mock_wandb_module
    
    def test_initialization_fails_without_wandb(self, mock_tokenizer):
        """Test that initialization fails gracefully when wandb is unavailable."""
        with patch.dict('sys.modules', {'wandb': None}):
            with pytest.raises(ImportError, match="wandb is required"):
                WandbTableTrajectoryLogger(tokenizer=mock_tokenizer)
    
    def test_log_prompt_response_mode(self, sample_trajectories, mock_tokenizer):
        """Test logging in prompt/response mode with trajectory limit verification."""
        mock_wandb_module = MagicMock()
        mock_table = MagicMock()
        mock_wandb_module.Table.return_value = mock_table
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(
                tokenizer=mock_tokenizer,
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
    
    def test_log_full_history_mode(self, sample_trajectories, mock_tokenizer):
        """Test logging in full conversation history mode."""
        mock_wandb_module = MagicMock()
        mock_table = MagicMock()
        mock_wandb_module.Table.return_value = mock_table
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(
                tokenizer=mock_tokenizer,
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
    
    def test_initialization(self, mock_tokenizer):
        """Test CSV logger initialization and directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVTrajectoryLogger(
                output_dir=tmpdir,
                tokenizer=mock_tokenizer,
                max_trajectories=5
            )
            
            # Verify initialization parameters
            assert logger.output_dir == tmpdir
            assert logger.tokenizer == mock_tokenizer
            assert logger.max_trajectories == 5
            assert os.path.exists(tmpdir)
    
    def test_log_trajectories_to_csv(self, sample_trajectories, mock_tokenizer):
        """Test logging trajectories to CSV file with proper content validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVTrajectoryLogger(
                output_dir=tmpdir,
                tokenizer=mock_tokenizer,
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
    
    def test_create_wandb_logger_from_config(self, mock_tokenizer):
        """Test creating WandB logger from configuration."""
        
        config = DictConfig({
            "enabled": True,
            "type": "wandb",
            "max_trajectories": 5,
            "max_text_length": 1000,
            "log_full_history": True
        })
        
        with patch.dict('sys.modules', {'wandb': MagicMock()}):
            logger = create_trajectory_logger_from_config(config, mock_tokenizer)
            
            assert isinstance(logger, WandbTableTrajectoryLogger)
            assert logger.tokenizer == mock_tokenizer
            assert logger.max_trajectories == 5
            assert logger.max_text_length == 1000
            assert logger.log_full_history is True
    
    def test_create_csv_logger_from_config(self, mock_tokenizer):
        """Test creating CSV logger from configuration."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DictConfig({
                "enabled": True,
                "type": "csv",
                "output_dir": tmpdir,
                "max_trajectories": 20
            })
            
            logger = create_trajectory_logger_from_config(config, mock_tokenizer)
            
            assert isinstance(logger, CSVTrajectoryLogger)
            assert logger.tokenizer == mock_tokenizer
            assert logger.output_dir == tmpdir
            assert logger.max_trajectories == 20
    
    def test_disabled_config_returns_none(self, mock_tokenizer):
        """Test that disabled configuration returns None."""
        
        config = DictConfig({
            "enabled": False,
            "type": "wandb"
        })
        
        logger = create_trajectory_logger_from_config(config, mock_tokenizer)
        assert logger is None
    
    def test_unknown_logger_type_raises_error(self, mock_tokenizer):
        """Test that unknown logger type raises ValueError."""
        
        config = DictConfig({
            "enabled": True,
            "type": "unknown_type"
        })
        
        with pytest.raises(ValueError, match="Unknown trajectory logger type"):
            create_trajectory_logger_from_config(config, mock_tokenizer)