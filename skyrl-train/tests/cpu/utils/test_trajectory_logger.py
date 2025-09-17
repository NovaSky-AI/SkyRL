"""
Unit tests for trajectory logging functionality.

This module tests the simplified TrajectoryLogger system with prompts, responses, and rewards.

Run with: uv run --extra dev --isolated pytest tests/cpu/utils/test_trajectory_logger.py
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig
import pandas as pd
import pytest

from skyrl_train.generators.trajectory_logger import (
    CSVTrajectoryLogger,
    TrajectoryLogger,
    WandbTableTrajectoryLogger,
    create_trajectory_logger_from_config,
)


# Test Fixtures
@pytest.fixture
def sample_data():
    """Create simple sample trajectory data."""
    return {
        "prompts": ["What is 2+2?", "What is 3+3?", "What is 4+4?"],
        "responses": ["The answer is 4.", "The answer is 6.", "The answer is 8."],
        "rewards": [1.0, 0.8, 0.5]
    }


@pytest.fixture
def concrete_logger():
    """Create a concrete implementation of TrajectoryLogger for testing."""
    class ConcreteLogger(TrajectoryLogger):
        def log(self, prompts, responses, rewards):
            pass
    return ConcreteLogger()


class TestWandbTableTrajectoryLogger:
    """Test WandB table trajectory logger."""
    
    def test_initialization(self):
        """Test WandB logger initialization."""
        mock_wandb_module = MagicMock()
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(max_trajectories=5)
            
            assert logger.max_trajectories == 5
            assert logger.wandb == mock_wandb_module
    
    def test_log_trajectories(self, sample_data):
        """Test logging trajectories to wandb."""
        mock_wandb_module = MagicMock()
        mock_table = MagicMock()
        mock_wandb_module.Table.return_value = mock_table
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(max_trajectories=2)
            
            logger.log(
                sample_data["prompts"],
                sample_data["responses"], 
                sample_data["rewards"]
            )
            
            # Verify table creation
            mock_wandb_module.Table.assert_called_once_with(
                columns=["step", "prompt", "response", "reward"]
            )
            
            # Verify trajectory limiting (max_trajectories=2)
            assert mock_table.add_data.call_count == 2
            
            # Verify wandb.log call with fixed prefix
            mock_wandb_module.log.assert_called_once()
            log_args = mock_wandb_module.log.call_args[0][0]
            assert "trajectories" in log_args
    
    def test_unlimited_trajectories(self, sample_data):
        """Test logging with unlimited trajectories."""
        mock_wandb_module = MagicMock()
        mock_table = MagicMock()
        mock_wandb_module.Table.return_value = mock_table
        
        with patch.dict('sys.modules', {'wandb': mock_wandb_module}):
            logger = WandbTableTrajectoryLogger(max_trajectories=-1)
            
            logger.log(
                sample_data["prompts"],
                sample_data["responses"],
                sample_data["rewards"]
            )
            
            # Should log all 3 trajectories
            assert mock_table.add_data.call_count == 3


class TestCSVTrajectoryLogger:
    """Test CSV trajectory logger."""
    
    def test_initialization(self):
        """Test CSV logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVTrajectoryLogger(output_dir=tmpdir, max_trajectories=5)
            
            assert logger.output_dir == tmpdir
            assert logger.max_trajectories == 5
            assert os.path.exists(tmpdir)
    
    def test_log_trajectories_to_csv(self, sample_data):
        """Test logging trajectories to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVTrajectoryLogger(output_dir=tmpdir, max_trajectories=2)
            
            logger.log(
                sample_data["prompts"],
                sample_data["responses"],
                sample_data["rewards"]
            )
            
            # Verify CSV file creation
            csv_file = os.path.join(tmpdir, "trajectories.csv")
            assert os.path.exists(csv_file)
            
            # Verify CSV content
            df = pd.read_csv(csv_file)
            assert len(df) == 2  # Limited by max_trajectories
            
            # Verify step values are indices (0, 1)
            assert list(df["step"]) == [0, 1]
            assert "What is 2+2?" in df["prompt"].values
            assert "The answer is 4." in df["response"].values
            assert 1.0 in df["reward"].values


class TestTrajectoryLoggerFactory:
    """Test the trajectory logger factory function."""
    
    def test_create_wandb_logger(self):
        """Test creating WandB logger from configuration."""
        config = DictConfig({
            "enabled": True,
            "type": "wandb",
            "max_trajectories": 5
        })
        
        with patch.dict('sys.modules', {'wandb': MagicMock()}):
            logger = create_trajectory_logger_from_config(config)
            
            assert isinstance(logger, WandbTableTrajectoryLogger)
            assert logger.max_trajectories == 5
    
    def test_create_csv_logger(self):
        """Test creating CSV logger from configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DictConfig({
                "enabled": True,
                "type": "csv",
                "output_dir": tmpdir,
                "max_trajectories": 10
            })
            
            logger = create_trajectory_logger_from_config(config)
            
            assert isinstance(logger, CSVTrajectoryLogger)
            assert logger.output_dir == tmpdir
            assert logger.max_trajectories == 10
    
    def test_disabled_config_assertion(self):
        """Test that disabled configuration raises assertion error."""
        config = DictConfig({
            "enabled": False,
            "type": "wandb"
        })
        
        with pytest.raises(AssertionError):
            create_trajectory_logger_from_config(config)
    
    def test_unknown_logger_type_error(self):
        """Test that unknown logger type raises ValueError."""
        config = DictConfig({
            "enabled": True,
            "type": "unknown"
        })
        
        with pytest.raises(ValueError, match="Unknown trajectory logger type"):
            create_trajectory_logger_from_config(config)