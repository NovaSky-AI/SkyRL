"""
Integration tests for trajectory logging feature.

Tests the complete trajectory logging workflow with real configurations:
- Configuration loading and parsing
- Trainer integration with trajectory logging
- Generator trajectory collection
- End-to-end logging functionality

Run with: uv run --extra dev --isolated pytest tests/cpu/test_trajectory_logging_integration.py
"""

import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

import hydra
import pandas as pd
import pytest
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.generators.base import GeneratorInput
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.generators.trajectory_logger import create_trajectory_logger_from_config
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


@pytest.fixture
def base_config() -> DictConfig:
    """Load base configuration for testing."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    
    # Override for CPU testing
    cfg.trainer.placement.policy_num_gpus_per_node = 0
    cfg.trainer.placement.critic_num_gpus_per_node = 0
    cfg.trainer.train_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.epochs = 1
    cfg.trainer.logger = "console"
    cfg.generator.n_samples_per_prompt = 1
    
    return cfg


@pytest.fixture
def trajectory_logging_config(base_config) -> DictConfig:
    """Configure trajectory logging in the base config."""
    base_config.generator.trajectory_logging.enabled = True
    base_config.generator.trajectory_logging.type = "csv"
    base_config.generator.trajectory_logging.max_trajectories = 5
    base_config.generator.trajectory_logging.log_interval = 1
    base_config.generator.trajectory_logging.output_dir = "./test_trajectory_logs"
    return base_config


@pytest.fixture
def mock_env_cfg():
    """Create mock environment configuration."""
    cfg = MagicMock()
    cfg.max_env_workers = 0
    cfg.env_class = "gsm8k"
    return cfg


@pytest.fixture
def mock_tokenizer():
    """Create a realistic mock tokenizer."""
    tokenizer = MagicMock()
    
    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
                return [[1, 2, 3, 4] for _ in messages]
            else:
                return [1, 2, 3, 4]
        else:
            if isinstance(messages, list) and isinstance(messages[0], dict):
                return " ".join([msg.get("content", "") for msg in messages])
            return str(messages)
    
    def mock_decode(tokens, **kwargs):
        return f"decoded_{len(tokens)}_tokens"
    
    def mock_tokenizer_call(text):
        return {"input_ids": [1, 2, 3, 4]}
    
    tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    tokenizer.decode.side_effect = mock_decode
    tokenizer.side_effect = mock_tokenizer_call
    tokenizer.eos_token_id = 4
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    
    return tokenizer


@pytest.fixture
def mock_inference_engine():
    """Create a mock inference engine."""
    engine = MagicMock()
    
    async def mock_generate(prompts, **kwargs):
        return {
            "responses": [f"Generated response {i}" for i in range(len(prompts))],
            "stop_reasons": ["stop"] * len(prompts)
        }
    
    engine.generate = AsyncMock(side_effect=mock_generate)
    return engine


@pytest.fixture 
def mock_env():
    """Create a mock SkyRL environment."""
    env = MagicMock()
    
    # Set return values instead of side_effect
    env.init.return_value = ([{"role": "user", "content": "Test question"}], {"test_param": "value"})
    
    def mock_step(response):
        return BaseTextEnvStepOutput(
            observations=[{"role": "assistant", "content": response}],
            reward=1.0,
            done=True,
            metadata={"success": True}
        )
    
    env.step.side_effect = mock_step
    return env


class TestTrajectoryLoggingIntegration:
    """Integration tests for trajectory logging feature."""
    
    def test_config_loading(self, trajectory_logging_config):
        """Test that trajectory logging configuration is properly loaded."""
        cfg = trajectory_logging_config
        
        # Verify trajectory logging config is present
        assert hasattr(cfg.generator, 'trajectory_logging')
        assert cfg.generator.trajectory_logging.enabled is True
        assert cfg.generator.trajectory_logging.type == "csv"
        assert cfg.generator.trajectory_logging.max_trajectories == 5
        assert cfg.generator.trajectory_logging.log_interval == 1
    
    @patch("skyrl_gym.make")
    def test_generator_trajectory_collection(
        self, mock_make, trajectory_logging_config, mock_tokenizer, 
        mock_inference_engine, mock_env, mock_env_cfg
    ):
        """Test that SkyRLGymGenerator collects trajectories when configured."""
        mock_make.return_value = mock_env
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config with test directory
            trajectory_logging_config.generator.trajectory_logging.output_dir = tmpdir
            
            # Create generator with trajectory logging from config
            generator = SkyRLGymGenerator(
                generator_cfg=trajectory_logging_config.generator,
                skyrl_gym_cfg=mock_env_cfg,
                inference_engine_client=mock_inference_engine,
                tokenizer=mock_tokenizer,
                model_name="test_model",
            )
            
            # Verify trajectory logger was created
            assert generator.trajectory_logger is not None
            assert generator.collect_trajectories is True
    
    @pytest.mark.asyncio
    @patch("skyrl_gym.make")
    async def test_end_to_end_trajectory_logging(
        self, mock_make, trajectory_logging_config, mock_tokenizer,
        mock_inference_engine, mock_env, mock_env_cfg
    ):
        """Test complete end-to-end trajectory logging workflow."""
        mock_make.return_value = mock_env
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config with test directory
            trajectory_logging_config.generator.trajectory_logging.output_dir = tmpdir
            
            # Create generator with dependency injection
            generator = SkyRLGymGenerator(
                generator_cfg=trajectory_logging_config.generator,
                skyrl_gym_cfg=mock_env_cfg,
                inference_engine_client=mock_inference_engine,
                tokenizer=mock_tokenizer,
                model_name="test_model",
            )
            
            # Generate trajectories
            input_batch: GeneratorInput = {
                "prompts": [
                    [{"role": "user", "content": "What is machine learning?"}],
                    [{"role": "user", "content": "Explain neural networks"}]
                ],
                "env_extras": [{"topic": "ML"}, {"topic": "DL"}],
                "env_classes": ["test_env", "test_env"],
            }
            
            # Execute generation - this should automatically trigger trajectory logging
            # since log_interval=1 in our test config
            generator_output = await generator.generate(input_batch)
            
            # Verify trajectory logging worked by checking for CSV file
            # The _handle_trajectory_logging method should have created the file automatically
            csv_file = os.path.join(tmpdir, "generation_trajectories_step_1.csv")
            assert os.path.exists(csv_file), f"Expected trajectory file not found at {csv_file}"
            
            # Verify CSV content contains expected trajectory data
            df = pd.read_csv(csv_file)
            
            # Should have 2 trajectories from the input batch
            assert len(df) == 2
            
            # Verify structure and metadata
            assert "prompt" in df.columns
            assert "response" in df.columns 
            assert "reward" in df.columns
            assert "step" in df.columns
            assert "env_class" in df.columns
            assert all(df["step"] == 1)  # batch_count starts at 1
            assert all(df["prefix"] == "generation")  # hardcoded in _handle_trajectory_logging
            assert all(df["env_class"] == "test_env")
            assert all(df["reward"] == 1.0)
            
            # Verify trajectory content
            prompts = df["prompt"].tolist()
            assert any("machine learning" in prompt.lower() for prompt in prompts)
            assert any("neural networks" in prompt.lower() for prompt in prompts)
    
    def test_trainer_trajectory_logging_integration(self, trajectory_logging_config, mock_tokenizer):
        """Test that RayPPOTrainer integrates with trajectory logging configuration."""
        # Create minimal dataset for trainer
        class DummyDataset:
            def __len__(self):
                return 2
            
            def __getitem__(self, idx):
                return f"sample_{idx}"
            
            def collate_fn(self, batch):
                return batch
        
        # Create mock generator
        mock_generator = MagicMock()
        mock_generator.trajectory_batch = []
        
        # Create trainer
        trainer = RayPPOTrainer(
            cfg=trajectory_logging_config,
            tracker=None,
            tokenizer=mock_tokenizer,
            train_dataset=DummyDataset(),
            eval_dataset=None,
            inference_engine_client=None,
            generator=mock_generator,
        )
        
        # Verify trajectory logging configuration is accessible
        assert hasattr(trainer.cfg.generator, 'trajectory_logging')
        assert trainer.cfg.generator.trajectory_logging.enabled is True
        
        # Test that trajectory logging is properly configured
        traj_config = trainer.cfg.generator.trajectory_logging
        assert traj_config.type == "csv"
        assert traj_config.max_trajectories == 5
        assert traj_config.log_interval == 1
    
    @patch("skyrl_gym.make")
    def test_trajectory_logging_disabled(
        self, mock_make, base_config, mock_tokenizer, mock_inference_engine, mock_env, mock_env_cfg
    ):
        """Test that trajectory logging can be disabled via configuration."""
        mock_make.return_value = mock_env
        
        # Ensure trajectory logging is disabled
        base_config.generator.trajectory_logging.enabled = False
        
        # Create generator without trajectory logger
        generator = SkyRLGymGenerator(
            generator_cfg=base_config.generator,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_inference_engine,
            tokenizer=mock_tokenizer,
            model_name="test_model",
        )
        
        # Verify no trajectory collection
        assert generator.trajectory_logger is None
        assert generator.collect_trajectories is False