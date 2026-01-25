"""
uv  run --isolated --extra dev pytest tests/cpu/test_trainer.py
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from jaxtyping import Float, Integer
from pytest import approx
from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.utils import validate_batch_sizes
from skyrl_train.workers.worker import CriticWorkerBase, PolicyWorkerBase
from skyrl_train.workers.worker_utils import BatchIterator

from tests.cpu.util import example_dummy_config


@pytest.fixture
def dummy_config():
    return example_dummy_config()


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


@pytest.fixture
def dummy_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2

    # encode("abc") -> [97, 98, 99]
    mock_tokenizer.encode.side_effect = lambda x: [ord(c) for c in x]

    # tokenizer("abc") -> {"input_ids": [...], "attention_mask": [...]}
    def fake_tokenizer_call(text, **kwargs):
        ids = [ord(c) for c in text]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    mock_tokenizer.side_effect = fake_tokenizer_call

    return mock_tokenizer


@pytest.fixture
def dummy_generator():
    return MagicMock()


def _get_test_data(trainer: RayPPOTrainer):
    trainer.critic_model = MagicMock()  # pretend we're using a critic

    batch_size = 2
    total_seq_len = 5
    action_len = 3

    # Create test data
    ret_sequences: Float[torch.Tensor, "batch_size total_seq_len"] = torch.randint(0, 1000, (batch_size, total_seq_len))
    ret_attention_masks: Float[torch.Tensor, "batch_size total_seq_len"] = torch.ones((batch_size, total_seq_len))
    ret_loss_masks: Integer[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32), torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32)], dim=0
    )
    base_log_probs: Float[torch.Tensor, "batch_size total_seq_len"] = torch.log(
        torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2], [0.25, 0.25, 0.25, 0.15, 0.10]])
    )
    action_log_probs: Float[torch.Tensor, "batch_size total_seq_len"] = torch.log(
        torch.tensor([[0.1, 0.3, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.1, 0.1]])
    )
    action_masks: Integer[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32), torch.tensor([1, 1, 1, 1, 1], dtype=torch.int32)], dim=0
    )
    actual_response_lengths: Float[torch.Tensor, "batch_size"] = action_masks.sum(dim=-1).to(float)
    rewards_all: Float[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])], dim=0
    )
    values: Float[torch.Tensor, "batch_size action_len"] = torch.randn(batch_size, action_len)
    uids: np.ndarray[str] = np.array(["0", "0"])

    # Run method
    data = TrainingInputBatch(
        {
            "sequences": ret_sequences,
            "attention_mask": ret_attention_masks,
            "loss_mask": ret_loss_masks,
            "base_action_log_probs": base_log_probs,
            "action_log_probs": action_log_probs,
            "response_mask": action_masks,
            "rewards": rewards_all,
            "values": values,
        },
    )
    data.metadata = {
        "uids": uids,
        "response_length": action_len,
        "avg_response_length": actual_response_lengths.mean().item(),
    }
    data = trainer.apply_reward_kl_penalty(data)

    return data


def test_calculate_kl_create_experience_batched(dummy_config):
    trainer = RayPPOTrainer(
        cfg=dummy_config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        generator=dummy_generator,
    )
    data = _get_test_data(trainer)
    # Assertions
    metrics = data.metadata["metrics"]
    assert metrics["avg_kl_max"] == approx(0.3143, abs=1e-4)
    # Note; the raw KL mean is 0.054, but then the masked mean is different.
    assert metrics["avg_kl"] == approx(0.1249, abs=1e-4)


@patch("skyrl_train.utils.ppo_utils.compute_advantages_and_returns", new_callable=MagicMock)
def test_calc_advantages_and_returns(mock_compute_adv_and_ret, dummy_config):
    trainer = RayPPOTrainer(
        cfg=dummy_config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        generator=dummy_generator,
    )
    data = _get_test_data(trainer)

    # Mocked return values
    mock_advantages = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])
    mock_returns = torch.tensor([[0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]])

    # Set up mocks
    mock_compute_adv_and_ret.return_value = (mock_advantages, mock_returns)

    # Run the method
    data = trainer.compute_advantages_and_returns(data)
    metrics = data.metadata["metrics"]

    # Assertions
    assert torch.allclose(data["advantages"], mock_advantages)
    assert torch.allclose(data["returns"], mock_returns)
    assert isinstance(metrics, dict)
    assert "avg_final_rewards" in metrics
    assert "avg_response_length" in metrics
    assert "avg_advantages_abs" in metrics
    assert metrics["avg_advantages"] == approx(
        torch.masked_select(mock_advantages, data["response_mask"].bool()).mean().item(), rel=1e-5
    )


def test_normalize_mini_batch_size():
    """Test the _normalize_mini_batch_size method initializes micro batch tracking.

    Workers don't need to know mini batch sizes per GPU.
    They receive batches from the trainer and split them into micro batches.
    _normalize_mini_batch_size only initializes micro batch tracking for gradient scaling.

    # TODO: Update naming once Megatron is updated to not be aware of mini batch sizes.
    """

    # Create minimal worker instances for testing
    class TestPolicyWorker(PolicyWorkerBase):
        def init_model(self, *args, **kwargs):
            pass

        def offload_to_cpu(self, pin_memory=True, non_blocking=True):
            pass

        def backload_to_gpu(self, non_blocking=True):
            pass

        def _forward_micro_batch(self, micro_batch):
            pass

    class TestCriticWorker(CriticWorkerBase):
        def init_model(self, *args, **kwargs):
            pass

        def offload_to_cpu(self, pin_memory=True, non_blocking=True):
            pass

        def backload_to_gpu(self, non_blocking=True):
            pass

        def _forward_micro_batch(self, micro_batch):
            pass

    def create_policy_worker_with_config(
        train_batch_size, policy_mini_batch_size, micro_train_batch_size_per_gpu, n_samples_per_prompt, dp_size
    ):
        """Helper to create policy worker with specific config."""
        cfg = get_default_config()
        cfg.trainer.train_batch_size = train_batch_size
        cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
        cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        cfg.generator.n_samples_per_prompt = n_samples_per_prompt
        cfg.trainer.algorithm.policy_loss_type = "regular"

        worker = TestPolicyWorker(
            cfg=cfg,
            world_size=dp_size,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )

        # Mock mesh_rank
        worker.mesh_rank = MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1)

        return worker

    def create_critic_worker_with_config(
        train_batch_size, critic_mini_batch_size, micro_train_batch_size_per_gpu, n_samples_per_prompt, dp_size
    ):
        """Helper to create critic worker with specific config."""
        cfg = get_default_config()
        cfg.trainer.train_batch_size = train_batch_size
        cfg.trainer.critic_mini_batch_size = critic_mini_batch_size
        cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        cfg.generator.n_samples_per_prompt = n_samples_per_prompt

        worker = TestCriticWorker(
            cfg=cfg,
            world_size=dp_size,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )

        # Mock mesh_rank
        worker.mesh_rank = MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1)

        return worker

    # Test Case 1: PolicyWorker initializes _micro_batches_accumulated
    policy_worker = create_policy_worker_with_config(
        train_batch_size=128, policy_mini_batch_size=16, micro_train_batch_size_per_gpu=2, n_samples_per_prompt=1, dp_size=4
    )
    policy_worker._normalize_mini_batch_size()

    assert hasattr(policy_worker, "_micro_batches_accumulated")
    assert policy_worker._micro_batches_accumulated == 0

    # Test Case 2: Basic valid configuration for CriticWorker
    # In the new design, critic worker only initializes _micro_batches_accumulated
    critic_worker = create_critic_worker_with_config(
        train_batch_size=128,
        critic_mini_batch_size=8,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=2,
        dp_size=4,
    )
    critic_worker._normalize_mini_batch_size()

    # Verify micro_batches_accumulated is initialized
    assert critic_worker._micro_batches_accumulated == 0

    # Test Case 3: Single GPU (dp_size=1) for PolicyWorker
    policy_worker = create_policy_worker_with_config(
        train_batch_size=32, policy_mini_batch_size=8, micro_train_batch_size_per_gpu=2, n_samples_per_prompt=1, dp_size=1
    )
    policy_worker._normalize_mini_batch_size()

    assert hasattr(policy_worker, "_micro_batches_accumulated")
    assert policy_worker._micro_batches_accumulated == 0

    # Test Case 4: High n_samples_per_prompt for CriticWorker
    # In the new design, critic worker only initializes _micro_batches_accumulated
    critic_worker = create_critic_worker_with_config(
        train_batch_size=256,
        critic_mini_batch_size=32,
        micro_train_batch_size_per_gpu=8,
        n_samples_per_prompt=4,
        dp_size=2,
    )
    critic_worker._normalize_mini_batch_size()

    # Verify micro_batches_accumulated is initialized
    assert critic_worker._micro_batches_accumulated == 0

    # Test Case 5: Error case - mesh_rank not initialized
    policy_worker_no_mesh = create_policy_worker_with_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=1,
        dp_size=4,
    )
    policy_worker_no_mesh.mesh_rank = None

    with pytest.raises(RuntimeError, match="mesh_rank must be initialized"):
        policy_worker_no_mesh._normalize_mini_batch_size()


def test_validate_batch_sizes():
    """Test the validate_batch_sizes function with various configurations to trigger all error cases."""

    def create_test_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        critic_mini_batch_size=8,
        micro_train_batch_size_per_gpu=2,
        micro_forward_batch_size_per_gpu=4,
        n_samples_per_prompt=2,
        policy_num_nodes=1,
        policy_num_gpus_per_node=4,
        critic_num_nodes=1,
        critic_num_gpus_per_node=4,
        policy_sequence_parallel_size=1,
        critic_sequence_parallel_size=1,
        critic_model_path=None,
    ):
        """Helper to create config for validation testing."""
        cfg = get_default_config()
        cfg.trainer.train_batch_size = train_batch_size
        cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
        cfg.trainer.critic_mini_batch_size = critic_mini_batch_size
        cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        cfg.trainer.micro_forward_batch_size_per_gpu = micro_forward_batch_size_per_gpu
        cfg.trainer.placement.policy_num_nodes = policy_num_nodes
        cfg.trainer.placement.policy_num_gpus_per_node = policy_num_gpus_per_node
        cfg.trainer.placement.critic_num_nodes = critic_num_nodes
        cfg.trainer.placement.critic_num_gpus_per_node = critic_num_gpus_per_node
        cfg.trainer.policy.sequence_parallel_size = policy_sequence_parallel_size
        cfg.trainer.critic.model.path = critic_model_path
        cfg.trainer.critic.sequence_parallel_size = critic_sequence_parallel_size
        cfg.trainer.algorithm.use_kl_loss = False
        cfg.trainer.algorithm.use_kl_in_reward = False
        cfg.generator.n_samples_per_prompt = n_samples_per_prompt
        return cfg

    # Test Case 1: Valid configuration
    cfg = create_test_config()
    validate_batch_sizes(cfg)  # Should not raise any exceptions

    # Test Case 2: Error case - train_batch_size < policy_mini_batch_size
    cfg = create_test_config(train_batch_size=8, policy_mini_batch_size=16)
    with pytest.raises(AssertionError):
        validate_batch_sizes(cfg)

    # Test Case 3: Error case - train_batch_size < critic_mini_batch_size
    cfg = create_test_config(train_batch_size=4, critic_mini_batch_size=8)
    with pytest.raises(AssertionError):
        validate_batch_sizes(cfg)

    # Test Case 4: Error case - policy_mini_batch_size = 0
    cfg = create_test_config(policy_mini_batch_size=0)
    with pytest.raises(AssertionError, match="policy_mini_batch_size must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 5: Error case - critic_mini_batch_size = 0
    cfg = create_test_config(critic_mini_batch_size=0, critic_model_path="test")
    with pytest.raises(AssertionError, match="critic_mini_batch_size must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 6: Error case - micro_train_batch_size_per_gpu = 0
    cfg = create_test_config(micro_train_batch_size_per_gpu=0)
    with pytest.raises(AssertionError, match="micro_train_batch_size_per_gpu must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 7: Error case - micro_forward_batch_size_per_gpu = 0
    cfg = create_test_config(micro_forward_batch_size_per_gpu=0)
    with pytest.raises(AssertionError, match="micro_forward_batch_size_per_gpu must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 8: Error case - train_batch_size not divisible by (policy_mini_batch_size * policy_dp_size)
    cfg = create_test_config(train_batch_size=100, policy_mini_batch_size=16, policy_num_gpus_per_node=4)
    # Should fail because train_batch_size is not evenly divisible by policy batch requirements
    with pytest.raises(AssertionError, match="train_batch_size .* should be divisible by policy_mini_batch_size"):
        validate_batch_sizes(cfg)

    # Test Case 9: Error case - train_batch_size not divisible by (critic_mini_batch_size * critic_dp_size)
    cfg = create_test_config(
        train_batch_size=100,
        policy_mini_batch_size=5,
        critic_mini_batch_size=16,
        critic_num_gpus_per_node=4,
        critic_model_path="test",
    )
    # Should fail because train_batch_size is not evenly divisible by critic batch requirements
    with pytest.raises(AssertionError, match="train_batch_size .* should be divisible by critic_mini_batch_size"):
        validate_batch_sizes(cfg)

    # Test Case 10: Error case - policy_mini_batch_size_per_gpu not divisible by micro_train_batch_size_per_gpu
    cfg = create_test_config(
        policy_mini_batch_size=8, n_samples_per_prompt=1, policy_num_gpus_per_node=1, micro_train_batch_size_per_gpu=3
    )
    # Should fail because policy mini batch per GPU is not evenly divisible by micro batch size
    with pytest.raises(
        AssertionError,
        match="normalized policy_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 11: Error case - critic_mini_batch_size_per_gpu not divisible by micro_train_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=144,
        policy_mini_batch_size=12,  # Policy validation passes
        critic_mini_batch_size=8,  # Critic micro batch divisibility fails
        n_samples_per_prompt=1,
        critic_num_gpus_per_node=1,
        micro_train_batch_size_per_gpu=3,
        critic_model_path="test",
    )
    # Should fail because critic mini batch per GPU is not evenly divisible by micro batch size
    with pytest.raises(
        AssertionError,
        match="normalized critic_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 12: Valid configuration with sequence parallelism
    cfg = create_test_config(
        policy_sequence_parallel_size=2,
        critic_sequence_parallel_size=2,
        policy_num_gpus_per_node=8,
        critic_num_gpus_per_node=8,
    )
    validate_batch_sizes(cfg)  # Should not raise any exceptions

    # Test Case 13: Valid configuration - train_batch_size not divisible by (critic_mini_batch_size * critic_dp_size), but critic model path is None
    cfg = create_test_config(
        train_batch_size=100,
        policy_mini_batch_size=5,
        critic_mini_batch_size=16,
        critic_num_gpus_per_node=4,
        critic_model_path=None,
    )
    validate_batch_sizes(cfg)

    # Test Case 14: Valid configuration - critic_mini_batch_size is invalid but critic model is not specified
    cfg = create_test_config(critic_mini_batch_size=0, critic_model_path=None)
    validate_batch_sizes(cfg)

    # Test Case 15: Error case - train_batch_size_per_gpu not divisible by policy_mini_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=10,
        policy_mini_batch_size=5,
        policy_num_gpus_per_node=2,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    with pytest.raises(
        AssertionError, match="policy_train_batch_size_per_gpu .* should be divisible by policy_mini_batch_size_per_gpu"
    ):
        validate_batch_sizes(cfg)

    # Test Case 16: Error case - train_batch_size_per_gpu not divisible by critic_mini_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=10,
        policy_mini_batch_size=10,
        policy_num_gpus_per_node=1,
        critic_mini_batch_size=5,
        critic_num_gpus_per_node=2,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
        critic_model_path="test",
    )
    with pytest.raises(
        AssertionError, match="critic_train_batch_size_per_gpu .* should be divisible by critic_mini_batch_size_per_gpu"
    ):
        validate_batch_sizes(cfg)


def test_validate_batch_sizes_lcm_dp_requirement():
    """Ensure train_batch_size is >= lcm(policy_dp, ref_dp) when ref is used; else >= policy_dp."""

    def create_config(train_batch_size, policy_dp, ref_dp, include_ref=True):
        cfg = get_default_config()
        cfg.trainer.train_batch_size = train_batch_size
        cfg.trainer.policy_mini_batch_size = train_batch_size
        cfg.trainer.critic_mini_batch_size = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.placement.policy_num_nodes = 1
        cfg.trainer.placement.policy_num_gpus_per_node = policy_dp
        cfg.trainer.placement.ref_num_nodes = 1
        cfg.trainer.placement.ref_num_gpus_per_node = ref_dp if include_ref else 1
        cfg.trainer.placement.critic_num_nodes = 1
        cfg.trainer.placement.critic_num_gpus_per_node = 1
        cfg.trainer.policy.sequence_parallel_size = 1
        cfg.trainer.ref.sequence_parallel_size = 1
        cfg.trainer.critic.model.path = None
        cfg.trainer.critic.sequence_parallel_size = 1
        cfg.trainer.algorithm.use_kl_loss = include_ref
        cfg.trainer.algorithm.use_kl_in_reward = False
        cfg.trainer.algorithm.policy_loss_type = "regular"
        cfg.generator.n_samples_per_prompt = 1
        return cfg

    # Fail: lcm(2, 3) = 6, but train_batch_size = 5 when ref is used
    cfg = create_config(train_batch_size=5, policy_dp=2, ref_dp=3, include_ref=True)
    with pytest.raises(
        AssertionError,
        match=r"least common multiple of the data parallel sizes",
    ):
        validate_batch_sizes(cfg)

    # Pass: train_batch_size equals lcm(2, 3) = 6 when ref is used
    cfg = create_config(train_batch_size=6, policy_dp=2, ref_dp=3, include_ref=True)
    validate_batch_sizes(cfg)

    # Pass: ref disabled -> requirement reduces to policy_dp. With policy_dp=2, tbs=2 is valid.
    cfg = create_config(train_batch_size=2, policy_dp=2, ref_dp=3, include_ref=False)
    validate_batch_sizes(cfg)


def test_convert_tinker_loss_config():
    """Test that Tinker absolute ratio bounds are correctly converted to SkyRL offsets."""
    # Tinker uses absolute bounds: [0.9, 1.1]
    # SkyRL uses offsets from 1.0: eps_clip_low=0.1, eps_clip_high=0.1
    tinker_config = {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
    skyrl_config = PolicyWorkerBase.convert_tinker_loss_config(tinker_config)

    assert skyrl_config["eps_clip_low"] == approx(0.1)
    assert skyrl_config["eps_clip_high"] == approx(0.1)

    # Test asymmetric bounds
    tinker_config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    skyrl_config = PolicyWorkerBase.convert_tinker_loss_config(tinker_config)

    assert skyrl_config["eps_clip_low"] == approx(0.2)
    assert skyrl_config["eps_clip_high"] == approx(0.2)

    # Test passthrough of unknown keys
    tinker_config = {"clip_low_threshold": 0.9, "some_other_key": 42}
    skyrl_config = PolicyWorkerBase.convert_tinker_loss_config(tinker_config)

    assert skyrl_config["eps_clip_low"] == approx(0.1)
    assert skyrl_config["some_other_key"] == 42
