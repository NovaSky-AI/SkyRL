"""Unit tests for TinkerTrainingAdapter."""

import pytest
from unittest.mock import MagicMock

from skyrl_train.training.tinker_adapter import (
    TinkerTrainingAdapter,
    ForwardBackwardOutput,
)


class TestForwardBackwardOutput:
    """Tests for ForwardBackwardOutput."""

    def test_init(self):
        """Test ForwardBackwardOutput initialization."""
        loss_fn_outputs = [
            {"logprobs": [0.1, 0.2, 0.3]},
            {"logprobs": [0.4, 0.5]},
        ]
        metrics = {"loss": 0.5, "clip_ratio": 0.1}

        result = ForwardBackwardOutput(
            loss_fn_outputs=loss_fn_outputs,
            metrics=metrics,
        )

        assert len(result.loss_fn_outputs) == 2
        assert result.metrics["loss"] == 0.5

    def test_to_dict(self):
        """Test ForwardBackwardOutput.to_dict()."""
        loss_fn_outputs = [{"logprobs": [0.1]}]
        metrics = {"loss": 0.5}

        result = ForwardBackwardOutput(
            loss_fn_outputs=loss_fn_outputs,
            metrics=metrics,
        )

        d = result.to_dict()
        assert d["loss_fn_outputs"] == loss_fn_outputs
        assert d["metrics"] == metrics


class TestTinkerTrainingAdapter:
    """Tests for TinkerTrainingAdapter."""

    def test_init(self):
        """Test TinkerTrainingAdapter initialization."""
        mock_dispatch = MagicMock()
        adapter = TinkerTrainingAdapter(mock_dispatch)

        assert adapter.worker_dispatch == mock_dispatch
        assert adapter.model_name == "policy"

    def test_init_custom_model_name(self):
        """Test initialization with custom model name."""
        mock_dispatch = MagicMock()
        adapter = TinkerTrainingAdapter(mock_dispatch, model_name="custom_model")

        assert adapter.model_name == "custom_model"

    def test_extract_tokens_from_model_input_flat(self):
        """Test extracting tokens from flat model input."""
        model_input = {"tokens": [1, 2, 3, 4, 5]}

        tokens = TinkerTrainingAdapter.extract_tokens_from_model_input(model_input)

        assert tokens == [1, 2, 3, 4, 5]

    def test_extract_tokens_from_model_input_chunked(self):
        """Test extracting tokens from chunked model input."""
        model_input = {
            "chunks": [
                {"tokens": [1, 2, 3]},
                {"tokens": [4, 5]},
            ]
        }

        tokens = TinkerTrainingAdapter.extract_tokens_from_model_input(model_input)

        assert tokens == [1, 2, 3, 4, 5]

    def test_extract_tokens_from_model_input_empty(self):
        """Test extracting tokens from empty model input."""
        model_input = {"chunks": []}

        tokens = TinkerTrainingAdapter.extract_tokens_from_model_input(model_input)

        assert tokens == []

    def test_convert_data_to_batch_cross_entropy(self):
        """Test converting cross-entropy data to batch."""
        mock_dispatch = MagicMock()
        adapter = TinkerTrainingAdapter(mock_dispatch)

        data = [
            {
                "model_input": {"tokens": [1, 2, 3]},
                "loss_fn_inputs": {
                    "target_tokens": [2, 3, 4],
                    "weights": [0, 1, 1],
                },
            },
            {
                "model_input": {"tokens": [5, 6]},
                "loss_fn_inputs": {
                    "target_tokens": [6, 7],
                    "weights": [1, 1],
                },
            },
        ]

        batch = adapter._convert_data_to_batch(data, "cross_entropy")

        # Check batch size and sequence length
        assert batch["sequences"].shape == (2, 3)  # max_len is 3
        assert batch["attention_mask"].shape == (2, 3)
        assert batch["loss_mask"].shape == (2, 3)

        # Check first sequence (no padding needed)
        assert batch["sequences"][0].tolist() == [1, 2, 3]
        assert batch["attention_mask"][0].tolist() == [1, 1, 1]
        assert batch["loss_mask"][0].tolist() == [0, 1, 1]

        # Check second sequence (left-padded)
        assert batch["sequences"][1].tolist() == [0, 5, 6]
        assert batch["attention_mask"][1].tolist() == [0, 1, 1]
        assert batch["loss_mask"][1].tolist() == [0, 1, 1]

    def test_convert_data_to_batch_importance_sampling(self):
        """Test converting importance sampling data to batch."""
        mock_dispatch = MagicMock()
        adapter = TinkerTrainingAdapter(mock_dispatch)

        data = [
            {
                "model_input": {"tokens": [1, 2, 3]},
                "loss_fn_inputs": {
                    "target_tokens": [2, 3, 4],
                    "logprobs": [-0.1, -0.2, -0.3],
                    "advantages": [0.5, 1.0, 0.8],
                },
            },
        ]

        batch = adapter._convert_data_to_batch(data, "importance_sampling")

        assert batch["action_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, -0.3])
        assert batch["advantages"][0].tolist() == pytest.approx([0.5, 1.0, 0.8])

    def test_convert_data_to_batch_empty_raises(self):
        """Test that empty data raises ValueError."""
        mock_dispatch = MagicMock()
        adapter = TinkerTrainingAdapter(mock_dispatch)

        with pytest.raises(ValueError, match="Data list cannot be empty"):
            adapter._convert_data_to_batch([], "cross_entropy")

    @pytest.mark.asyncio
    async def test_forward_backward_calls_dispatch(self):
        """Test that forward_backward calls WorkerDispatch correctly."""
        mock_dispatch = MagicMock()
        mock_dispatch.forward_backward.return_value = {
            "loss": 0.5,
            "clip_ratio": 0.1,
        }

        adapter = TinkerTrainingAdapter(mock_dispatch)

        data = [
            {
                "model_input": {"tokens": [1, 2, 3]},
                "loss_fn_inputs": {
                    "target_tokens": [2, 3, 4],
                    "weights": [0, 1, 1],
                },
            },
        ]

        result = await adapter.forward_backward(data, "cross_entropy")

        # Verify dispatch was called
        mock_dispatch.forward_backward.assert_called_once()
        call_args = mock_dispatch.forward_backward.call_args
        assert call_args[0][0] == "policy"  # model_name

        # Verify batch was passed
        batch = call_args[0][1]
        assert batch.metadata["loss_fn"] == "cross_entropy"

        # Verify result
        assert isinstance(result, ForwardBackwardOutput)
        assert result.metrics["loss"] == 0.5

    @pytest.mark.asyncio
    async def test_forward_backward_ppo(self):
        """Test forward_backward with PPO loss."""
        mock_dispatch = MagicMock()
        mock_dispatch.forward_backward.return_value = {
            "loss": 0.3,
            "clip_ratio": 0.15,
        }

        adapter = TinkerTrainingAdapter(mock_dispatch)

        data = [
            {
                "model_input": {"tokens": [1, 2, 3]},
                "loss_fn_inputs": {
                    "target_tokens": [2, 3, 4],
                    "logprobs": [-0.1, -0.2, -0.3],
                    "advantages": [0.5, 1.0, 0.8],
                },
            },
        ]

        result = await adapter.forward_backward(
            data,
            "ppo",
            loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
        )

        # Verify batch metadata
        batch = mock_dispatch.forward_backward.call_args[0][1]
        assert batch.metadata["loss_fn"] == "regular"  # SkyRL's name for PPO
        assert batch.metadata["loss_fn_config"]["clip_low_threshold"] == 0.9

        # Verify result
        assert isinstance(result, ForwardBackwardOutput)
        assert result.metrics["loss"] == 0.3

    @pytest.mark.asyncio
    async def test_forward_backward_unsupported_loss(self):
        """Test that unsupported loss function raises error."""
        mock_dispatch = MagicMock()
        adapter = TinkerTrainingAdapter(mock_dispatch)

        data = [{"model_input": {"tokens": [1]}, "loss_fn_inputs": {}}]

        with pytest.raises(ValueError, match="Unsupported loss function"):
            await adapter.forward_backward(data, "unknown_loss")

    @pytest.mark.asyncio
    async def test_optim_step_calls_dispatch(self):
        """Test that optim_step calls WorkerDispatch correctly."""
        mock_dispatch = MagicMock()
        mock_dispatch.optim_step.return_value = 1.5  # grad_norm

        adapter = TinkerTrainingAdapter(mock_dispatch)

        grad_norm = await adapter.optim_step(learning_rate=1e-4)

        mock_dispatch.optim_step.assert_called_once_with("policy")
        assert grad_norm == 1.5

    @pytest.mark.asyncio
    async def test_optim_step_no_learning_rate(self):
        """Test optim_step without learning rate."""
        mock_dispatch = MagicMock()
        mock_dispatch.optim_step.return_value = None

        adapter = TinkerTrainingAdapter(mock_dispatch)

        grad_norm = await adapter.optim_step()

        mock_dispatch.optim_step.assert_called_once_with("policy")
        assert grad_norm is None

    def test_loss_fn_map(self):
        """Test that loss function map contains expected entries."""
        assert TinkerTrainingAdapter.LOSS_FN_MAP["cross_entropy"] == "cross_entropy"
        assert TinkerTrainingAdapter.LOSS_FN_MAP["importance_sampling"] == "importance_sampling"
        assert TinkerTrainingAdapter.LOSS_FN_MAP["ppo"] == "regular"
