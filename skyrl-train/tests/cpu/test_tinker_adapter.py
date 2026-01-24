"""Unit tests for TinkerInferenceAdapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from skyrl_train.inference_engines.tinker_adapter import (
    TinkerInferenceAdapter,
    TinkerSampleResult,
)


class TestTinkerSampleResult:
    """Tests for TinkerSampleResult."""

    def test_init(self):
        """Test TinkerSampleResult initialization."""
        sequences = [
            {"tokens": [1, 2, 3], "logprobs": [0.1, 0.2, 0.3], "stop_reason": "stop"},
            {"tokens": [4, 5], "logprobs": [0.4, 0.5], "stop_reason": "length"},
        ]
        result = TinkerSampleResult(sequences=sequences, prompt_logprobs=[0.9, 0.8])

        assert len(result.sequences) == 2
        assert result.sequences[0]["tokens"] == [1, 2, 3]
        assert result.prompt_logprobs == [0.9, 0.8]

    def test_to_dict(self):
        """Test TinkerSampleResult.to_dict()."""
        sequences = [{"tokens": [1, 2], "logprobs": [], "stop_reason": "stop"}]
        result = TinkerSampleResult(sequences=sequences)

        d = result.to_dict()
        assert d["sequences"] == sequences
        assert d["prompt_logprobs"] is None


class TestTinkerInferenceAdapter:
    """Tests for TinkerInferenceAdapter."""

    def test_extract_prompt_tokens(self):
        """Test extracting tokens from ModelInput dict."""
        model_input = {
            "chunks": [
                {"tokens": [1, 2, 3]},
                {"tokens": [4, 5]},
            ]
        }
        tokens = TinkerInferenceAdapter.extract_prompt_tokens(model_input)
        assert tokens == [1, 2, 3, 4, 5]

    def test_extract_prompt_tokens_empty(self):
        """Test extracting tokens from empty ModelInput."""
        model_input = {"chunks": []}
        tokens = TinkerInferenceAdapter.extract_prompt_tokens(model_input)
        assert tokens == []

    def test_convert_sampling_params_basic(self):
        """Test basic sampling params conversion."""
        params = {
            "temperature": 0.7,
            "max_tokens": 100,
        }
        result = TinkerInferenceAdapter.convert_sampling_params(params)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert result["top_k"] == -1  # default
        assert result["top_p"] == 1.0  # default

    def test_convert_sampling_params_full(self):
        """Test full sampling params conversion."""
        params = {
            "temperature": 0.5,
            "max_tokens": 200,
            "top_k": 50,
            "top_p": 0.9,
            "seed": 42,
            "stop_tokens": [100, 200],
            "stop_strings": ["END"],
        }
        result = TinkerInferenceAdapter.convert_sampling_params(params)

        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 200
        assert result["top_k"] == 50
        assert result["top_p"] == 0.9
        assert result["seed"] == 42
        assert result["stop_token_ids"] == [100, 200]
        assert result["stop"] == ["END"]

    def test_convert_output_stop_reason_mapping(self):
        """Test that stop reasons are mapped correctly."""
        mock_client = MagicMock()
        adapter = TinkerInferenceAdapter(mock_client)

        # Test "eos" maps to "stop"
        output = {
            "response_ids": [[1, 2, 3]],
            "stop_reasons": ["eos"],
            "response_logprobs": [[0.1, 0.2, 0.3]],
        }
        result = adapter._convert_output(output)
        assert result.sequences[0]["stop_reason"] == "stop"

        # Test "stop" stays as "stop"
        output["stop_reasons"] = ["stop"]
        result = adapter._convert_output(output)
        assert result.sequences[0]["stop_reason"] == "stop"

        # Test "length" stays as "length"
        output["stop_reasons"] = ["length"]
        result = adapter._convert_output(output)
        assert result.sequences[0]["stop_reason"] == "length"

        # Test unknown reason maps to "length"
        output["stop_reasons"] = ["unknown"]
        result = adapter._convert_output(output)
        assert result.sequences[0]["stop_reason"] == "length"

    def test_convert_output_multiple_samples(self):
        """Test converting output with multiple samples."""
        mock_client = MagicMock()
        adapter = TinkerInferenceAdapter(mock_client)

        output = {
            "response_ids": [[1, 2], [3, 4, 5], [6]],
            "stop_reasons": ["stop", "length", "eos"],
            "response_logprobs": [[0.1, 0.2], [0.3, 0.4, 0.5], [0.6]],
        }
        result = adapter._convert_output(output)

        assert len(result.sequences) == 3
        assert result.sequences[0]["tokens"] == [1, 2]
        assert result.sequences[1]["tokens"] == [3, 4, 5]
        assert result.sequences[2]["tokens"] == [6]
        assert result.sequences[0]["stop_reason"] == "stop"
        assert result.sequences[1]["stop_reason"] == "length"
        assert result.sequences[2]["stop_reason"] == "stop"

    def test_convert_output_no_logprobs(self):
        """Test converting output without logprobs."""
        mock_client = MagicMock()
        adapter = TinkerInferenceAdapter(mock_client)

        output = {
            "response_ids": [[1, 2, 3]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }
        result = adapter._convert_output(output)

        assert result.sequences[0]["logprobs"] == []

    @pytest.mark.asyncio
    async def test_sample_calls_client(self):
        """Test that sample() calls the inference client correctly."""
        mock_client = MagicMock()
        mock_client.sample = AsyncMock(
            return_value={
                "response_ids": [[1, 2, 3], [4, 5, 6]],
                "stop_reasons": ["stop", "length"],
                "response_logprobs": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            }
        )

        adapter = TinkerInferenceAdapter(mock_client)
        result = await adapter.sample(
            prompt_tokens=[10, 20, 30],
            num_samples=2,
            sampling_params={"temperature": 0.7, "max_tokens": 50},
            session_id="test-session",
        )

        # Verify client was called with correct args
        mock_client.sample.assert_called_once_with(
            prompt_token_ids=[10, 20, 30],
            num_samples=2,
            sampling_params={"temperature": 0.7, "max_tokens": 50},
            session_id="test-session",
        )

        # Verify result conversion
        assert len(result.sequences) == 2
        assert result.sequences[0]["tokens"] == [1, 2, 3]
        assert result.sequences[1]["tokens"] == [4, 5, 6]
