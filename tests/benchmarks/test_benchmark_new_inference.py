"""Unit tests for benchmark_new_inference pure functions.

These tests cover request translation, and dataset loading"""

import argparse
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from transformers import AutoTokenizer
from vllm.benchmarks.datasets import SampleRequest

from skyrl.benchmarks.benchmark_new_inference import (
    ConversationResult,
    _conversation_to_chat_messages,
    calculate_multi_turn_results,
    get_sample_requests,
    load_sharegpt_conversations,
    run_benchmark,
    run_multi_turn_benchmark,
    sample_request_to_engine_input,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture
def sample_request():
    return SampleRequest(
        prompt="Hello, how are you?",
        prompt_len=5,
        expected_output_len=32,
    )


# ---------------------------------------------------------------------------
# sample_request_to_engine_input
# ---------------------------------------------------------------------------


class TestSampleRequestToEngineInput:
    def test_basic_conversion(self, tokenizer, sample_request):
        engine_input = sample_request_to_engine_input(sample_request, tokenizer)

        assert "prompt_token_ids" in engine_input
        assert "sampling_params" in engine_input
        assert len(engine_input["prompt_token_ids"]) == 1
        assert isinstance(engine_input["prompt_token_ids"][0], list)
        assert all(isinstance(t, int) for t in engine_input["prompt_token_ids"][0])
        assert engine_input["sampling_params"]["max_tokens"] == 32

    def test_token_ids_match_tokenizer(self, tokenizer, sample_request):
        engine_input = sample_request_to_engine_input(sample_request, tokenizer)
        expected_ids = tokenizer.encode(sample_request.prompt)
        assert engine_input["prompt_token_ids"][0] == expected_ids

    def test_output_len_mapping(self, tokenizer):
        request = SampleRequest(prompt="test", prompt_len=1, expected_output_len=256)
        engine_input = sample_request_to_engine_input(request, tokenizer)
        assert engine_input["sampling_params"]["max_tokens"] == 256


# ---------------------------------------------------------------------------
# get_sample_requests
# ---------------------------------------------------------------------------


class TestGetSampleRequests:
    def test_random_dataset(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="random",
            num_prompts=5,
            input_len=64,
            output_len=32,
            seed=42,
        )
        requests = get_sample_requests(args, tokenizer)

        assert len(requests) == 5
        for r in requests:
            assert isinstance(r, SampleRequest)
            assert isinstance(r.prompt, str)
            assert r.prompt_len > 0
            assert r.expected_output_len > 0

    def test_sharegpt_missing_path(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="sharegpt",
            dataset_path=None,
            num_prompts=5,
            seed=0,
        )
        with pytest.raises(ValueError, match="--dataset-path is required"):
            get_sample_requests(args, tokenizer)

    def test_gsm8k_dataset(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="gsm8k",
            num_prompts=5,
            output_len=64,
            seed=42,
        )
        requests = get_sample_requests(args, tokenizer)

        assert len(requests) == 5
        for r in requests:
            assert isinstance(r, SampleRequest)
            assert isinstance(r.prompt, str)
            assert r.prompt_len > 0
            assert r.expected_output_len == 64

    def test_gsm8k_dataset_uses_answer_length(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="gsm8k",
            num_prompts=3,
            output_len=None,
            seed=0,
        )
        requests = get_sample_requests(args, tokenizer)

        assert len(requests) == 3
        # With output_len=None, expected_output_len comes from answer token count
        for r in requests:
            assert r.expected_output_len > 0

    def test_unknown_dataset(self, tokenizer):
        args = argparse.Namespace(dataset_name="nonexistent")
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_sample_requests(args, tokenizer)


# ---------------------------------------------------------------------------
# run_benchmark (with mocked client)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_batch_mode(self, tokenizer):
        """Test batch mode (request_rate=inf) with a mock client."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "responses": ["output text"],
            "response_ids": [[1, 2, 3, 4, 5]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }

        requests = [
            SampleRequest(prompt="Hello", prompt_len=3, expected_output_len=16),
            SampleRequest(prompt="World", prompt_len=3, expected_output_len=16),
        ]

        results, duration = asyncio.run(
            run_benchmark(
                client=mock_client,
                requests=requests,
                tokenizer=tokenizer,
                request_rate=float("inf"),
            )
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.output_len == 5 for r in results)
        assert duration > 0
        assert mock_client.generate.call_count == 2

    def test_batch_mode_with_failure(self, tokenizer):
        """Test that exceptions in generate are captured as failed results."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            {
                "responses": ["ok"],
                "response_ids": [[1, 2]],
                "stop_reasons": ["stop"],
                "response_logprobs": None,
            },
            RuntimeError("connection failed"),
        ]

        requests = [
            SampleRequest(prompt="Hello", prompt_len=3, expected_output_len=16),
            SampleRequest(prompt="World", prompt_len=3, expected_output_len=16),
        ]

        results, duration = asyncio.run(
            run_benchmark(
                client=mock_client,
                requests=requests,
                tokenizer=tokenizer,
                request_rate=float("inf"),
            )
        )

        assert len(results) == 2
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        assert len(successful) == 1
        assert len(failed) == 1
        assert "connection failed" in failed[0].error

    def test_online_mode(self, tokenizer):
        """Test online mode (finite request_rate) with a mock client."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "responses": ["output"],
            "response_ids": [[10, 20, 30]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }

        requests = [
            SampleRequest(prompt="Test", prompt_len=2, expected_output_len=8),
            SampleRequest(prompt="Data", prompt_len=2, expected_output_len=8),
        ]

        # Use a high request rate to keep the test fast
        results, duration = asyncio.run(
            run_benchmark(
                client=mock_client,
                requests=requests,
                tokenizer=tokenizer,
                request_rate=1000.0,
            )
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.output_len == 3 for r in results)


# ---------------------------------------------------------------------------
# Multi-turn: conversation loading and conversion
# ---------------------------------------------------------------------------


class TestConversationToChat:
    def test_basic_conversion(self):
        turns = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
            {"from": "human", "value": "How are you?"},
        ]
        messages = _conversation_to_chat_messages(turns)
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_empty_conversation(self):
        assert _conversation_to_chat_messages([]) == []


class TestLoadSharegptConversations:
    def test_load_from_file(self, tmp_path):
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                ]
            },
            {"conversations": [{"from": "human", "value": "short"}]},  # only 1 turn, filtered
        ]
        path = tmp_path / "sharegpt.json"
        path.write_text(json.dumps(data))

        convos = load_sharegpt_conversations(str(path), num_conversations=10, min_turns=2)
        assert len(convos) == 2  # third entry has only 1 turn

    def test_num_conversations_limit(self, tmp_path):
        data = [
            {"conversations": [{"from": "human", "value": f"Q{i}"}, {"from": "gpt", "value": f"A{i}"}]}
            for i in range(20)
        ]
        path = tmp_path / "sharegpt.json"
        path.write_text(json.dumps(data))

        convos = load_sharegpt_conversations(str(path), num_conversations=5)
        assert len(convos) == 5

    def test_deterministic_with_seed(self, tmp_path):
        data = [
            {"conversations": [{"from": "human", "value": f"Q{i}"}, {"from": "gpt", "value": f"A{i}"}]}
            for i in range(10)
        ]
        path = tmp_path / "sharegpt.json"
        path.write_text(json.dumps(data))

        c1 = load_sharegpt_conversations(str(path), num_conversations=5, seed=42)
        c2 = load_sharegpt_conversations(str(path), num_conversations=5, seed=42)
        assert c1 == c2


# ---------------------------------------------------------------------------
# Multi-turn: calculate_multi_turn_results
# ---------------------------------------------------------------------------


class TestCalculateMultiTurnResults:
    def test_all_successful(self):
        results = [
            ConversationResult(
                success=True,
                num_turns=3,
                total_latency=1.5,
                per_turn_latencies=[0.3, 0.5, 0.7],
                total_input_tokens=300,
                total_output_tokens=150,
            ),
            ConversationResult(
                success=True,
                num_turns=2,
                total_latency=1.0,
                per_turn_latencies=[0.4, 0.6],
                total_input_tokens=200,
                total_output_tokens=100,
            ),
        ]
        result = calculate_multi_turn_results(results, duration=2.0)

        assert result.completed == 2
        assert result.failed == 0
        assert result.total_input_tokens == 500
        assert result.total_output_tokens == 250
        assert result.request_throughput == pytest.approx(1.0)  # 2/2
        assert result.output_throughput == pytest.approx(125.0)  # 250/2

    def test_with_failures(self):
        results = [
            ConversationResult(
                success=True,
                num_turns=2,
                total_latency=1.0,
                per_turn_latencies=[0.5, 0.5],
                total_input_tokens=200,
                total_output_tokens=100,
            ),
            ConversationResult(
                success=False,
                num_turns=0,
                total_latency=0.1,
                per_turn_latencies=[],
                total_input_tokens=0,
                total_output_tokens=0,
                error="failed",
            ),
        ]
        result = calculate_multi_turn_results(results, duration=1.5)

        assert result.completed == 1
        assert result.failed == 1

    def test_all_failed(self):
        results = [
            ConversationResult(
                success=False,
                num_turns=0,
                total_latency=0.1,
                per_turn_latencies=[],
                total_input_tokens=0,
                total_output_tokens=0,
                error="err",
            ),
        ]
        result = calculate_multi_turn_results(results, duration=1.0)
        assert result.completed == 0
        assert result.failed == 1
        assert result.percentiles_e2el_ms == []


# ---------------------------------------------------------------------------
# Multi-turn: run_multi_turn_benchmark (with mocked client)
# ---------------------------------------------------------------------------


class TestRunMultiTurnBenchmark:
    def test_basic_two_turn_conversation(self, tokenizer):
        """Two-turn conversation: user->assistant->user->assistant."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "responses": ["Sure, I can help."],
            "response_ids": [[1, 2, 3, 4]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }

        conversations = [
            [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
                {"from": "human", "value": "How are you?"},
                {"from": "gpt", "value": "Good"},
            ]
        ]

        results, duration = asyncio.run(
            run_multi_turn_benchmark(
                client=mock_client,
                conversations=conversations,
                tokenizer=tokenizer,
                max_tokens_per_turn=64,
            )
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].num_turns == 2  # 2 user turns
        assert len(results[0].per_turn_latencies) == 2
        assert results[0].total_output_tokens == 8  # 4 tokens * 2 turns
        assert mock_client.generate.call_count == 2

    def test_conversation_with_failure(self, tokenizer):
        """Exception on second turn is captured."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            {
                "responses": ["First response"],
                "response_ids": [[1, 2]],
                "stop_reasons": ["stop"],
                "response_logprobs": None,
            },
            RuntimeError("connection lost"),
        ]

        conversations = [
            [
                {"from": "human", "value": "Turn 1"},
                {"from": "gpt", "value": "..."},
                {"from": "human", "value": "Turn 2"},
                {"from": "gpt", "value": "..."},
            ]
        ]

        results, duration = asyncio.run(
            run_multi_turn_benchmark(
                client=mock_client,
                conversations=conversations,
                tokenizer=tokenizer,
                max_tokens_per_turn=32,
            )
        )

        assert len(results) == 1
        assert not results[0].success
        assert results[0].num_turns == 1  # only first turn succeeded
        assert "connection lost" in results[0].error

    def test_concurrent_conversations(self, tokenizer):
        """Multiple conversations running concurrently."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "responses": ["Response"],
            "response_ids": [[10, 20]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }

        conversations = [[{"from": "human", "value": f"Q{i}"}, {"from": "gpt", "value": f"A{i}"}] for i in range(3)]

        results, duration = asyncio.run(
            run_multi_turn_benchmark(
                client=mock_client,
                conversations=conversations,
                tokenizer=tokenizer,
                max_tokens_per_turn=32,
                concurrency=3,
            )
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.num_turns == 1 for r in results)

    def test_no_user_turns(self, tokenizer):
        """Conversation with only assistant turns returns failure."""
        mock_client = AsyncMock()

        conversations = [
            [
                {"from": "gpt", "value": "I'm an assistant"},
            ]
        ]

        results, duration = asyncio.run(
            run_multi_turn_benchmark(
                client=mock_client,
                conversations=conversations,
                tokenizer=tokenizer,
                max_tokens_per_turn=32,
            )
        )

        assert len(results) == 1
        assert not results[0].success
        assert "No user turns" in results[0].error
        assert mock_client.generate.call_count == 0
