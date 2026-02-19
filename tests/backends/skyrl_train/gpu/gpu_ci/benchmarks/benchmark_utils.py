"""Utilities for batched generation benchmarks."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from transformers import PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput


@dataclass
class BenchmarkResult:
    """Structured benchmark results with per-iteration latency tracking."""

    scenario: str
    batch_size: int
    input_len: int
    output_len: int
    codepath: str
    num_iterations: int
    elapsed_time: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    per_iteration_latencies: List[float] = field(default_factory=list)

    @property
    def requests_per_second(self) -> float:
        return self.total_requests / self.elapsed_time if self.elapsed_time > 0 else 0

    @property
    def output_tokens_per_second(self) -> float:
        return self.total_output_tokens / self.elapsed_time if self.elapsed_time > 0 else 0

    @property
    def avg_latency(self) -> float:
        if not self.per_iteration_latencies:
            return 0.0
        return sum(self.per_iteration_latencies) / len(self.per_iteration_latencies)

    @property
    def p50_latency(self) -> float:
        return float(np.percentile(self.per_iteration_latencies, 50)) if self.per_iteration_latencies else 0.0

    @property
    def p90_latency(self) -> float:
        return float(np.percentile(self.per_iteration_latencies, 90)) if self.per_iteration_latencies else 0.0

    @property
    def p99_latency(self) -> float:
        return float(np.percentile(self.per_iteration_latencies, 99)) if self.per_iteration_latencies else 0.0


def generate_synthetic_prompt_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    input_len: int,
    batch_size: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Generate reproducible synthetic prompt token IDs for benchmarking.

    Randomly samples from non-special tokens, similar to vLLM RandomDataset.
    """
    rng = np.random.default_rng(seed)
    vocab_size = tokenizer.vocab_size
    prohibited = set(tokenizer.all_special_ids)
    allowed = np.array([i for i in range(vocab_size) if i not in prohibited])

    if len(allowed) == 0:
        raise ValueError("No allowed tokens (vocab has only special tokens)")

    indices = rng.integers(0, len(allowed), size=(batch_size, input_len))
    return allowed[indices].tolist()


async def run_benchmark(
    client: Any,
    prompt_token_ids: List[List[int]],
    sampling_params: Dict[str, Any],
    num_iterations: int = 15,
    warmup_iterations: int = 2,
) -> tuple[float, int, List[float]]:
    """
    Run batched generation benchmark with per-iteration timing.

    Returns:
        (total_elapsed, total_output_tokens, per_iteration_latencies)
    """
    input_batch = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    # Warmup
    for _ in range(warmup_iterations):
        await client.generate(input_batch)

    # Timed runs
    latencies = []
    total_output_tokens = 0
    overall_start = time.perf_counter()
    for _ in range(num_iterations):
        iter_start = time.perf_counter()
        output = await client.generate(input_batch)
        latencies.append(time.perf_counter() - iter_start)
        for ids in output["response_ids"]:
            total_output_tokens += len(ids)
    elapsed = time.perf_counter() - overall_start

    return elapsed, total_output_tokens, latencies
