"""Benchmark forward/backward passes for the TinkerEngine."""

import argparse
import time

import jax
from cloudpathlib import AnyPath

from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import types


# Configuration
NUM_ADAPTERS = 4
MAX_LORA_ADAPTERS = 8
MAX_LORA_RANK = 32
TRAIN_MICRO_BATCH_SIZE = 8
N_REQUESTS = 32
SAMPLES_PER_REQUEST = 2
SEQ_LEN = 64
NUM_STEPS = 30
WARMUP_STEPS = 5


def make_fwd_bwd_input(token_lists: list[list[int]]) -> types.ForwardBackwardInput:
    samples = []
    for tokens in token_lists:
        targets = tokens[1:] + [0]
        weights = [1] * len(tokens)
        samples.append(
            types.Datum(
                model_input=types.ModelInput(chunks=[types.ModelInputChunk(tokens=tokens)]),
                loss_fn_inputs=types.LossFnInputs(
                    target_tokens=types.TensorData(data=targets),
                    weights=types.TensorData(data=weights),
                    advantages=types.TensorData(data=[]),
                    logprobs=types.TensorData(data=[]),
                ),
            )
        )
    return types.ForwardBackwardInput(data=samples, loss_fn="cross_entropy")


def build_engine(base_model: str) -> TinkerEngine:
    config = EngineConfig(
        base_model=base_model,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=MAX_LORA_ADAPTERS,
        max_lora_rank=MAX_LORA_RANK,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
    )
    engine = TinkerEngine(config)

    for i in range(NUM_ADAPTERS):
        engine.process_single_request(
            types.RequestType.CREATE_MODEL,
            f"adapter_{i}",
            {"lora_config": {"rank": MAX_LORA_RANK, "alpha": MAX_LORA_RANK}},
        )

    return engine


def build_batch(engine: TinkerEngine) -> dict[str, tuple[str, types.ForwardBackwardInput]]:
    token_lists = [
        [int(x) for x in jax.random.randint(jax.random.PRNGKey(i), (SEQ_LEN,), 1, 1000)]
        for i in range(SAMPLES_PER_REQUEST)
    ]
    fb_input = make_fwd_bwd_input(token_lists)

    model_ids = list(engine.models.keys())
    return {str(i): (model_ids[i % len(model_ids)], fb_input) for i in range(N_REQUESTS)}


def reset_accumulators(engine: TinkerEngine) -> None:
    engine.accumulated_grads = type(engine.accumulated_grads).create(
        engine.lora_params, engine.config.max_lora_adapters
    )


def run_bench(base_model: str):
    print(f"Benchmarking: {base_model}")
    print("Building engine...")
    engine = build_engine(base_model)

    print("Building batch...")
    reqs = build_batch(engine)

    print(f"Warming up ({WARMUP_STEPS} steps)...")
    for _ in range(WARMUP_STEPS):
        engine.process_forward_backward_batch(reqs)
        reset_accumulators(engine)

    print(f"Running benchmark ({NUM_STEPS} steps)...")
    jax.block_until_ready(engine.lora_params)

    start = time.perf_counter()
    for _ in range(NUM_STEPS):
        engine.process_forward_backward_batch(reqs)
        reset_accumulators(engine)
    jax.block_until_ready(engine.lora_params)
    elapsed = time.perf_counter() - start

    total_tokens = NUM_STEPS * N_REQUESTS * SAMPLES_PER_REQUEST * SEQ_LEN

    print(f"\nsteps:       {NUM_STEPS}")
    print(f"elapsed:     {elapsed:.3f} s")
    print(f"steps/sec:   {NUM_STEPS / elapsed:.2f}")
    print(f"tokens/sec:  {total_tokens / elapsed:.0f}")
    print(f"ms/step:     {(elapsed / NUM_STEPS) * 1000:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark forward/backward passes")
    parser.add_argument(
        "--model",
        default="trl-internal-testing/tiny-Qwen3ForCausalLM",
        help="Model to benchmark (default: tiny-Qwen3ForCausalLM)",
    )
    args = parser.parse_args()
    run_bench(args.model)
