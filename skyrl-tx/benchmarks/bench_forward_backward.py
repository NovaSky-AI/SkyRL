# bench_forward_backward.py
import time

from cloudpathlib import AnyPath

from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import types

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def make_fwd_bwd_input(token_lists: list[list[int]]) -> types.ForwardBackwardInput:
    """Same helper as in test_engine.py, inlined here."""
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


def build_engine(num_adapters: int = 4, train_micro_batch_size: int = 8) -> TinkerEngine:
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
        train_micro_batch_size=train_micro_batch_size,
    )
    engine = TinkerEngine(config)

    for i in range(num_adapters):
        model_id = f"adapter_{i}"
        engine.process_single_request(
            types.RequestType.CREATE_MODEL,
            model_id,
            {"lora_config": {"rank": 32, "alpha": 32}},
        )

    return engine


def build_batch(
    engine: TinkerEngine,
    n_requests: int,
    samples_per_request: int,
    seq_len: int,
) -> dict[str, tuple[str, types.ForwardBackwardInput]]:
    """Mix requests across adapters to exercise AccumulatedGradients."""
    token_lists = [list(range(1, seq_len + 1)) for _ in range(samples_per_request)]
    fb_input = make_fwd_bwd_input(token_lists)

    model_ids = list(engine.models.keys())
    reqs: dict[str, tuple[str, types.ForwardBackwardInput]] = {}

    for i in range(n_requests):
        model_id = model_ids[i % len(model_ids)]
        reqs[str(i)] = (model_id, fb_input)

    return reqs


def reset_accumulators(engine: TinkerEngine) -> None:
    """Reset accumulated gradients - works with both per-adapter and global accumulator patterns."""
    if isinstance(engine.accumulated_grads, dict):
        # Pattern 1: One accumulator per adapter (current implementation)
        for acc in engine.accumulated_grads.values():
            acc.reset()
    else:
        # Pattern 2: Single global accumulator for all adapters
        engine.accumulated_grads.reset()


def run_bench(
    n_requests: int = 32,
    samples_per_request: int = 2,
    seq_len: int = 64,
    num_steps: int = 30,
    warmup_steps: int = 5,
):
    engine = build_engine(num_adapters=4, train_micro_batch_size=8)
    reqs = build_batch(engine, n_requests, samples_per_request, seq_len)

    # Warmup – pay JIT cost outside the measured region.
    for _ in range(warmup_steps):
        engine.process_forward_backward_batch(reqs)
        reset_accumulators(engine)

    # Measure steady-state fwd+bwd+accum timing.
    start = time.perf_counter()
    for _ in range(num_steps):
        engine.process_forward_backward_batch(reqs)
        reset_accumulators(engine)
    elapsed = time.perf_counter() - start

    total_tokens = num_steps * n_requests * samples_per_request * seq_len
    steps_per_sec = num_steps / elapsed
    toks_per_sec = total_tokens / elapsed

    print(f"steps:       {num_steps}")
    print(f"elapsed:     {elapsed:.3f} s")
    print(f"steps/sec:   {steps_per_sec:.2f}")
    print(f"tokens/sec:  {toks_per_sec:.0f}")


if __name__ == "__main__":
    run_bench()
