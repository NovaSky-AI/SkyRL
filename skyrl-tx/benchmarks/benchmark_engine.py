"""Benchmark forward/backward and sampling for the TinkerEngine."""

import argparse
import time

import jax

from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig, add_model
from tx.tinker import types


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


def make_sample_input(prompt_tokens: list[int], max_tokens: int, checkpoint_id: str) -> types.SampleInput:
    return types.SampleInput(
        base_model=None,
        prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=prompt_tokens)]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=max_tokens, seed=42),
        num_samples=1,
        checkpoint_id=checkpoint_id,
        prompt_logprobs=False,
    )


def build_engine(config: EngineConfig, num_adapters: int) -> TinkerEngine:
    engine = TinkerEngine(config)
    for i in range(num_adapters):
        model_id = f"adapter_{i}"
        engine.process_single_request(
            types.RequestType.CREATE_MODEL,
            model_id,
            {"lora_config": {"rank": config.backend_config["max_lora_rank"], "alpha": 32, "seed": i}},
        )
        # Mark as loaded so sampling uses in-memory weights
        engine.backend.models[model_id].loaded_checkpoint_id = model_id
    return engine


def run_fwd_bwd_bench(engine: TinkerEngine, args: argparse.Namespace):
    print("\n=== Forward/Backward Benchmark ===")

    token_lists = [
        [int(x) for x in jax.random.randint(jax.random.PRNGKey(i), (args.seq_len,), 1, 1000)]
        for i in range(args.samples_per_request)
    ]
    fb_input = make_fwd_bwd_input(token_lists)
    model_ids = list(engine.backend.models.keys())
    reqs = {str(i): (model_ids[i % len(model_ids)], fb_input) for i in range(args.num_requests)}

    print(f"Warming up ({args.num_warmup_steps} steps)...")
    for _ in range(args.num_warmup_steps):
        engine.process_forward_backward(reqs)

    print(f"Running benchmark ({args.num_steps} steps)...")
    start = time.perf_counter()
    for _ in range(args.num_steps):
        engine.process_forward_backward(reqs)
    elapsed = time.perf_counter() - start

    total_tokens = args.num_steps * args.num_requests * args.samples_per_request * args.seq_len
    print("\nResults:")
    print(f"  steps:       {args.num_steps}")
    print(f"  elapsed:     {elapsed:.3f} s")
    print(f"  tokens/sec:  {total_tokens / elapsed:.0f}")
    print(f"  sec/step:     {elapsed / args.num_steps:.2f}")


def run_sample_bench(engine: TinkerEngine, args: argparse.Namespace):
    print("\n=== Sampling Benchmark ===")

    model_ids = list(engine.backend.models.keys())
    reqs = {}
    for i in range(args.num_requests):
        prompt_tokens = [int(x) for x in jax.random.randint(jax.random.PRNGKey(i), (args.seq_len,), 1, 1000)]
        model_id = model_ids[i % len(model_ids)]
        reqs[str(i)] = (model_id, make_sample_input(prompt_tokens, args.sample_max_tokens, checkpoint_id=model_id))

    print(f"Warming up ({args.num_warmup_steps} steps)...")
    for _ in range(args.num_warmup_steps):
        engine.process_sample(reqs)

    print(f"Running benchmark ({args.num_steps} steps)...")
    start = time.perf_counter()
    for _ in range(args.num_steps):
        engine.process_sample(reqs)
    elapsed = time.perf_counter() - start

    total_tokens = args.num_steps * args.num_requests * args.sample_max_tokens
    print("\nResults:")
    print(f"  steps:                {args.num_steps}")
    print(f"  elapsed:              {elapsed:.3f} s")
    print(f"  tokens generated/sec: {total_tokens / elapsed:.0f}")
    print(f"  sec/step:              {elapsed / args.num_steps:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark forward/backward and sampling for TinkerEngine")
    add_model(parser, EngineConfig)

    parser.add_argument("--benchmark", choices=["fwd_bwd", "sample", "all"], default="all", help="Benchmark to run")
    parser.add_argument("--num-steps", type=int, default=5, help="Number of benchmark steps")
    parser.add_argument("--num-warmup-steps", type=int, default=2, help="Number of warmup steps")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests per batch")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for inputs")
    parser.add_argument("--samples-per-request", type=int, default=1, help="Samples per request (fwd_bwd only)")
    parser.add_argument("--num-adapters", type=int, default=2, help="Number of LoRA adapters to create")
    parser.add_argument("--sample-max-tokens", type=int, default=128, help="Max tokens to generate (sampling only)")

    args = parser.parse_args()

    # Build EngineConfig from parsed args
    config_fields = {name: getattr(args, name) for name in EngineConfig.model_fields.keys() if hasattr(args, name)}
    config = EngineConfig(**config_fields)

    bench_config = {k: v for k, v in vars(args).items() if k not in EngineConfig.model_fields}
    print(f"EngineConfig: {config}")
    print(f"BenchmarkConfig: {bench_config}")
    print("Building engine...")

    engine = build_engine(config, args.num_adapters)

    if args.benchmark in ("fwd_bwd", "all"):
        run_fwd_bwd_bench(engine, args)

    if args.benchmark in ("sample", "all"):
        run_sample_bench(engine, args)


if __name__ == "__main__":
    main()
