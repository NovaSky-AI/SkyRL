import time
from cloudpathlib import AnyPath
from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import types
import jax

# Use Qwen3 (supported) with larger vocab
# Qwen3-0.6B has 151,936 tokens (vs 1,000 for tiny)
BASE_MODEL = "Qwen/Qwen3-0.6B"

def make_fwd_bwd_input(token_lists):
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

def build_engine():
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=4,
        max_lora_rank=32,
        train_micro_batch_size=4,
    )
    engine = TinkerEngine(config)
    
    for i in range(2):
        model_id = f"adapter_{i}"
        engine.process_single_request(
            types.RequestType.CREATE_MODEL,
            model_id,
            {"lora_config": {"rank": 32, "alpha": 32}},
        )
    
    return engine

def build_batch(engine, n_requests=8, samples_per_request=2, seq_len=128):
    # Generate random tokens
    token_lists = [
        [int(x) for x in jax.random.randint(
            jax.random.PRNGKey(i), (seq_len,), 1, 1000
        )]
        for i in range(samples_per_request)
    ]
    fb_input = make_fwd_bwd_input(token_lists)
    
    model_ids = list(engine.models.keys())
    reqs = {}
    
    for i in range(n_requests):
        model_id = model_ids[i % len(model_ids)]
        reqs[str(i)] = (model_id, fb_input)
    
    return reqs

def reset_accumulators(engine):
    engine.accumulated_grads = type(engine.accumulated_grads).create(
        engine.lora_params, engine.config.max_lora_adapters
    )

def run_bench(num_steps=20, warmup_steps=3):
    print(f"\n{'='*80}")
    print(f"Benchmarking: {BASE_MODEL}")
    print(f"Vocab size: 151,936 tokens")
    print(f"{'='*80}\n")
    
    print("Building engine (this will download the model ~1.2GB)...")
    engine = build_engine()
    
    print("Building batch...")
    reqs = build_batch(engine, n_requests=8, samples_per_request=2, seq_len=128)
    
    print(f"Warming up ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        engine.process_forward_backward_batch(reqs)
        reset_accumulators(engine)
        print(f"  Warmup {i+1}/{warmup_steps}")
    
    print(f"\nRunning benchmark ({num_steps} steps)...")
    jax.block_until_ready(engine.lora_params)
    
    start = time.perf_counter()
    for i in range(num_steps):
        engine.process_forward_backward_batch(reqs)
        reset_accumulators(engine)
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{num_steps}")
    
    jax.block_until_ready(engine.lora_params)
    elapsed = time.perf_counter() - start
    
    total_tokens = num_steps * 8 * 2 * 128
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"steps:       {num_steps}")
    print(f"elapsed:     {elapsed:.3f} s")
    print(f"steps/sec:   {num_steps / elapsed:.2f}")
    print(f"tokens/sec:  {total_tokens / elapsed:.0f}")
    print(f"ms/step:     {(elapsed / num_steps) * 1000:.2f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_bench()
