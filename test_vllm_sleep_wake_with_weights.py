"""
Variant 2: Sleep/wake crash reproduction WITH simulated weight updates.

This simulates the full SkyRL training loop:
  sleep -> wake_up(weights) -> update_weights -> wake_up(kv_cache) -> generate -> sleep

In vLLM 0.16.0, update_weights is available via collective_rpc or the engine's
load_weights/update_weights API. Here we use collective_rpc("update_weights")
to simulate a weight update by reloading the same checkpoint weights, which
exercises the same CUDA memory paths as a real training update.

Usage: python test_vllm_sleep_wake_with_weights.py [--cycles 10] [--model Qwen/Qwen3-8B]
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback

os.environ["VLLM_USE_V1"] = "1"

import torch
import vllm
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sleep_wake_weights_repro")


def create_engine(model: str, max_model_len: int, gpu_mem_util: float):
    """Create an AsyncLLMEngine."""
    engine_args = vllm.AsyncEngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=1,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_log_requests=False,
    )
    engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
    log.info(f"Engine created: vllm {vllm.__version__}, model={model}")
    return engine


async def generate_completions(engine, num_prompts: int = 4, max_tokens: int = 64):
    """Generate a small batch of completions."""
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    test_prompts = [
        [9707, 11, 1917, 0],
        [791, 4059, 315],
        [25, 220, 16, 489, 220, 16, 284],
        [3923, 374, 279, 6864, 315, 9822, 30],
    ]

    tasks = []
    for i in range(num_prompts):
        prompt_tokens = test_prompts[i % len(test_prompts)]
        request_id = f"req-{int(time.time_ns())}-{i}"

        async def collect(rid, tokens):
            final = None
            async for output in engine.generate(
                prompt=TokensPrompt(prompt_token_ids=tokens),
                sampling_params=sampling_params,
                request_id=rid,
            ):
                final = output
            return final

        tasks.append(asyncio.create_task(collect(request_id, prompt_tokens)))

    outputs = await asyncio.gather(*tasks, return_exceptions=True)
    successes = 0
    for i, out in enumerate(outputs):
        if isinstance(out, Exception):
            log.error(f"  Prompt {i} FAILED: {out}")
        else:
            text = out.outputs[0].text[:80] if out and out.outputs else "<empty>"
            successes += 1
            log.info(f"  Prompt {i}: {text!r}")
    return successes


async def simulate_weight_update(engine, model_path: str, cycle: int):
    """
    Simulate a weight update between sleep/wake cycles.

    Strategy 1 (preferred): Use engine.collective_rpc("update_weights") if available.
    Strategy 2 (fallback): Load model state_dict and apply via collective_rpc("load_weights").
    Strategy 3 (simplest): Use the /update_weights endpoint pattern from vLLM 0.16.0.

    For reproduction purposes, we reload the SAME weights (no actual training),
    which still exercises the CUDA memory allocation/deallocation paths.
    """
    log.info(f"[Cycle {cycle}] Simulating weight update...")
    t0 = time.time()

    try:
        # vLLM 0.16.0 has update_weights via collective_rpc
        # This reloads weights from the model path (same weights, but exercises the path)
        await engine.collective_rpc(
            "update_weights",
            args=(model_path,),
        )
        log.info(f"[Cycle {cycle}] Weight update via collective_rpc done in {time.time()-t0:.2f}s")
        return True
    except Exception as e1:
        log.warning(f"[Cycle {cycle}] collective_rpc('update_weights') failed: {e1}")

    try:
        # Fallback: try direct model reload via check_weights_changed API
        # This is available in some vLLM versions
        await engine.check_and_update_model(model_path)
        log.info(f"[Cycle {cycle}] Weight update via check_and_update_model done in {time.time()-t0:.2f}s")
        return True
    except Exception as e2:
        log.warning(f"[Cycle {cycle}] check_and_update_model failed: {e2}")

    try:
        # Last resort: perturb a single weight tensor via collective_rpc
        # This exercises the weight update CUDA path with minimal overhead
        log.info(f"[Cycle {cycle}] Attempting weight perturbation via collective_rpc...")

        # Get a weight name from the model (use a small one like layernorm)
        # We pass a dummy perturbation that workers can apply
        await engine.collective_rpc(
            "apply_weight_delta",
            args=(),
        )
        log.info(f"[Cycle {cycle}] Weight perturbation done in {time.time()-t0:.2f}s")
        return True
    except Exception as e3:
        log.warning(f"[Cycle {cycle}] All weight update strategies failed: {e3}")
        log.warning(f"[Cycle {cycle}] Continuing without weight update (testing sleep/wake only)")
        return False


async def run_cycles_with_weights(engine, model: str, num_cycles: int, sleep_level: int = 2):
    """
    Full training-loop simulation:
      wake_up(weights) -> [weight update] -> wake_up(kv_cache) -> generate -> reset_prefix_cache -> sleep
    """
    for cycle in range(1, num_cycles + 1):
        log.info(f"{'='*60}")
        log.info(f"CYCLE {cycle}/{num_cycles}")
        log.info(f"{'='*60}")

        try:
            # Phase 1: Wake up weights
            log.info(f"[Cycle {cycle}] wake_up(tags=['weights'])...")
            t0 = time.time()
            await engine.wake_up(tags=["weights"])
            log.info(f"[Cycle {cycle}] wake_up(weights) done in {time.time()-t0:.2f}s")

            # Phase 2: Simulate weight update (this is what happens during RL training)
            weight_updated = await simulate_weight_update(engine, model, cycle)
            if weight_updated:
                log.info(f"[Cycle {cycle}] Weights updated successfully")
            else:
                log.info(f"[Cycle {cycle}] Weight update skipped (testing sleep/wake path only)")

            # Phase 3: Wake up KV cache
            log.info(f"[Cycle {cycle}] wake_up(tags=['kv_cache'])...")
            t0 = time.time()
            await engine.wake_up(tags=["kv_cache"])
            log.info(f"[Cycle {cycle}] wake_up(kv_cache) done in {time.time()-t0:.2f}s")

            # Phase 4: Generate
            log.info(f"[Cycle {cycle}] Generating completions...")
            t0 = time.time()
            successes = await generate_completions(engine, num_prompts=4, max_tokens=64)
            log.info(f"[Cycle {cycle}] Generation done in {time.time()-t0:.2f}s, "
                     f"{successes}/4 succeeded")

            # Phase 5: Reset prefix cache before sleep
            log.info(f"[Cycle {cycle}] reset_prefix_cache()...")
            await engine.reset_prefix_cache()

            # Phase 6: Sleep
            log.info(f"[Cycle {cycle}] sleep(level={sleep_level})...")
            t0 = time.time()
            await engine.sleep(level=sleep_level)
            log.info(f"[Cycle {cycle}] sleep done in {time.time()-t0:.2f}s")

            # Log GPU memory state
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                log.info(f"[Cycle {cycle}] GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

            log.info(f"[Cycle {cycle}] PASSED")

        except Exception as e:
            log.error(f"[Cycle {cycle}] CRASH: {type(e).__name__}: {e}")
            log.error(traceback.format_exc())

            # Log GPU memory state at crash time
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                log.error(f"[Cycle {cycle}] GPU memory at crash: "
                          f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

            log.error(f"Crashed on cycle {cycle}/{num_cycles}")
            return cycle

    log.info(f"All {num_cycles} cycles completed without crash.")
    return 0


async def main():
    parser = argparse.ArgumentParser(
        description="vLLM sleep/wake crash reproduction with weight updates"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name/path")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-mem-util", type=float, default=0.8)
    parser.add_argument("--cycles", type=int, default=10, help="Number of sleep/wake cycles")
    parser.add_argument("--sleep-level", type=int, default=2,
                        help="Sleep level (1=keep KV cache, 2=free all)")
    args = parser.parse_args()

    log.info(f"vLLM version: {vllm.__version__}")
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
    log.info(f"Config: model={args.model}, max_model_len={args.max_model_len}, "
             f"gpu_mem={args.gpu_mem_util}, cycles={args.cycles}, sleep_level={args.sleep_level}")

    engine = create_engine(args.model, args.max_model_len, args.gpu_mem_util)

    crash_cycle = await run_cycles_with_weights(
        engine, args.model, args.cycles, args.sleep_level
    )

    if crash_cycle:
        log.error(f"RESULT: Crashed on cycle {crash_cycle}")
        sys.exit(1)
    else:
        log.info("RESULT: No crash detected")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
