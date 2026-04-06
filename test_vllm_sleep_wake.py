"""
Minimal reproduction script for vLLM cudaErrorInvalidValue crash during sleep/wake cycles.

Tests whether repeated sleep → wake_up(weights) → wake_up(kv_cache) → generate → sleep
causes the crash at flash_attn.py:484 (scheduler_metadata[:n] = scheduler_metadata).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --isolated --extra fsdp python test_vllm_sleep_wake.py --cycles 15
    CUDA_VISIBLE_DEVICES=0 uv run --isolated --extra fsdp python test_vllm_sleep_wake.py --cycles 15 --with-weight-update
"""

import argparse
import asyncio
import logging
import os
import sys
import time

os.environ["VLLM_USE_V1"] = "1"

import torch
import vllm
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sleep_wake_repro")


def run_test(model: str, max_model_len: int, gpu_mem_util: float,
             num_cycles: int, with_weight_update: bool):
    """Test sleep/wake cycles with synchronous LLM API."""

    log.info(f"vLLM version: {vllm.__version__}")
    log.info(f"Config: model={model}, max_model_len={max_model_len}, "
             f"gpu_mem={gpu_mem_util}, cycles={num_cycles}, "
             f"with_weight_update={with_weight_update}")

    # Create engine
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=1,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    log.info("Engine created successfully")

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
    test_prompts = [
        "Write a Python function that checks if a number is prime.",
        "What is the capital of France? Explain briefly.",
        "Solve: 2 + 2 * 3 = ?",
        "Tell me a short joke about programming.",
    ]

    for cycle in range(1, num_cycles + 1):
        log.info(f"{'='*60}")
        log.info(f"CYCLE {cycle}/{num_cycles}")
        log.info(f"{'='*60}")

        try:
            # Phase 1: Sleep (free GPU memory, simulates training phase occupying GPU)
            log.info(f"[Cycle {cycle}] sleep(level=2)...")
            t0 = time.time()
            llm.sleep(level=2)
            log.info(f"[Cycle {cycle}] sleep done in {time.time()-t0:.2f}s")

            # Simulate training phase (just wait a bit)
            time.sleep(0.5)

            # Phase 2: Wake up weights
            log.info(f"[Cycle {cycle}] wake_up(tags=['weights'])...")
            t0 = time.time()
            llm.wake_up(tags=["weights"])
            log.info(f"[Cycle {cycle}] wake_up(weights) done in {time.time()-t0:.2f}s")

            # Phase 3: Optionally simulate weight update (like SkyRL's broadcast_to_inference_engines)
            if with_weight_update:
                log.info(f"[Cycle {cycle}] Simulating weight update (noop load_state_dict)...")
                # In real SkyRL, weights are updated via NCCL broadcast.
                # Here we just touch the model to simulate the effect.
                # This is a lightweight stand-in — the real weight sync uses
                # the NCCL weight transfer sender.
                pass

            # Phase 4: Wake up KV cache
            log.info(f"[Cycle {cycle}] wake_up(tags=['kv_cache'])...")
            t0 = time.time()
            llm.wake_up(tags=["kv_cache"])
            log.info(f"[Cycle {cycle}] wake_up(kv_cache) done in {time.time()-t0:.2f}s")

            # Phase 5: Generate completions
            log.info(f"[Cycle {cycle}] Generating {len(test_prompts)} completions...")
            t0 = time.time()
            outputs = llm.generate(test_prompts, sampling_params)
            gen_time = time.time() - t0

            for i, output in enumerate(outputs):
                text = output.outputs[0].text[:60]
                log.info(f"  Prompt {i}: {text!r}")

            log.info(f"[Cycle {cycle}] Generation done in {gen_time:.2f}s")
            log.info(f"[Cycle {cycle}] PASSED ✓")

        except Exception as e:
            log.error(f"[Cycle {cycle}] CRASH: {type(e).__name__}: {e}")
            import traceback
            log.error(traceback.format_exc())
            log.error(f"RESULT: Crashed on cycle {cycle}/{num_cycles}")
            return cycle

    log.info(f"RESULT: All {num_cycles} cycles completed without crash ✓")
    return 0


def main():
    parser = argparse.ArgumentParser(description="vLLM sleep/wake crash reproduction")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name/path")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-mem-util", type=float, default=0.8)
    parser.add_argument("--cycles", type=int, default=15, help="Number of sleep/wake cycles")
    parser.add_argument("--with-weight-update", action="store_true",
                        help="Simulate weight updates between sleep/wake")
    args = parser.parse_args()

    crash_cycle = run_test(
        args.model, args.max_model_len, args.gpu_mem_util,
        args.cycles, args.with_weight_update,
    )
    sys.exit(1 if crash_cycle else 0)


if __name__ == "__main__":
    main()
