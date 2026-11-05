"""One DSv4.1-Flash engine, several short load phases, vLLM scheduler stats every 30 s.

Phases (name, concurrency, obs tokens per turn): 640 sessions (the bench's 80/GPU), 640 with no observation
prefill (pure decode), 64, 256. Each phase: WARMUP s, then MEASURE s of generated tok/s, per GPU and per sequence.

    python dsv41_inference_diag.py --model <dir> [--lora 0|1] [--warmup 120] [--measure 120]
        [--phases c640,c640_noobs] [--engine-kwargs '{"max_num_batched_tokens": 4096}']
        [--profile c640:8 --profile-dir <dir>]   # torch-profile 8 s of phase c640 after its warmup
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_inference_bench import Stats, session  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402


async def phase(engine, base, name, concurrency, obs, warmup, measure, profile_s=0):
    args = argparse.Namespace(**{**vars(base), "obs": obs})
    stats = Stats()
    stop = asyncio.Event()

    async def worker(i):
        rng = random.Random(1000 + i)
        while not stop.is_set():
            await session(engine, args, stats, rng, args.prompt_len, i)

    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.sleep(warmup)
    if profile_s:
        await engine.start_profile()
        await asyncio.sleep(profile_s)
        await engine.stop_profile()
        print(f"[diag] {name}: profiled {profile_s}s", flush=True)
    stats.counting = True
    t0 = time.time()
    await asyncio.sleep(measure)
    dt = time.time() - t0
    depth = sum(stats.depth.values()) / max(len(stats.depth), 1)
    print(
        f"[diag] {name}: conc={concurrency} obs={obs} gen {stats.gen / dt:.0f} tok/s ({stats.gen / dt / args.tp:.0f}/GPU, "
        f"{stats.gen / dt / concurrency:.1f}/seq) uncached-prompt~{stats.turns * obs / dt:.0f} tok/s turns {stats.turns} "
        f"depth {depth:.0f}",
        flush=True,
    )
    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(20)


async def stats_loop(engine):
    while True:
        await asyncio.sleep(30)
        await engine.do_log_stats()


async def main(a):
    lora = {"enable_lora": True, "max_lora_rank": 32, "max_loras": 1} if a.lora else {}
    kwargs = dict(
        model=a.model,
        tensor_parallel_size=a.tp,
        trust_remote_code=True,
        max_model_len=262144,
        max_num_seqs=1024,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        swa_bounded_replay=False,
        async_scheduling=True,
        distributed_executor_backend="mp",
        disable_log_stats=False,
        max_num_batched_tokens=32768,  # the recipe's value; --engine-kwargs overrides
        **lora,
    )
    if a.profile:
        kwargs["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": a.profile_dir}
    kwargs.update(json.loads(a.engine_kwargs))
    print(f"[diag] engine kwargs: {kwargs}", flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    t_ready = time.time()
    print(f"[diag] engine ready", flush=True)
    asyncio.create_task(stats_loop(engine))
    base = argparse.Namespace(
        tp=a.tp, gen=1065, obs=1870, max_context=262144, max_turns=69, top_k=20, prompt_len=2048
    )
    profile_phase, _, profile_s = (a.profile or "").partition(":")
    wanted = a.phases.split(",") if a.phases else None
    for name, conc, obs in [("c640", 640, 1870), ("c640_noobs", 640, 0), ("c64", 64, 1870), ("c256", 256, 1870)]:
        if wanted is None or name in wanted:
            prof = int(profile_s) if name == profile_phase else 0
            await phase(engine, base, name, conc, obs, a.warmup, a.measure, prof)
    print(f"[diag] all phases done {time.time() - t_ready:.0f}s after engine ready", flush=True)
    engine.shutdown()
    os._exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--warmup", type=int, default=120)
    p.add_argument("--measure", type=int, default=120)
    p.add_argument("--phases", default="", help="comma-separated subset of c640,c640_noobs,c64,c256")
    p.add_argument("--engine-kwargs", default="{}", help="JSON merged over the default AsyncEngineArgs")
    p.add_argument("--profile", default="", help="PHASE:SECONDS to torch-profile after that phase's warmup")
    p.add_argument("--profile-dir", default="/tmp/dsv41_vllm_profile")
    asyncio.run(main(p.parse_args()))
