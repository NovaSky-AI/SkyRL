"""Long-horizon multi-turn decode throughput of one vLLM engine serving DeepSeek-V4.1-Flash, depthfirst's shape.

Each slot's first session is one agent rollout already DEPTH tokens into its context (synthetic history, depth
uniform in [0, MAX_DEPTH]; later sessions start from a PROMPT_LEN prompt) and then runs turns of exactly GEN generated tokens (EOS ignored) followed by OBS fresh
observation tokens, for MAX_TURNS turns or until the context passes MAX_CONTEXT; a finished session is replaced by a new one, so
CONCURRENCY sessions are always in flight. Each turn resends the whole context, so prefix caching serves the
history as it does for a real multi-turn rollout.

The engine is built as the RL recipe builds it (released FP8 / MXFP4 weights, TP, LoRA layers enabled so the
MoE takes the LoRA-capable kernel path, SWA bounded replay off, async scheduling). No adapter is active, so
the LoRA matmuls themselves are not timed: a lower bound on the LoRA cost.

After WARMUP_S the script counts generated and prompt tokens over MEASURE_S and prints tokens/s per GPU.

    python dsv41_inference_bench.py --model <dir> [--tp 8] [--concurrency 640] [--gen 1065] [--obs 1870]
"""

import argparse
import asyncio
import os
import random
import time
import uuid

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

VOCAB_LO, VOCAB_HI = 1000, 100000


class Stats:
    def __init__(self):
        self.gen = 0
        self.prompt = 0
        self.turns = 0
        self.sessions = 0
        self.counting = False
        self.depth = {}  # worker id -> current context length of its in-flight session

    def add(self, gen: int, prompt: int) -> None:
        if self.counting:
            self.gen += gen
            self.prompt += prompt
            self.turns += 1


async def session(engine: AsyncLLM, args, stats: Stats, rng: random.Random, depth: int, wid: int) -> None:
    context = [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(depth)]
    params = SamplingParams(
        max_tokens=args.gen, min_tokens=args.gen, ignore_eos=True, temperature=1.0, top_k=args.top_k, logprobs=0
    )
    turns = 0
    while turns < args.max_turns and len(context) + args.gen + args.obs < args.max_context:
        stats.depth[wid] = len(context)
        final = None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=context), params, str(uuid.uuid4())):
            final = out
        generated = list(final.outputs[0].token_ids)
        stats.add(len(generated), len(context))
        context = context + generated + [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(args.obs)]
        turns += 1
    stats.sessions += 1


async def worker(engine, args, stats, seed, stop: asyncio.Event) -> None:
    rng = random.Random(seed)
    # Only the first rollout of each slot starts deep (a one-time uncached prefill during warmup); later ones start
    # from a fresh prompt and grow turn by turn, so in steady state depths spread over [0, MAX_DEPTH] with the
    # history served from the prefix cache, as in real rollouts.
    depth = rng.randrange(args.min_depth, args.max_depth + 1)
    while not stop.is_set():
        await session(engine, args, stats, rng, depth, seed)
        depth = args.prompt_len


async def main(args) -> None:
    lora = {"enable_lora": True, "max_lora_rank": 32, "max_loras": 1} if args.lora else {}
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_context,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        swa_bounded_replay=False,
        async_scheduling=True,
        distributed_executor_backend="mp",
        disable_log_stats=False,
        **lora,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    stats = Stats()
    stop = asyncio.Event()
    tasks = [asyncio.create_task(worker(engine, args, stats, i, stop)) for i in range(args.concurrency)]
    print(f"[bench] {args.concurrency} sessions started; warmup {args.warmup_s}s", flush=True)
    await asyncio.sleep(args.warmup_s)
    stats.counting = True
    t0 = time.time()
    for _ in range(args.measure_s // 60):
        await asyncio.sleep(60)
        dt = time.time() - t0
        print(
            f"[bench] {dt:.0f}s gen {stats.gen / dt:.0f} tok/s ({stats.gen / dt / args.tp:.0f}/GPU) "
            f"prompt {stats.prompt / dt:.0f} tok/s turns {stats.turns} "
            f"mean in-flight context {sum(stats.depth.values()) / max(len(stats.depth), 1):.0f}",
            flush=True,
        )
    dt = time.time() - t0
    print(
        f"[bench] RESULT tp={args.tp} concurrency={args.concurrency} gen={args.gen} obs={args.obs} "
        f"depth=[{args.min_depth},{args.max_depth}] lora={args.lora}: gen {stats.gen / dt:.0f} tok/s "
        f"= {stats.gen / dt / args.tp:.0f} tok/s/GPU; prompt (incl. cached) {stats.prompt / dt:.0f} tok/s; "
        f"{stats.turns} turns in {dt:.0f}s",
        flush=True,
    )
    stop.set()
    for t in tasks:
        t.cancel()
    # Shut the engine down before the hard exit: os._exit alone leaves the EngineCore / TP worker processes
    # alive, holding the GPUs.
    engine.shutdown()
    os._exit(0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=640)
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument("--gen", type=int, default=1065)
    p.add_argument("--obs", type=int, default=1870)
    p.add_argument("--min-depth", type=int, default=0)
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--max-depth", type=int, default=200000)
    p.add_argument("--max-context", type=int, default=262144)
    p.add_argument("--max-turns", type=int, default=69)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--lora", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--warmup-s", type=int, default=600)
    p.add_argument("--measure-s", type=int, default=900)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
