"""vLLM alone: do DeepSeek-V4.1's decode-time logprobs match a prefill re-score of the same tokens?

Rollout-vs-trainer logprob outliers re-score to the trainer's value in a vLLM prefill, so they come from the
decode path. This samples a response for each prompt (the prompt part of ``rows.pt`` rows, see
``rescore_vllm_v41.py``), records each sampled token's decode-time logprob, re-scores prompt + response with
``prompt_logprobs`` in one prefill, and compares the two per token. Engine knobs bisect the decode path.

    uv run --isolated --extra megatron python decode_vs_prefill_v41.py <model> <rows.pt> \\
        [--prefix-caching] [--no-swa-bounded-replay] [--enforce-eager] [--max-num-seqs N] [--max-tokens 1024] [--repeats 2]
        [--skyrl-patches]
"""

import argparse

import torch


# SkyRL's worker extension installs SkyRL's vLLM patches in every worker at import (incl. the V4.1 prefill NaN fix).
WORKER_EXTENSION = "skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("rows")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=2, help="samples per prompt")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=0)
    p.add_argument("--skyrl-patches", action="store_true", help="load SkyRL's vLLM patches (worker extension)")
    p.add_argument(
        "--no-swa-bounded-replay",
        action="store_true",
        help="keep V4.1's sliding-window KV in the prefix cache instead of recomputing the hit's last window",
    )
    p.add_argument(
        "--prefix-caching",
        action="store_true",
        help="as SkyRL's rollout engines run: the samples of a prompt reuse its cached KV (the re-score never does)",
    )
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = torch.load(args.rows)
    prompts = [ids[: len(ids) - L] for ids, L in zip(rows["input_ids"], rows["resp_len"])]
    prompts = list(dict.fromkeys(tuple(x) for x in prompts))  # 12 samples per prompt share it
    extra = {"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}
    if args.skyrl_patches:
        extra["worker_extension_cls"] = WORKER_EXTENSION
    if args.no_swa_bounded_replay:
        extra["swa_bounded_replay"] = False
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=max(len(x) for x in prompts) + args.max_tokens + 16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        logprobs_mode="raw_logprobs",
        enable_prefix_caching=args.prefix_caching,
        attention_backend="FLASHMLA_SPARSE_DSV41",
        enforce_eager=args.enforce_eager,
        seed=0,
        **extra,
    )
    print(f"DVP prefix_caching={args.prefix_caching} swa_bounded_replay={not args.no_swa_bounded_replay}", flush=True)

    gen = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, logprobs=0, n=args.repeats, seed=1)
    outs = llm.generate([TokensPrompt(prompt_token_ids=list(x)) for x in prompts], gen)
    seqs, decode_lp, plens = [], [], []
    for prompt, out in zip(prompts, outs):
        for c in out.outputs:
            toks = list(c.token_ids)
            seqs.append(list(prompt) + toks)
            plens.append(len(prompt))
            decode_lp.append([lp[t].logprob for lp, t in zip(c.logprobs, toks)])
    print(f"DVP sampled {len(seqs)} responses, {sum(map(len, decode_lp)):,} tokens", flush=True)

    score = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    # A unique cache salt per request keeps the reference prefill off any cached prefix.
    res = llm.generate([TokensPrompt(prompt_token_ids=s, cache_salt=f"rescore-{i}") for i, s in enumerate(seqs)], score)
    diffs, worst = [], []
    for i, (s, o) in enumerate(zip(seqs, res)):
        pre = [o.prompt_logprobs[j][s[j]].logprob for j in range(plens[i], len(s))]
        d = (torch.tensor(decode_lp[i]) - torch.tensor(pre)).abs()
        diffs.append(d)
        for j in d.topk(min(3, d.numel())).indices.tolist():
            worst.append((float(d[j]), i, j, decode_lp[i][j], pre[j]))
    d = torch.cat(diffs)
    print(
        f"DVP |decode - prefill| over {d.numel():,} tokens: mean {d.mean():.4f}  median {d.median():.5f}  max {d.max():.3f}"
    )
    for t in (0.1, 0.5, 1, 2, 5):
        print(f"  > {t}: {int((d > t).sum())} ({float((d > t).float().mean()):.1e})")
    per_seq = [int((x > 1).sum()) for x in diffs]
    print(
        f"  sequences with any > 1: {sum(c > 0 for c in per_seq)} / {len(per_seq)}; per-sequence counts {sorted(per_seq)[-10:]}"
    )
    print("DVP worst tokens (|diff|, seq, response index, decode lp, prefill lp):")
    for w in sorted(worst, reverse=True)[:15]:
        print(f"  {w[0]:6.2f}  seq {w[1]:>3}  tok {w[2]:>5}  decode {w[3]:8.3f}  prefill {w[4]:8.3f}")


if __name__ == "__main__":
    main()
