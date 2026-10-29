"""vLLM alone: does a prefix-cache hit change DeepSeek-V4.1's decode logprobs?

For each prompt, greedy-decodes the same continuation three ways in one engine, using cache salts to force a miss
or a hit on the prompt's blocks:

- ``fresh_a`` / ``fresh_b``: unique salts, so the prompt is prefilled by the request itself (two runs: noise floor);
- ``hit``: a salt whose prompt blocks a separate warm-up request (``max_tokens=1``) already cached.

Each set runs as one batch of the same size. Per prompt, logprobs are compared along the common greedy prefix
(until the two token sequences diverge). ``num_cached_tokens`` confirms the hits.

    uv run --isolated --extra megatron python prefix_hit_vs_fresh_v41.py <model> <rows.pt> [--max-tokens 512]
"""

import argparse

import torch


def _apply_skyrl_workspace_patch(worker) -> bool:
    # The V4.1 prefill NaN fix (vllm#60613, not in 0.32.0); NaN first tokens would confound the comparison.
    from skyrl.backends.skyrl_train.patches.vllm.patch_deepseek_v41_prefill_workspace import (
        apply_deepseek_v41_prefill_workspace_patch,
    )

    return apply_deepseek_v41_prefill_workspace_patch()


def compare(a, b):
    """Per-token |logprob| gaps over the common greedy prefix of two outputs, and the divergence index."""
    ta, tb = list(a.token_ids), list(b.token_ids)
    n = 0
    while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
        n += 1
    gaps = [abs(a.logprobs[i][ta[i]].logprob - b.logprobs[i][tb[i]].logprob) for i in range(n)]
    # The first differing token's logprob gap is informative too (a near-tie flip or a real gap).
    first_div = None
    if n < min(len(ta), len(tb)):
        first_div = (a.logprobs[n][ta[n]].logprob, b.logprobs[n][tb[n]].logprob)
    return torch.tensor(gaps), n, first_div


def summarize(name, results):
    gaps = torch.cat([g for g, _, _ in results if g.numel()])
    common = [n for _, n, _ in results]
    print(f"PHF {name}: {gaps.numel():,} common-prefix tokens over {len(results)} prompts")
    print(f"  |dlogprob| mean {gaps.mean():.5f}  median {gaps.median():.6f}  max {gaps.max():.3f}")
    for t in (0.01, 0.1, 0.5, 1):
        print(f"  > {t}: {int((gaps > t).sum())} ({float((gaps > t).float().mean()):.1e})")
    print(f"  common greedy prefix length: median {int(torch.tensor(common).median())}, min {min(common)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("rows")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = torch.load(args.rows)
    prompts = [ids[: len(ids) - L] for ids, L in zip(rows["input_ids"], rows["resp_len"])]
    prompts = [list(x) for x in dict.fromkeys(tuple(x) for x in prompts)]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=max(map(len, prompts)) + args.max_tokens + 16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        logprobs_mode="raw_logprobs",
        enable_prefix_caching=True,
        attention_backend="FLASHMLA_SPARSE_DSV41",
        seed=0,
    )
    print("PHF workspace patch:", llm.collective_rpc(_apply_skyrl_workspace_patch), flush=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=0)

    def run(tag, params=greedy):
        reqs = [TokensPrompt(prompt_token_ids=x, cache_salt=f"{tag}-{i}") for i, x in enumerate(prompts)]
        return llm.generate(reqs, params)

    fresh_a = run("fresh_a")
    fresh_b = run("fresh_b")
    run("shared", SamplingParams(temperature=0.0, max_tokens=1))  # warm the shared salt's prompt blocks
    hit = run("shared")

    hits = [o.num_cached_tokens or 0 for o in hit]
    misses = [o.num_cached_tokens or 0 for o in fresh_a]
    lens = [len(x) for x in prompts]
    print(
        f"PHF {len(prompts)} prompts (median {int(torch.tensor(lens).median())} tokens); cached prompt tokens: "
        f"hit median {int(torch.tensor(hits).median())} (min {min(hits)}), fresh_a max {max(misses)}"
    )

    floor = [compare(a.outputs[0], b.outputs[0]) for a, b in zip(fresh_a, fresh_b)]
    hit_vs = [compare(a.outputs[0], h.outputs[0]) for a, h in zip(fresh_a, hit)]
    summarize("fresh_a vs fresh_b (noise floor)", floor)
    summarize("fresh_a vs hit", hit_vs)
    print("PHF first-token |dlogprob| (fresh_a vs fresh_b, fresh_a vs hit):")
    first = [
        (
            abs(
                a.outputs[0].logprobs[0][a.outputs[0].token_ids[0]].logprob
                - b.outputs[0].logprobs[0][b.outputs[0].token_ids[0]].logprob
            ),
            abs(
                a.outputs[0].logprobs[0][a.outputs[0].token_ids[0]].logprob
                - h.outputs[0].logprobs[0][h.outputs[0].token_ids[0]].logprob
            ),
        )
        for a, b, h in zip(fresh_a, fresh_b, hit)
        if a.outputs[0].token_ids[0] == b.outputs[0].token_ids[0] == h.outputs[0].token_ids[0]
    ]
    f = torch.tensor(first)
    print(
        f"  floor mean {f[:, 0].mean():.5f} max {f[:, 0].max():.4f}; hit mean {f[:, 1].mean():.5f} max {f[:, 1].max():.4f}"
    )
    worst = sorted(
        ((float(g.max()), i, int(g.argmax()), n) for i, (g, n, _) in enumerate(hit_vs) if g.numel()), reverse=True
    )[:10]
    print("PHF worst hit gaps (|dlogprob|, prompt, token index, common prefix length):")
    for w in worst:
        print(f"  {w[0]:.4f}  prompt {w[1]:>3}  tok {w[2]:>4}  common {w[3]}")


if __name__ == "__main__":
    main()
