"""Re-score rollout sequences with vLLM's prompt logprobs, to see which path the rollout-vs-trainer outliers come from.

Input: a ``.pt`` of rows (``input_ids`` = prompt + response, ``resp_len``, and the rollout (vLLM) and trainer
(Megatron) logprobs of the response tokens), e.g. extracted from ``trainer.dump_data_batch`` dumps. Every token
is re-scored in one prefill (``prompt_logprobs``), batched and then one request at a time, and compared with both
the rollout's value (sampled token by token in decode) and the trainer's. On the tokens where rollout and trainer
disagree by more than 1, it reports which side each re-score lands on.

    run_probe_any.sh-style env, one 8-GPU node:
    uv run --isolated --extra megatron python rescore_vllm_v41.py <model> <rows.pt> [--skyrl-workspace-patch]
"""

import argparse
import time

import torch


def _apply_skyrl_workspace_patch(worker) -> bool:
    from skyrl.backends.skyrl_train.patches.vllm.patch_deepseek_v41_prefill_workspace import (
        apply_deepseek_v41_prefill_workspace_patch,
    )

    return apply_deepseek_v41_prefill_workspace_patch()


def _token_logprobs(output, ids, resp_len):
    """Logprob of each actual response token from a prompt_logprobs result."""
    plp = output.prompt_logprobs
    start = len(ids) - resp_len
    return [plp[i][ids[i]].logprob for i in range(start, len(ids))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("rows")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--skyrl-workspace-patch", action="store_true")
    p.add_argument("--single-limit", type=int, default=40, help="rows to re-score one request at a time")
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="small values chunk each prefill, so most tokens read their context back from the KV cache (as decode does)",
    )
    p.add_argument("--step", default="", help="only rows from this dump step (e.g. 1: scored with the same weights)")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = torch.load(args.rows)
    if args.step:
        keep = [i for i, t in enumerate(rows["tag"]) if t.split(":")[0] == args.step]
        rows = {k: [v[i] for i in keep] for k, v in rows.items()}
    n = len(rows["input_ids"])
    extra = {}
    if args.max_num_batched_tokens:
        extra = dict(max_num_batched_tokens=args.max_num_batched_tokens, enable_chunked_prefill=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=max(len(x) for x in rows["input_ids"]) + 16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        logprobs_mode="processed_logprobs",
        enable_prefix_caching=False,
        attention_backend="FLASHMLA_SPARSE_DSV41",
        seed=0,
        **extra,
    )
    if args.skyrl_workspace_patch:
        print("RESCORE workspace patch:", llm.collective_rpc(_apply_skyrl_workspace_patch), flush=True)
    params = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in rows["input_ids"]]

    t = time.time()
    batched = [
        _token_logprobs(o, ids, L)
        for o, ids, L in zip(llm.generate(prompts, params), rows["input_ids"], rows["resp_len"])
    ]
    print(f"RESCORE batched {n} rows in {time.time() - t:.0f}s", flush=True)
    single = {}
    for i in range(min(n, args.single_limit)):
        o = llm.generate([prompts[i]], params, use_tqdm=False)[0]
        single[i] = _token_logprobs(o, rows["input_ids"][i], rows["resp_len"][i])

    def summarize(name, scores, idx):
        roll = torch.cat([torch.tensor(rows["rollout"][i]) for i in idx])
        train = torch.cat([torch.tensor(rows["trainer"][i]) for i in idx])
        new = torch.cat([torch.tensor(scores[i]) for i in idx]).float()
        big = (roll - train).abs() > 1
        print(f"\n{name}: {len(idx)} rows, {roll.numel():,} tokens")
        print(
            f"  mean |rescore - rollout| {(new - roll).abs().mean():.4f}   |rescore - trainer| {(new - train).abs().mean():.4f}   |rollout - trainer| {(roll - train).abs().mean():.4f}"
        )
        if big.any():
            nr, nt = (new - roll).abs()[big], (new - train).abs()[big]
            print(
                f"  on the {int(big.sum())} tokens with |rollout - trainer| > 1: rescore closer to rollout on {int((nr < nt).sum())}, to trainer on {int((nt <= nr).sum())}"
            )
            print(f"    mean |rescore - rollout| {nr.mean():.3f}   |rescore - trainer| {nt.mean():.3f}")

    for tag in ("outlier", "control"):
        idx = [i for i in range(n) if rows["tag"][i].endswith(tag)]
        summarize(f"batched / {tag} rows", batched, idx)
        sidx = [i for i in idx if i in single]
        if sidx:
            summarize(f"single-request / {tag} rows", single, sidx)


if __name__ == "__main__":
    main()
