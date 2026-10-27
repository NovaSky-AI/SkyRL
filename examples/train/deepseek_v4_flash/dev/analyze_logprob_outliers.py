"""Where do the rollout-vs-trainer logprob outliers come from?

Reads a ``trainer.dump_data_batch=true`` dump (``{export_path}/dumped_data/global_step_N_training_input.pkl``),
which holds both logprobs of every sampled response token: ``rollout_logprobs`` (vLLM) and ``action_log_probs``
(the trainer's forward pass), with ``loss_mask`` over the response. Prints the |diff| distribution, which side is
the confident one for the worst tokens, how they cluster by sequence and position, and the worst tokens in
context.

    uv run --isolated --extra megatron python examples/train/deepseek_v4_flash/dev/analyze_logprob_outliers.py \\
        <dump.pkl> --tokenizer <model path> [--top 25] [--context 12]
"""

import argparse
import pickle

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--context", type=int, default=12, help="response tokens of context on each side")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    with open(args.dump, "rb") as f:
        batch = pickle.load(f)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rollout = batch["rollout_logprobs"].float()
    train = batch["action_log_probs"].float()
    mask = batch["loss_mask"] > 0
    num_actions = rollout.shape[1]
    response_ids = batch["sequences"][:, -num_actions:]
    # Each full sequence (prompt + response) is left-padded, so the response is right-aligned in the action columns.
    abs_pos = (batch["attention_mask"].cumsum(dim=1) - 1)[:, -num_actions:]
    resp_start = mask.float().argmax(dim=1)

    diff = (rollout - train).abs()
    d = diff[mask]
    n = d.numel()
    print(f"{rollout.shape[0]} sequences, {n:,} response tokens")
    print(
        f"mean {d.mean():.4f}  median {d.median():.5f}  p99 {d.quantile(0.99) if n < 2**24 else float('nan'):.4f}  max {d.max():.3f}"
    )
    for t in (0.1, 0.5, 1, 2, 5, 10):
        c = int((d > t).sum())
        print(f"  |diff| > {t:>4}: {c:>8,} tokens ({c / n:.2e})")

    # Which engine is the confident one for big diffs, and how likely the token was.
    big = mask & (diff > 1)
    if big.any():
        r, m = rollout[big], train[big]
        print(f"\n{int(big.sum())} tokens with |diff| > 1:")
        print(
            f"  vLLM more confident (rollout > trainer): {int((r > m).sum())}, trainer more confident: {int((m > r).sum())}"
        )
        print(f"  vLLM logprob of those tokens: median {r.median():.2f}, trainer: median {m.median():.2f}")
        print(f"  vLLM p >= 0.1 (a likely token, not tail sampling): {int((r > -2.303).sum())}")
        per_seq = big.sum(dim=1)
        rows = sorted(int(r) for r in per_seq.nonzero().flatten())
        print(f"  spread over {len(rows)} sequences (rows {rows}); max in one sequence {int(per_seq.max())}")
        print(f"  absolute positions (prompt + response): {sorted(int(x) for x in abs_pos[big])}")

    # |diff| by absolute position, to rule a long-context effect in or out.
    print("\nBy absolute position:")
    edges = [0, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 1 << 20]
    p = abs_pos[mask]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            x = d[sel]
            print(
                f"  [{lo:>5}, {hi:>7}): {int(sel.sum()):>9,} tokens  mean {x.mean():.4f}  frac > 1 {(x > 1).float().mean():.1e}"
            )

    print(f"\nTop {args.top} tokens:")
    flat = torch.where(mask, diff, torch.zeros_like(diff)).flatten()
    for idx in flat.topk(min(args.top, n)).indices.tolist():
        row, col = divmod(idx, num_actions)
        lo, hi = max(0, col - args.context), min(num_actions, col + args.context + 1)
        ids = response_ids[row]
        before = tok.decode(ids[lo:col].tolist())
        token = tok.decode([int(ids[col])])
        after = tok.decode(ids[col + 1 : hi].tolist())
        print(
            f"  seq {row:>5} token {col - int(resp_start[row]):>5}/{int(mask[row].sum()):>5} (abs {int(abs_pos[row, col]):>5})  "
            f"diff {diff[row, col]:6.2f}  "
            f"vLLM {rollout[row, col]:8.3f}  trainer {train[row, col]:8.3f}  "
            f"...{before!r} [[{token!r}]] {after!r}..."
        )


if __name__ == "__main__":
    main()
