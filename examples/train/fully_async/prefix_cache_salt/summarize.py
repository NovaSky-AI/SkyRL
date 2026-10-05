"""Summarize prefix-cache reuse and rollout-vs-train logprob gaps for one or more reproduction runs.

uv run --isolated examples/train/fully_async/prefix_cache_salt/summarize.py <run_dir> [<run_dir> ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run_dir: Path) -> None:
    requests = read_jsonl(run_dir / "requests.jsonl")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    print(
        f"\n=== {run_dir}  ({len(requests)} LLM calls; client salts requests: "
        f"{any(r['client_salts'] for r in requests)})"
    )

    # A turn sent after a sync can only reuse its own earlier turns' blocks if they were computed under the
    # previous weights: cached tokens on those turns are stale prefix reuse.
    print(
        f"{'weight_version':>14} {'turns':>6} {'cached/prompt':>14} {'after-sync turns':>17} {'their cached/prompt':>20}"
    )
    by_version = defaultdict(list)
    for r in requests:
        by_version[r["weight_version"]].append(r)
    for version in sorted(by_version):
        rows = by_version[version]
        crossed = [r for r in rows if r["turn_after_sync"]]

        def frac(rs):
            prompt = sum(r["prompt_tokens"] for r in rs)
            return f"{sum(r['cached_tokens'] for r in rs) / prompt:.3f}" if prompt else "-"

        print(f"{version:>14} {len(rows):>6} {frac(rows):>14} {len(crossed):>17} {frac(crossed):>20}")

    print(f"{'step':>5} {'logprob |diff| mean':>20} {'max':>10}")
    for m in metrics:
        mean = m.get("policy/rollout_train_logprobs_abs_diff_mean")
        high = m.get("policy/rollout_train_logprobs_abs_diff_max")
        print(
            f"{m['step']:>5} {mean if mean is None else f'{mean:.4f}':>20} {high if high is None else f'{high:.3f}':>10}"
        )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(Path(arg))
