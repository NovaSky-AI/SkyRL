"""Summarize ultra-sweep results.jsonl into a throughput/memory table.

Usage: python analyze_sweep.py /home/ray/ultra_sweep/results.jsonl
For each tag, reports the steady-state (non-warmup, non-error) median step time,
tokens/s, max peak reserved GB, min free GB, and OOM/error status.
"""

import json
import sys
from collections import defaultdict
from statistics import median


def main(path):
    by_tag = defaultdict(list)
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["tag"] not in by_tag:
                order.append(r["tag"])
            by_tag[r["tag"]].append(r)

    hdr = (
        f"{'tag':<34} {'tp':>2} {'pp':>2} {'cp':>2} {'ep':>3} {'dp':>2} "
        f"{'mtpm':>8} {'mode':>7} {'maxseq':>7} {'tok/step':>9} "
        f"{'step_s':>7} {'tok/s':>9} {'pkRsv':>6} {'minFree':>7} {'status':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag in order:
        recs = by_tag[tag]
        steady = [r for r in recs if not r.get("warmup") and not r.get("error")]
        oom = any(r.get("oom") for r in recs)
        err = next((r["error"] for r in recs if r.get("error")), None)
        any0 = recs[0]
        if steady:
            st = median([r["step_time_s"] for r in steady])
            tps = median([r["tokens_per_s"] for r in steady if r["tokens_per_s"]])
            pk = max([r["peak_reserved_gb"] for r in steady if r["peak_reserved_gb"]] or [0])
            mf = min([r["min_free_gb"] for r in steady if r["min_free_gb"]] or [0])
            status = "OK"
        else:
            st = tps = pk = mf = 0
            # use warmup peak if present
            pk = max([r.get("peak_reserved_gb") or 0 for r in recs] or [0])
            mf = min([r.get("min_free_gb") or 99 for r in recs] or [0])
            status = "OOM" if oom else ("ERR" if err else "NONE")
        print(
            f"{tag:<34} {any0['tp']:>2} {any0['pp']:>2} {any0['cp']:>2} {any0['ep']:>3} {any0['dp']:>2} "
            f"{any0['mtpm']:>8} {any0.get('mode',''):>7} {any0.get('max_seqlen',0):>7} "
            f"{any0.get('total_tokens',0):>9} {st:>7.1f} {tps:>9.0f} {pk:>6.1f} {mf:>7.1f} {status:>8}"
        )
        if err and status != "OK":
            print(f"    └─ {err[:160]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/ray/ultra_sweep/results.jsonl")
