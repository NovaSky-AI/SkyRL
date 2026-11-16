"""Turn a dumped SkyRL training batch into all-to-all traffic matrices (CPU only).

One expert-parallel (EP) all-to-all moves, for one MoE layer of one microbatch, ``M[src, dst]``
token-expert rows from EP rank ``src`` to EP rank ``dst``. This script replays SkyRL's split of the
batch (see ``moe_imbalance.py``) and writes every such matrix:

    uv run --isolated --extra skyrl-train python extract_routes.py DUMP.pkl --out routes.npz \
        --dp 16 --ep 16 --num-experts 512 --max-tokens-per-microbatch 10752

Output ``routes.npz``: ``matrices`` int32 [calls, ep, ep] and ``meta`` int32 [calls, 4] =
(mini_batch, microbatch_slot, ep_group, layer). ``a2a_bench.py`` replays the file; it needs no
model and no dump. Same assumptions as ``moe_imbalance.py`` (TP1, CP1, PP1, ETP1).
"""

import argparse

import numpy as np
from moe_imbalance import contiguous_split, load_batch, microbatch_slots, per_sample_stats


def build_matrices(lengths, values, cu, bounds, dp, ep, num_experts, cap):
    n = len(lengths)
    # gpus-per-node only feeds the node statistics, which this script does not use.
    recv, _, _, _ = per_sample_stats(values, cu, n, ep, num_experts // ep, ep)
    layers = recv.shape[1]
    matrices, meta = [], []
    for mb, (lo, hi) in enumerate(bounds):
        rank_ids = contiguous_split(np.arange(lo, hi, dtype=np.int64), dp)
        for slot_idx, slot in enumerate(microbatch_slots(rank_ids, lengths, cap)):
            for g in range(dp // ep):
                # rows[src, layer, dst]: what EP rank `src` of this group sends for this microbatch
                rows = np.stack([recv[slot[g * ep + e]].sum(axis=0) for e in range(ep)])
                for layer in range(layers):
                    matrices.append(rows[:, layer, :])
                    meta.append((mb, slot_idx, g, layer))
    return np.asarray(matrices, dtype=np.int32), np.asarray(meta, dtype=np.int32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--out", required=True)
    p.add_argument("--dp", type=int, required=True)
    p.add_argument("--ep", type=int, required=True)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--max-tokens-per-microbatch", type=int, required=True)
    a = p.parse_args(argv)
    assert a.dp % a.ep == 0 and a.num_experts % a.ep == 0

    lengths, values, cu, bounds = load_batch(a.dump)
    matrices, meta = build_matrices(lengths, values, cu, bounds, a.dp, a.ep, a.num_experts, a.max_tokens_per_microbatch)
    np.savez_compressed(a.out, matrices=matrices, meta=meta, topk=values.shape[2], num_experts=a.num_experts)
    rows = matrices.sum(axis=(1, 2))
    print(f"wrote {a.out}: {len(matrices)} all-to-all calls, ep {a.ep}, layers {values.shape[1]}")
    print(f"rows per call: p50 {int(np.median(rows)):,}, max {int(rows.max()):,}; total {int(rows.sum()):,}")


if __name__ == "__main__":
    main()
