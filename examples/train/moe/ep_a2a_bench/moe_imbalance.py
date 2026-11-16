"""Expert-parallel load imbalance from a dumped SkyRL training batch (CPU only).

Collect the input with one GPU run of the recipe plus ``trainer.dump_data_batch=true``
(router replay on, so the batch carries ``rollout_expert_indices``). Each step writes
``{export_path}/dumped_data/global_step_{N}_training_input.pkl``. Then, from the repo root:

    uv run --isolated --extra skyrl-train python moe_imbalance.py DUMP.pkl [DUMP2.pkl ...] \
        --dp 16 --ep 16 --num-experts 512 --max-tokens-per-microbatch 10752 --gpus-per-node 8

What it does: replays SkyRL's own split of the batch (per mini-batch, contiguous equal-count
chunks per DP rank as in ``MeshDispatch.stage_chunks``; per rank, the "balanced" token packer as
in ``TokenBasedBatchIterator``; every rank runs the max microbatch count). Each microbatch slot is
one synchronized all-to-all across an EP group, so per slot it compares ranks:

  A (input)   tokens per rank                      -> fixed by balancing work across DP ranks
  B (routing) token-expert rows received per rank  -> fixed by expert placement / replication

"waste" = 1 - sum(mean) / sum(max): the fraction of rank-time spent waiting if the cost of a
slot is proportional to the slowest rank. It is a count model, not a timing; check it against a
measured all-to-all wait.

It also reports the same metrics for a counterfactual assignment that balances tokens across DP
ranks (greedy largest-first), and dispatch bytes for stock / once-per-rank / once-per-node sends.

Assumes TP1, CP1, PP1, ETP1 and Megatron's default rank order, so DP rank r is EP rank r % ep in
EP group r // ep, and expert x lives on EP rank x // (num_experts / ep).
"""

import argparse
import pickle
import sys
from typing import Dict, List

import numpy as np

from skyrl.train.dataset.bin_packing import make_seq_packer


def load_batch(path: str):
    with open(path, "rb") as f:
        batch = pickle.load(f)
    lengths = batch["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int64)
    routes = batch["rollout_expert_indices"]
    if routes is None:
        sys.exit(f"{path}: no rollout_expert_indices; run with router replay (R3) on")
    values = routes.values.cpu().numpy()
    cu = routes.cu_seqlens.cpu().numpy().astype(np.int64)
    bounds = batch.metadata.get("policy_mini_batch_boundaries") or [(0, len(lengths))]
    return lengths, values, cu, [tuple(b) for b in bounds]


def per_sample_stats(values, cu, n, ep, experts_per_rank, ranks_per_node):
    """Per sample, summed over layers unless noted:

    recv[s, layer, e]     token-expert rows sent to EP rank e (stock dispatcher; per layer)
    rank_present[s, e]    tokens with at least one expert on EP rank e (once-per-rank dispatch)
    node_present[s, m]    tokens with at least one expert on node m (once-per-node dispatch)
    """
    num_layers = values.shape[1]
    nodes = -(-ep // ranks_per_node)
    recv = np.zeros((n, num_layers, ep), dtype=np.int64)
    rank_present = np.zeros((n, ep), dtype=np.int64)
    node_present = np.zeros((n, nodes), dtype=np.int64)
    route_rows = np.zeros(n, dtype=np.int64)
    for s in range(n):
        seg = values[cu[s] : cu[s + 1]].astype(np.int64)  # [tokens, layers, topk]
        route_rows[s] = seg.shape[0]
        if seg.shape[0] == 0:
            continue
        dest = seg // experts_per_rank  # EP rank per (token, layer, k)
        for layer in range(num_layers):
            d = dest[:, layer]  # [tokens, topk]
            recv[s, layer] = np.bincount(d.ravel(), minlength=ep)
            hit = np.zeros((d.shape[0], ep), dtype=bool)
            np.put_along_axis(hit, d, True, axis=1)
            rank_present[s] += hit.sum(axis=0)
            node_hit = hit.reshape(d.shape[0], nodes, -1).any(axis=2) if ep % ranks_per_node == 0 else None
            if node_hit is not None:
                node_present[s] += node_hit.sum(axis=0)
    return recv, rank_present, node_present, route_rows


def contiguous_split(ids: np.ndarray, dp: int) -> List[np.ndarray]:
    if len(ids) % dp:
        # stage_chunks pads with loss-masked dummies; only step-wise training hits this.
        print(f"  note: mini-batch of {len(ids)} not divisible by dp={dp}; last rank gets fewer", file=sys.stderr)
    size = -(-len(ids) // dp)
    return [ids[i * size : (i + 1) * size] for i in range(dp)]


def balanced_split(ids: np.ndarray, lengths: np.ndarray, dp: int) -> List[np.ndarray]:
    """Greedy largest-first onto the least-loaded rank (token cost). Rank counts may differ."""
    load = np.zeros(dp)
    out: List[List[int]] = [[] for _ in range(dp)]
    for s in sorted(ids.tolist(), key=lambda s: -lengths[s]):
        r = int(np.argmin(load))
        out[r].append(s)
        load[r] += lengths[s]
    return [np.array(sorted(o), dtype=np.int64) for o in out]


def microbatch_slots(rank_ids: List[np.ndarray], lengths: np.ndarray, cap: int) -> List[List[np.ndarray]]:
    """slots[i][r] = sample ids of rank r's i-th microbatch (empty = padding microbatch)."""
    packer = make_seq_packer("balanced", bin_capacity=cap)
    per_rank = []
    for ids in rank_ids:
        bins = packer.pack(lengths[ids].tolist()) if len(ids) else []
        per_rank.append([ids[np.array(b, dtype=np.int64)] for b in bins])
    n_slots = max(len(b) for b in per_rank)
    empty = np.zeros(0, dtype=np.int64)
    return [[b[i] if i < len(b) else empty for b in per_rank] for i in range(n_slots)]


def waste(maxes: List[float], means: List[float]) -> float:
    total = float(np.sum(maxes))
    return 1.0 - float(np.sum(means)) / total if total else 0.0


def cross_node_rows(slots, recv, rank_present, node_present, ep, ranks_per_node) -> Dict[str, float]:
    """Rows leaving the sender's node, per dispatch scheme, for one assignment."""
    dp = len(slots[0])
    out = {"stock": 0.0, "per_rank": 0.0, "per_node": 0.0, "stock_all": 0.0}
    rows_by_rank = recv.sum(axis=1)  # [n, ep]
    for slot in slots:
        for r in range(dp):
            ids = slot[r]
            if len(ids) == 0:
                continue
            g, e = divmod(r, ep)
            src_node = (g * ep + e) // ranks_per_node - (g * ep) // ranks_per_node  # node index inside the group
            local = slice(src_node * ranks_per_node, (src_node + 1) * ranks_per_node)
            rows = rows_by_rank[ids].sum(axis=0)
            present = rank_present[ids].sum(axis=0)
            out["stock_all"] += rows.sum()
            out["stock"] += rows.sum() - rows[local].sum()
            out["per_rank"] += present.sum() - present[local].sum()
            npres = node_present[ids].sum(axis=0)
            out["per_node"] += npres.sum() - npres[src_node]
    return out


def analyze(slots, lengths, recv, ep) -> Dict[str, float]:
    dp = len(slots[0])
    groups = [list(range(g * ep, (g + 1) * ep)) for g in range(dp // ep)]
    tok_max, tok_mean, sq_max, sq_mean, rv_max, rv_mean, ratios = [], [], [], [], [], [], []
    for slot in slots:
        for grp in groups:
            toks = np.array([lengths[slot[r]].sum() for r in grp], dtype=np.float64)
            sq = np.array([(lengths[slot[r]].astype(np.float64) ** 2).sum() for r in grp])
            tok_max.append(toks.max())
            tok_mean.append(toks.mean())
            sq_max.append(sq.max())
            sq_mean.append(sq.mean())
            ids = np.concatenate([slot[r] for r in grp])
            if len(ids) == 0:
                continue
            per_layer = recv[ids].sum(axis=0)  # [layers, ep]
            rv_max.extend(per_layer.max(axis=1).tolist())
            rv_mean.extend(per_layer.mean(axis=1).tolist())
            m = per_layer.mean(axis=1)
            ratios.extend((per_layer.max(axis=1)[m > 0] / m[m > 0]).tolist())
    return {
        "slots": len(slots),
        "A_tokens_waste": waste(tok_max, tok_mean),
        "A_attn_L2_waste": waste(sq_max, sq_mean),
        "B_recv_waste": waste(rv_max, rv_mean),
        "B_recv_max_over_mean_p50": float(np.median(ratios)) if ratios else 0.0,
        "B_recv_max_over_mean_p90": float(np.percentile(ratios, 90)) if ratios else 0.0,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", nargs="+")
    p.add_argument("--dp", type=int, required=True)
    p.add_argument("--ep", type=int, required=True)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--max-tokens-per-microbatch", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=8)
    a = p.parse_args(argv)
    assert a.dp % a.ep == 0 and a.num_experts % a.ep == 0
    epr = a.num_experts // a.ep

    for path in a.dumps:
        lengths, values, cu, bounds = load_batch(path)
        n, layers, topk = len(lengths), values.shape[1], values.shape[2]
        recv, rank_present, node_present, route_rows = per_sample_stats(values, cu, n, a.ep, epr, a.gpus_per_node)
        print(f"\n== {path}")
        print(f"samples {n}, layers {layers}, top-k {topk}, mini-batches {len(bounds)}, tokens {lengths.sum()}")
        print(
            f"sequence length p10/p50/p90/max: {np.percentile(lengths, [10, 50, 90]).astype(int).tolist()} {lengths.max()}"
        )
        missing = int((lengths - route_rows).sum())
        if missing:
            print(f"tokens without route rows (ignored): {missing} ({missing / lengths.sum():.2%})")

        # Expert hotness, whole batch: busiest expert vs mean, per layer.
        expert_counts = np.stack(
            [np.bincount(values[:, l].ravel().astype(np.int64), minlength=a.num_experts) for l in range(layers)]
        )
        hot = expert_counts.max(axis=1) / expert_counts.mean(axis=1)
        print(f"busiest expert / mean expert, over layers: p50 {np.median(hot):.2f}, max {hot.max():.2f}")

        stock = int(route_rows.sum()) * layers * topk
        print(
            f"dispatch rows per step, all layers, any destination: stock {stock:,}; "
            f"once per rank {int(rank_present.sum()):,} ({1 - rank_present.sum() / stock:.1%} fewer)"
        )

        for name, split in (("current (contiguous)", contiguous_split), ("token-balanced", None)):
            agg: Dict[str, List[float]] = {}
            cross = {"stock": 0.0, "per_rank": 0.0, "per_node": 0.0, "stock_all": 0.0}
            for lo, hi in bounds:
                ids = np.arange(lo, hi, dtype=np.int64)
                rank_ids = split(ids, a.dp) if split else balanced_split(ids, lengths, a.dp)
                slots = microbatch_slots(rank_ids, lengths, a.max_tokens_per_microbatch)
                for k, v in analyze(slots, lengths, recv, a.ep).items():
                    agg.setdefault(k, []).append(v)
                if split:  # cross-node bytes depend on placement only through the sender's node
                    for k, v in cross_node_rows(slots, recv, rank_present, node_present, a.ep, a.gpus_per_node).items():
                        cross[k] += v
            if split and cross["stock"]:
                print(
                    f"rows leaving the sender's node: stock {cross['stock']:,.0f} "
                    f"({cross['stock'] / cross['stock_all']:.0%} of all rows); once per rank "
                    f"{1 - cross['per_rank'] / cross['stock']:.1%} fewer; once per node "
                    f"{1 - cross['per_node'] / cross['stock']:.1%} fewer"
                )
            mean = {k: float(np.mean(v)) for k, v in agg.items()}
            print(
                f"[{name}] microbatch slots/mini-batch {mean['slots']:.1f} | "
                f"A tokens waste {mean['A_tokens_waste']:.1%}, A attention(L^2) waste {mean['A_attn_L2_waste']:.1%} | "
                f"B recv waste {mean['B_recv_waste']:.1%}, recv max/mean p50 {mean['B_recv_max_over_mean_p50']:.2f} "
                f"p90 {mean['B_recv_max_over_mean_p90']:.2f}"
            )


if __name__ == "__main__":
    main()
