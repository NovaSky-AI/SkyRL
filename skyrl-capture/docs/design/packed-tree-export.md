# `tree_samples`: one row per trajectory, with a tree attention mask

Design note. Nothing here is built.

`token_samples` emits one row per root-to-leaf path, so branches that share a
prefix carry that prefix in every row and the trainer pays for the same tokens
once per branch. The graph knows they are the same tokens; the export format
throws that away and the trainer cannot get it back.

`tree_samples` keeps it: **one row per trajectory**, every node's tokens stored
exactly once, plus the structure a trainer needs to attend over it correctly.

This is an addition. `token_samples` does not change, and nothing has to migrate.

## When it is worth building

The saving is not a property of this format. It measures **how much the harness
rewrites its history**, and the *shape* of the rewrite decides whether it
compounds. FLOPs saved, Qwen3-30B-A3B:

| pattern | shape | saving |
| --- | --- | --- |
| No history rewriting | one chain, no fork | **0%**, at any depth |
| Reasoning stripped each turn | fork per turn, shared prefix grows | 40% at T=8, 87% at T=100 |
| Best-of-N at each step | fork per turn, multiplicity | 70% at T=8, 96% at T=100 |
| Summarization / compaction | one fork, shares only the system prompt | ~0% — a correctness win, not a compute one |
| One sub-agent fan-out | one fork, `N` children share context `C` | ceilinged, below |

**The rule: a fork whose shared prefix grows with depth compounds without
bound. A fork sharing a fixed prefix does not.** A one-shot fan-out saves
`(N-1)·C / (N·(C+M))` for `M` tokens of work per child, which tends to
`C/(C+M)`: 91% when children are short against the inherited context, 33% when
they do twice as much work as they inherited. Width stops helping past about
N=8, and turns never enter. **Sub-agents that run long are not worth packing.**

## The shape that compounds

Qwen3's chat template strips `<think>` from every assistant message before the
last user turn, so the history is rewritten *every turn* and capture records a
fork per turn — the rewritten message is a client-authored sibling of the
sampled one. Measured shape, from the 2026-09-15 Harbor run:

```
root: system + task prompt                       1032t
 |-- scaffold + A0 (sampled)                     4080t   <- leaf, branch 0
 `-- c0 (thinking stripped) + tool result         478t   <- spine
      |-- scaffold + A1 (sampled)                1353t   <- leaf, branch 1
      `-- c1 (thinking stripped) + tool result    478t   <- spine
           `-- ...
```

A spine of non-trainable rewritten context, one sampled leaf per turn. Branch
`k` carries the whole spine up to turn `k`, so:

* **unpacked** grows **O(T²)** in tokens, **O(T³)** in attention pairs
* **packed** grows **O(T)** and **O(T²)**
* **trainable tokens are identical** — only context duplicates

Node model validated against the export at T=2 (5112 + 2863 = 7975 tokens, to
the token), then scaled on the same measured node sizes:

| turns | unpacked | packed | tokens saved | FLOPs saved | useful work, unpacked -> packed |
| --- | --- | --- | --- | --- | --- |
| 2 | 7,975 | 6,943 | 12.9% | 10.8% | 68.1% -> 78.2% |
| 8 | 39,589 | 22,327 | 43.6% | 38.7% | 45.3% -> 80.3% |
| 32 | 338,125 | 83,863 | 75.2% | 70.0% | 20.1% -> 81.0% |
| 64 | 1,164,461 | 165,911 | 85.8% | 81.6% | 11.6% -> 81.1% |

**The last column is the real argument**, not the percentage saved. "Useful
work" is trainable tokens over tokens pushed through the network. Unpacked it
decays as `1/T` — at 64 turns, 88% of the forward pass is re-encoding context
the model has already seen. Packed it is flat at ~81% at any depth. Per-branch
export makes the forward pass quadratic in turns for a linear amount of
gradient.

SkyRL's step-wise baseline has a partial answer already: `merge_stepwise_output`
(`skyrl/train/generators/utils.py`) greedily merges consecutive turns whenever
`prompt[i] + response[i]` is a prefix of `prompt[i+1]`, producing one row with
zeros at the observation deltas. **Where it applies it is exactly equivalent to
this proposal** — measured on a linear-history run (Qwen3-4B-Instruct, 4 turns):
4 rows -> 1, 20,522 -> 7,207 tokens, token-identical to capture's single branch.

Its limit is the prefix condition. When the harness rewrites history the merge
flushes and starts a new group, so it becomes a silent no-op: on the 30B
Thinking run it merged **10 rows into 10, saving 0.0%**.

Two differences are worth stating, because they are what this format is for.

**Derived, not declared.** Prefix merging is a global flag applied after the
rows have been flattened, recovering structure by comparing token prefixes.
SkyRL's default leaves it off, and Harbor hard-requires `step_wise_trajectories`
either way. The graph has no such switch: it is built at capture time from
message identity, so one trajectory yields one row and another yields four
because of what actually happened, not because of how the job was configured.

**They fail differently.** When the prefix condition breaks, merging abandons
the whole boundary and the next group re-carries its entire prompt — sharing
retained: zero. A graph forks at the point of divergence and keeps everything
above it: on the same run, each trajectory still shared its 1035- or 1348-token
prompt across both branches. Partial sharing survives a rewrite; prefix matching
is all-or-nothing.

Three results worth keeping:

* **Attention is not negligible here.** Activation sparsity cuts FFN cost ~10x
  and does nothing to attention, so a sparse MoE has a *larger* attention share
  than a dense model. For this config, linear layers cost 6.08 GFLOP/token and
  attention 786 kFLOP per (query, key) pair — equal at **7,735 keys**. Attention
  is 22% of forward FLOPs at T=2 and 59% at T=64. "FFN is 99%" is a dense,
  short-context fact.
* **The whole attention saving is the shared prefix not re-attending to itself
  once per branch.** Leaf-to-leaf and leaf-to-prefix pairs are identical in both
  formats; only the spine-to-spine term changes (10.78e9 -> 0.49e9 at T=64).
* **This reduces FLOPs; it does not obviously raise MFU.** A sparse mask can
  lower utilization while lowering wall-clock. Measure tokens-of-gradient per
  second.

Caveat: the thinking-strip is a property of the **chat template**, verified here
on Qwen3 templates only. Check your model's template before quoting these
numbers for it. For RL the template ships with the model being trained, so it is
settled per run.

## The row

One row per trajectory. `N` tokens, `M` nodes.

```
{
  "schema_version": ..., "trajectory_id": ...,
  "labels": [...], "annotations": {...}, "tokenizer": ..., "model": ...,

  # per token, length N, in DFS pre-order over the node tree
  "input_ids":        [N],
  "loss_mask":        [N],
  "rollout_logprobs": [N],
  "position_ids":     [N],   # offset along this token's own root path
  "token_node":       [N],   # index into the node arrays
  "token_owner_path": [N],   # leaf that owns this token for reward; -1 if not trainable

  # per node, length M
  "node_ids":    [M],        # capture's node ids, for provenance
  "node_parent": [M],        # index of parent, -1 for the root
  "node_enter":  [M],        # DFS interval, see below
  "node_exit":   [M],

  # per leaf, i.e. per branch in `token_samples` terms
  "paths": [ {"path_id": ..., "leaf_node": int, "abandoned": bool,
              "stop_reason": ..., "masked_reason": ... | null,
              "trainable_count": int}, ... ]
}
```

`paths` is what makes it lossless: a consumer wanting the old rows walks each
leaf's ancestor chain and concatenates.

## The attention mask

A token attends to its own node and any ancestor node. Nothing else — siblings
and cousins are different conversations.

The ancestor test is interval containment from a DFS pre-order numbering: stamp
`enter` on descent, `exit` on the way out. Then

> `a` is an ancestor-or-self of `d`  <=>  `enter[a] <= enter[d] < exit[a]`

Gather to token level once at load, and the flex-attention predicate is three
comparisons:

```python
def tree_mask(b, h, q_idx, kv_idx):
    is_ancestor = (tok_enter[kv_idx] <= tok_enter[q_idx]) & \
                  (tok_enter[q_idx]  <  tok_exit[kv_idx])
    return is_ancestor & (kv_idx <= q_idx)
```

`kv_idx <= q_idx` supplies causality *within* a node, where the interval test is
trivially true. Across nodes it is already implied by DFS order.

No N×N matrix is ever materialized; the mask is two int arrays of length N.
Verified on a 3-way fan-out with two levels of nesting: for every leaf the
predicate restricted to that leaf's tokens is exactly the causal mask of the
concatenated path, with no token attending off its own path.

**The mask is the feature, not an optimization on top of packing.** Dense causal
attention over the 165,911 packed tokens at T=64 would be 13.76e9 pairs against
13.07e9 for the unpacked rows — packing *without* the mask is a net loss. A
partially-correct mask does not degrade gracefully.

**Block sparsity.** Flex attention pays off only if the mask is block-structured
at its block size (128). DFS pre-order gives that for free — a node's tokens are
contiguous, and so is its ancestor set. Shared prefixes under ~128 tokens will
not show a measurable win.

## Position IDs

A token's position is its offset along **its own root path**, not its index in
the packed buffer; RoPE is wrong otherwise. Two branches will carry overlapping
position IDs, which is correct.

Emit it rather than have the consumer derive it. It is redundant — `token_node`
and `node_parent` determine it — but it is N ints against a buffer already N
tokens long, the graph is what knows the answer, and a consumer that recomputes
it wrong gets silently degraded training rather than an error.

## Train-once, and rewards

The train-once rule exists in `token_samples` because a shared sampled node
appears in several rows and training it once per branch double-counts it. **Here
the rule is structural**: each node's tokens appear exactly once, so `loss_mask`
is just "did the model produce this token".

Rewards still need an owner, because one trajectory can have several leaves with
different rewards and a shared sampled node belongs to all of them. Keep the
existing choice — first leaf to claim it, in the order `token_samples` uses —
and record it in `token_owner_path`. The consumer maps `paths[owner].reward` to
the token. This keeps the two formats numerically identical, which is the point.

Loss normalization is the consumer's business. The format supplies per-token
attribution and nothing more.

## Overlong paths

`token_samples` zeroes the whole row on a context-length overflow. Here a row is
the whole trajectory, so that would mask every sibling. Per path instead:

* A path whose leaf stopped on `context_length` is not trainable; record
  `stop_reason` and `masked_reason` on the path entry so it stays visible.
* Siblings are unaffected.
* When a sampled node is reachable from several leaves, **prefer a non-overlong
  leaf as its owner**. The overflow is a property of the leaf; the shared
  ancestor's tokens were produced before it. Only a node reachable *solely* from
  overlong paths is masked.
* Rewards are untouched. The export reports `stop_reason` per path; the harness's
  scorer decides whether to drop it.

**This is an intentional divergence from `token_samples`**, not a bug to fix by
making the acceptance test pass.

## Dependency

This needed **per-node token spans**, which the export did not emit. It does
now: `token_sample_records` emits `node_spans` alongside `node_ids`, built in
the same loop that flattens the path. That dependency is met, so what is left
here is the packing itself.

## Implementation

In `src/skyrl_capture/export/formats.py`, alongside `token_sample_records`:

1. Walk `view.nodes` from the root, DFS, assigning `enter`/`exit` and appending
   each node's `token_ids`, `sampled_mask`, `logprobs` to the flat arrays.
2. Track a running path-token count to fill `position_ids`.
3. Claim sampled nodes for the first leaf that reaches them, as
   `token_sample_records` does — except a non-overlong leaf wins over an overlong
   one regardless of order. Two passes, not one clever ordering.
4. Emit `paths` from `view.leaves()`, reusing `_abandoned` and
   `_sample_conditions`.
5. Register `tree-samples` in the export runner next to `token-samples`.

No trainer code lives in this repo. The SkyRL side — a `compose`-equivalent
building the flex-attention `BlockMask` — is a separate change there, and should
not start until the acceptance test passes.

## Acceptance test

```
for each leaf in paths:
    walk leaf -> root, concatenate node tokens in root->leaf order
    assert (input_ids, loss_mask, rollout_logprobs) == the token_samples row
        with the same path_id
```

Byte equality, on a trajectory that branches. Holds **when no path is
overlong**; with one the formats diverge on purpose, so that case needs its own
test — one overlong path and one not, asserting the good path keeps its mask and
shared ancestors stay trainable.

Add a three-way fan-out case. The interval test is what distinguishes a cousin
from an ancestor, and a 2-branch tree cannot catch a bug there.

## Deferred

**Packing across trajectories.** A GRPO group shares its entire prompt — 1035
and 1348 tokens per pair of rollouts in the measured run. Treating a group as one
forest with a shared root would save that too, but it needs a batch-level export
rather than a per-trajectory one. Revisit once this ships. The only requirement
on v1 is not to foreclose it, and it does not: a forest is a tree with a
synthetic root, and the predicate is unchanged.

## Open

**Routed experts.** `rollout_expert_indices` is per-token and packs the same way,
but is untested in any run so far.
