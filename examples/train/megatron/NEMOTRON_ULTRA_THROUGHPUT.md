# Nemotron-3-Ultra-550B — Megatron training throughput & memory sweep

Goal: find, on **8× nodes of 8×H200-141GB (64 GPUs, EFA)**, (1) the maximum
`trainer.max_tokens_per_microbatch` (MTPM) that fits for training fwd+bwd, and
(2) the Megatron parallelism (TP/PP/CP/EP/DP) that maximizes training throughput
for full-finetuning GRPO of Nemotron-3-Ultra-550B.

## Method

Runs use a dedicated harness that executes the **real** Megatron fwd+bwd training
path on fabricated rollouts (no vLLM generation), so numbers reflect genuine
training cost while iterating fast:

- Trainer: `examples/train_scripts/full_context/trainer_ultra_sweep.py`
  (extends the dummy `FullCtxTrainer`; logs per-step wall time + peak CUDA
  reserved/allocated memory across policy workers to a JSONL file).
- Launcher: `examples/train/megatron/run_ultra_sweep.sh` (all knobs are env vars).
- Analysis: `examples/train_scripts/full_context/analyze_sweep.py`.

Colocated config matches the validated recipe (`run_megatron_nemotron_ultra.sh`):
optimizer CPU-offloaded, `recompute_granularity=full`, `remove_microbatch_padding=true`,
vLLM colocated but **asleep during training** (engines sleep at init in colocate
mode, and the harness never wakes them — exactly the memory state of the real
training step). Peak reserved ≈ caching-allocator high-water mark during the step.

`max_tokens_per_microbatch` bin-packs each microbatch up to that many tokens **per
DP rank**; a single sequence longer than the budget gets its own microbatch. So the
MTPM memory ceiling is per-DP-rank and independent of DP size.

## TL;DR

- **Max `max_tokens_per_microbatch`** at the validated config (TP8/PP4/EP16/DP2): **~64k** tokens/microbatch
  (per DP rank). 64k fits; 72k+ OOMs. The model + bf16 grads take ~38 GiB/GPU (optimizer CPU-offloaded)
  and a sleeping vLLM holds ~5–8 GiB, leaving ~95 GiB for activations even with `recompute_granularity=full`.
- **Highest training throughput**: **TP8 / PP2 / EP32 / DP4** at MTPM≈32k → **~8540 tok/s**, vs **7720 tok/s**
  for the PP4/DP2 baseline at MTPM=48k — a ~11% gain from doubling data parallelism (DP2→DP4). PP2 is what
  frees the GPUs for DP4; it costs MTPM headroom (caps ~32–40k) but nets faster.
- **The config space is tightly pinned** by the model (108 layers, MoE with 512 experts) and 141 GiB H200s:
  PP must divide 108 (**PP=8 is invalid**), EP8 doubles expert memory and **OOMs at load**, and TP4 doubles
  activation memory (sequence-parallel shards activations by TP) and **OOMs**. Viable full-FT configs are
  essentially {TP8/PP4/EP16/DP2, TP8/PP2/EP32/DP4}.
- **Long context is activation-bound by the single longest sequence**: with `remove_microbatch_padding`, any
  sequence longer than MTPM gets its own microbatch that must fit alone. The *single-sequence* ceiling is
  **~40–48k tokens** at CP1/PP4/DP2 (well below the ~64k *packed* ceiling — a long contiguous sequence has a
  much larger per-microbatch footprint than an equal-token pack of short ones).
- **Context parallelism roughly doubles that ceiling to ~96k.** CP *composes* with EP (in Megatron-Core
  `EP` divides `TP·CP·DP`, so CP does not steal from EP's budget): **`TP8/PP4/CP2/EP16/DP1` fits a single 96k
  sequence** (128k OOMs) while keeping baseline expert memory. CP=4 via `TP8/PP2/CP4/EP32/DP1` is *valid* but
  *worse* — it still OOMs at 128k because dropping to PP2 (needed to free GPUs for CP4) doubles the weights and
  eats the budget CP frees. So the practical long-context recipe is **PP4 + CP2** (≤~96k/seq, at the cost of
  DP→1), and the 60k±30k distribution becomes mostly trainable (clamp ~96k truncates only the ~10% tail vs
  ~half at CP1's ~40k clamp).
- **Long sequences are *more* throughput-efficient per token** (~12k tok/s on a ~39k-mean distribution at
  PP4/DP2, vs ~7.7k for uniform 10 240-token seqs): bigger microbatches use the GEMMs better and incur less
  per-microbatch/pipeline overhead.
- **The throughput-optimal config is sequence-length-dependent**: PP2/EP32/DP4 for short/medium sequences
  (≤~48k); PP4/EP16/DP2 for long sequences (higher single-sequence ceiling).

## Cluster / model facts (measured)
- 64× H200-141 GiB (8 nodes), EFA. GPU usable ≈ 139.8 GiB; sleeping colocated vLLM holds ~5–8 GiB during training.
- Model: 108 layers (hybrid Mamba2 + attention + latent MoE, 512 experts). At TP8/PP4/EP16: **9.44B params/GPU**
  → ~18.9 GiB bf16 weights + ~18.9 GiB bf16 grads (~38 GiB fixed; AdamW master/moments CPU-offloaded).
- MoE expert memory/GPU ∝ (108/PP)·(512/EP); **PP·EP is the invariant**. Baseline PP4·EP16 = 64.
  PP2 needs EP32 to match (PP2·EP32=64); PP4·EP8=32 ⇒ 2× expert memory ⇒ OOM.

## Stage 1 — max tokens per microbatch (TP8/PP4/EP16/DP2, uniform 10240-token seqs)

| MTPM (setting) | largest microbatch | result |
|---:|---:|:--|
| 65536 (64k) | 61 440 tok | **FITS** — steady 77 s/step @ 327 680 tok |
| 73728 (72k) | 71 680 tok | FAIL (DistBackend; one rank OOM aborts NCCL — boundary) |
| 81920 (80k) | 81 920 tok | OOM (forward_backward) |
| 98304 (96k) | 92 160 tok | OOM |
| 131072 (128k) | 131 072 tok | OOM (needed +14.1 GiB; only 8.2 GiB free) |

**Max safe MTPM ≈ 64k** for packed short/medium sequences.

## Stage 2 — parallelism sweep for throughput (fixed 655 360-token workload, uniform 10240-token seqs)

| Config (TP/PP/CP/EP, DP) | MTPM | step time | **throughput** | in-step peak | result |
|:--|---:|---:|---:|---:|:--|
| **TP8/PP2/EP32, DP4** | 32k | 76.8 s | **8 539 tok/s** | 109.9 GB | **OK — fastest** |
| TP8/PP4/EP16, DP2 (baseline) | 48k | 84.9 s | 7 719 tok/s | 108.3 GB | OK |
| TP8/PP2/EP32, DP4 | 48k | — | — | — | OOM (PP2 caps MTPM ~32–40k) |
| TP4/PP4/EP16, DP4 | 32k | — | — | — | OOM (TP4 ⇒ 2× activation via SP) |
| TP8/PP4/EP8, DP2 | 48k | — | — | — | OOM at model load (2× expert mem) |
| TP8/PP8/EP8, DP1 | — | — | — | — | INVALID (108 not divisible by 8) |
| TP4/PP8/EP8, DP2 | — | — | — | — | INVALID (108 not divisible by 8) |

Doubling data parallelism (DP2→DP4) is the throughput lever; PP2 is the only way to free GPUs for DP4
given the layer-count and EP constraints. TP must stay 8 (sequence parallelism shards activations by TP).

## Stage 3 — long context (variable length) & context parallelism

**CP composes with EP** (not with DP-budget-for-EP as first assumed): in Megatron-Core the expert group is
formed over `TP·CP·DP`, so with `ETP=1`, **`EP` must divide `TP·CP·DP`** — adding CP does *not* force EP down.
This makes CP genuinely useful here. Single-sequence ceiling (one sequence alone in its microbatch):

| Config | world | single-seq ceiling | note |
|:--|:--|---:|:--|
| CP1 / PP4 / EP16 / DP2 | 8·4·1·2 | **~40–48k tok** | single 40 960 fits (peak 110 GB); 49 152 OOMs. Long contiguous seq footprint ≫ equal-token pack of short seqs, so single-seq ceiling < packed 64k. |
| **CP2 / PP4 / EP16 / DP1** | 8·4·2·1 | **~96k tok** | EP16 still valid (`EP \| TP·CP·DP = 16`); **single 98 304 FITS**, 131 072 OOMs. Keeps PP4's low weights (~38 GiB) and shards the seq 2×. Best long-context config. |
| CP4 / PP2 / EP32 / DP1 | 8·2·4·1 | <128k | *valid* (`EP32 \| 8·4·1=32`) and loads, but **131 072 OOMs**: dropping to PP2 to free GPUs for CP4 doubles weights (~76 GiB) and eats the budget CP frees — worse than PP4/CP2. |

So **CP roughly doubles the usable single-sequence length** (~40–48k → ~96k) via **PP4 + CP2**, at the cost of
collapsing DP to 1 (≈ half the data-parallel throughput). The 60k±30k distribution is then **mostly trainable**:
clamping at ~96k truncates only the ~10% upper tail (vs truncating ~half at CP1's ~40k clamp). The extreme
131k tail still OOMs (the LM-head logits / non-CP-sharded buffers don't shrink enough); CP4 doesn't fix it
because of the PP2 weight penalty.

### Throughput on a variable-length long-context distribution

The requested distribution was 256 samples ~ N(60k, 30k) tokens. Because the single-sequence
ceiling is ~40–48k (above), the distribution must be **truncated** to a value that fits as a single
microbatch; the 60k mean is not trainable per-sequence on 64 GPUs. Measured on PP4/EP16/DP2 with
sequences ~ N(60k, 30k) **clamped to [1k, 40 960]** (realized mean ~39k, max 40 960), MTPM=40 960:

| Config | distribution (realized) | MTPM | step time | **throughput** | peak |
|:--|:--|---:|---:|---:|---:|
| TP8/PP4/EP16/DP2 | 64 varlen seqs, mean ~39k, max 40 960 | 40 960 | 207 s | **~12 070 tok/s** | 112.9 GB |

(Throughput is per-token and count-independent; 64 samples used for tractable wall-clock. Earlier
256-sample runs confirmed the packing/fit behaviour and the OOM on sequences >~48k.)

**Long sequences are *more* throughput-efficient per token** than short ones (~12.1k tok/s here vs
~7.7k for uniform 10 240-token seqs at the same PP4/DP2): a microbatch of one ~40k sequence (or a few
medium ones) uses the GEMMs far better and incurs less per-microbatch / pipeline overhead than packing
many tiny 10 240-token sequences. So for long-context RL the throughput ceiling is set by **fitting the
longest single sequence** (clamp responses to ~40k), not by aggregate tokens.

## Recommendations

- **Short/medium sequences (≤~40k), max throughput:** `TP8 / PP2 / EP32 / ETP1 / DP4`, MTPM≈32k
  (~8.5k tok/s on uniform short seqs; ~11% over the PP4/DP2 baseline).
- **Long context, no CP:** `TP8 / PP4 / EP16 / ETP1 / DP2`, MTPM≈40–64k (single-sequence ceiling ~40–48k;
  ~12k tok/s on a ~39k-mean distribution — long seqs are more throughput-efficient per token).
- **Longest single sequences (up to ~96k):** `TP8 / PP4 / CP2 / EP16 / ETP1 / DP1` — CP2 ~doubles the
  single-seq ceiling to ~96k (EP16 stays valid since `EP \| TP·CP·DP`). Costs DP→1 (≈ half the DP throughput),
  so use it only when sequences actually exceed ~48k. Prefer **PP4+CP2** over PP2+CP4 (PP2's weights negate CP).
- **For the 60k±30k distribution:** clamp responses to ~96k with PP4/CP2 (≈10% of samples truncated), or to
  ~40k with PP4/DP2 (≈half truncated, but full DP throughput). The full untruncated 131k tail is not trainable
  on 64×H200.
- Keep `TP=8` (sequence parallelism shards activations by TP — TP4 doubles activation memory and OOMs),
  optimizer CPU-offload on, and `recompute_granularity=full`.


## Reproducing

```bash
# Stage 1 — MTPM ceiling at the baseline config (uniform 10k-token seqs).
TP=8 PP=4 CP=1 EP=16 ETP=1 MTPM=65536 MODE=uniform SEQ_LEN=10240 \
  TAG=s1_tp8pp4ep16_mtpm65536 SWEEP_RESULTS_FILE=/home/ray/ultra_sweep/results.jsonl \
  bash examples/train/megatron/run_ultra_sweep.sh

# Stage 2 — highest-throughput config (DP4).
TP=8 PP=2 CP=1 EP=32 ETP=1 MTPM=32768 MODE=uniform SEQ_LEN=10240 NUM_SEQ=64 \
  TAG=s2_tp8pp2ep32_dp4 SWEEP_RESULTS_FILE=/home/ray/ultra_sweep/results.jsonl \
  bash examples/train/megatron/run_ultra_sweep.sh

# Stage 3 — long-context varlen distribution (clamped to the ~40k single-seq ceiling).
TP=8 PP=4 CP=1 EP=16 ETP=1 MODE=varlen AVG_LEN=60000 STD_LEN=30000 MAX_LEN=40960 \
  MTPM=40960 NUM_SEQ=64 TAG=s3_varlen_clamp40k \
  SWEEP_RESULTS_FILE=/home/ray/ultra_sweep/results.jsonl \
  bash examples/train/megatron/run_ultra_sweep.sh

python examples/train_scripts/full_context/analyze_sweep.py /home/ray/ultra_sweep/results.jsonl
```
