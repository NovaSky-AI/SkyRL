"""
visualize.py
============

Read a scored parquet from score_dataset.py and produce a set of plots
saved under ./plots/.

Usage:
    python visualize.py scored.parquet
    python visualize.py scored.parquet --out my_plots/
    python visualize.py scored.parquet --provider claude      # default
    python visualize.py scored.parquet --provider openrouter
    python visualize.py scored.parquet --provider all         # overlay all judges

Output files (all PNG, 150 dpi):
    plots/axis_distributions.png   -- per-axis score histograms (1-5)
    plots/weighted_total_hist.png  -- weighted_total distribution + outcome split
    plots/outcome_vs_score.png     -- box/strip: taste score by verifier outcome
    plots/axis_heatmap.png         -- mean per-axis score, grouped by env_key
    plots/correlation_matrix.png   -- Pearson r between axes + verifier
    plots/score_vs_turns.png       -- weighted_total vs num_turns scatter
    plots/inter_rater.png          -- per-axis kappa bars (only when >1 judge present)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

AXES = ("intent_clarity", "efficiency", "recovery", "ui_grounding", "coherence")
WEIGHTS = {
    "intent_clarity": 0.20,
    "efficiency": 0.20,
    "recovery": 0.20,
    "ui_grounding": 0.25,
    "coherence": 0.15,
}
AXIS_LABELS = {
    "intent_clarity": "Intent Clarity",
    "efficiency": "Efficiency",
    "recovery": "Recovery",
    "ui_grounding": "UI Grounding",
    "coherence": "Coherence",
}

PROVIDER_PREFIXES = {"anthropic": "claude", "openai": "gpt", "openrouter": "openrouter"}
PROVIDER_COLORS = {"claude": "#d4702a", "gpt": "#10a37f", "openrouter": "#7c3aed"}

sns.set_theme(style="whitegrid", font_scale=1.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_prefixes(df: pd.DataFrame) -> list[str]:
    return [p for p in PROVIDER_PREFIXES.values() if f"{p}_total" in df.columns]


def _axis_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{a}" for a in AXES]


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def cohen_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if pd.notna(x) and pd.notna(y)]
    if len(pairs) < 2:
        return float("nan")
    labels = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    if len(labels) < 2:
        return float("nan")
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pa = {l: sum(1 for x, _ in pairs if x == l) / n for l in labels}
    pb = {l: sum(1 for _, y in pairs if y == l) / n for l in labels}
    pe = sum(pa[l] * pb[l] for l in labels)
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# Plot 1: per-axis score distributions
# ---------------------------------------------------------------------------


def plot_axis_distributions(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    n_axes = len(AXES)
    fig, axes = plt.subplots(1, n_axes, figsize=(3.5 * n_axes, 4), sharey=False)
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    for ax, axis in zip(axes, AXES):
        for prefix in prefixes:
            col = f"{prefix}_{axis}"
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            color = PROVIDER_COLORS.get(prefix, "steelblue")
            ax.hist(
                vals,
                bins=bins,
                alpha=0.65,
                color=color,
                label=prefix,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(AXIS_LABELS[axis], fontsize=10, fontweight="bold")
        ax.set_xlabel("Score (1-5)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.set_xlim(0.5, 5.5)

    axes[0].set_ylabel("Count")
    if len(prefixes) > 1:
        axes[-1].legend(title="Judge", fontsize=8)

    fig.suptitle("Per-axis Score Distributions", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out / "axis_distributions.png")


# ---------------------------------------------------------------------------
# Plot 2: weighted_total histogram split by verifier outcome
# ---------------------------------------------------------------------------


def plot_weighted_total_hist(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(prefixes), figsize=(5 * len(prefixes), 4), squeeze=False)

    for ax, prefix in zip(axes[0], prefixes):
        col = f"{prefix}_total"
        if col not in df.columns:
            ax.set_visible(False)
            continue
        bins = np.linspace(1, 5, 17)
        for outcome, color, label in [
            (True, "#2a9d8f", "verifier pass"),
            (False, "#e76f51", "verifier fail"),
        ]:
            sub = df[df["verifier_score"] == int(outcome)][col].dropna()
            if len(sub):
                ax.hist(sub, bins=bins, alpha=0.7, color=color, label=f"{label} (n={len(sub)})", edgecolor="white")
        ax.axvline(df[col].dropna().mean(), color="black", linestyle="--", linewidth=1.2, label=f"mean={df[col].dropna().mean():.2f}")
        ax.set_title(f"{prefix} — weighted_total", fontweight="bold")
        ax.set_xlabel("weighted_total (1-5)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Weighted Total Distribution by Verifier Outcome", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out / "weighted_total_hist.png")


# ---------------------------------------------------------------------------
# Plot 3: taste score by verifier outcome (box + strip)
# ---------------------------------------------------------------------------


def plot_outcome_vs_score(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    rows = []
    for prefix in prefixes:
        col = f"{prefix}_total"
        if col not in df.columns:
            continue
        sub = df[["verifier_score", col]].dropna().copy()
        sub["judge"] = prefix
        sub = sub.rename(columns={col: "score"})
        rows.append(sub)
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    long["outcome"] = long["verifier_score"].map({1: "pass", 0: "fail"})

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"pass": "#2a9d8f", "fail": "#e76f51"}
    sns.boxplot(
        data=long, x="judge", y="score", hue="outcome",
        palette=palette, width=0.5, linewidth=1.0, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=long, x="judge", y="score", hue="outcome",
        palette=palette, dodge=True, size=3, alpha=0.5, jitter=True, ax=ax,
        legend=False,
    )
    ax.set_ylabel("weighted_total (1-5)")
    ax.set_xlabel("Judge")
    ax.set_title("Taste Score by Verifier Outcome", fontweight="bold")
    ax.legend(title="Outcome", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "outcome_vs_score.png")


# ---------------------------------------------------------------------------
# Plot 4: mean per-axis score heatmap by env_key
# ---------------------------------------------------------------------------


def plot_axis_heatmap(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    if "env_key" not in df.columns:
        return
    prefix = prefixes[0]
    cols = {a: f"{prefix}_{a}" for a in AXES}
    present = {a: c for a, c in cols.items() if c in df.columns}
    if not present:
        return

    grp = df.groupby("env_key")[[c for c in present.values()]].mean()
    grp.columns = [AXIS_LABELS[a] for a in present]
    grp = grp.sort_values(list(grp.columns)[0])

    fig, ax = plt.subplots(figsize=(len(present) * 1.4 + 1, max(3, len(grp) * 0.5 + 1)))
    sns.heatmap(
        grp, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.7},
    )
    ax.set_title(f"Mean Axis Scores by env_key ({prefix})", fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    _savefig(fig, out / "axis_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 5: correlation matrix (axes + verifier)
# ---------------------------------------------------------------------------


def plot_correlation_matrix(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    prefix = prefixes[0]
    axis_cols = [f"{prefix}_{a}" for a in AXES if f"{prefix}_{a}" in df.columns]
    total_col = f"{prefix}_total"
    cols = axis_cols + ([total_col] if total_col in df.columns else []) + ["verifier_score"]
    sub = df[[c for c in cols if c in df.columns]].dropna()
    if sub.shape[1] < 2:
        return

    rename = {f"{prefix}_{a}": AXIS_LABELS[a] for a in AXES}
    rename[f"{prefix}_total"] = "weighted_total"
    rename["verifier_score"] = "verifier"
    sub = sub.rename(columns=rename)

    corr = sub.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Pearson Correlation — {prefix}", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out / "correlation_matrix.png")


# ---------------------------------------------------------------------------
# Plot 6: weighted_total vs num_turns scatter
# ---------------------------------------------------------------------------


def plot_score_vs_turns(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    if "num_turns" not in df.columns:
        return
    prefix = prefixes[0]
    col = f"{prefix}_total"
    if col not in df.columns:
        return
    sub = df[["num_turns", col, "verifier_score"]].dropna().copy()
    sub["outcome"] = sub["verifier_score"].map({1: "pass", 0: "fail"})

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"pass": "#2a9d8f", "fail": "#e76f51"}
    sns.scatterplot(data=sub, x="num_turns", y=col, hue="outcome", palette=palette, alpha=0.7, s=50, ax=ax)
    # regression line
    m, b = np.polyfit(sub["num_turns"], sub[col], 1)
    xs = np.linspace(sub["num_turns"].min(), sub["num_turns"].max(), 100)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.2, linestyle="--", label=f"slope={m:.3f}")
    ax.set_xlabel("num_turns")
    ax.set_ylabel("weighted_total (1-5)")
    ax.set_title(f"Taste Score vs. Trajectory Length ({prefix})", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "score_vs_turns.png")


# ---------------------------------------------------------------------------
# Plot 7: inter-rater kappa bars
# ---------------------------------------------------------------------------


def plot_inter_rater(df: pd.DataFrame, prefixes: list[str], out: Path) -> None:
    if len(prefixes) < 2:
        return
    pairs = [(prefixes[i], prefixes[j]) for i in range(len(prefixes)) for j in range(i + 1, len(prefixes))]

    fig, axes_arr = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
    for ax, (p, q) in zip(axes_arr[0], pairs):
        kappas = []
        labels = []
        for axis in AXES:
            ca, cb = f"{p}_{axis}", f"{q}_{axis}"
            if ca not in df.columns or cb not in df.columns:
                continue
            k = cohen_kappa(df[ca].tolist(), df[cb].tolist())
            kappas.append(k)
            labels.append(AXIS_LABELS[axis])
        if not kappas:
            ax.set_visible(False)
            continue
        colors = ["#2a9d8f" if k >= 0.4 else "#e9c46a" if k >= 0.2 else "#e76f51" for k in kappas]
        bars = ax.barh(labels, kappas, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.2, 1.0)
        ax.set_title(f"Cohen's κ: {p} vs {q}", fontweight="bold")
        ax.set_xlabel("κ")
        for bar, k in zip(bars, kappas):
            ax.text(max(k + 0.02, 0.02), bar.get_y() + bar.get_height() / 2,
                    f"{k:.2f}", va="center", fontsize=8)
    fig.suptitle("Inter-rater Agreement per Axis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out / "inter_rater.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="scored parquet from score_dataset.py")
    ap.add_argument("--out", default="plots", help="output directory (default: plots/)")
    ap.add_argument(
        "--provider",
        default="auto",
        help="judge prefix to use for single-judge plots (claude/gpt/openrouter/all/auto)",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    prefixes = _detect_prefixes(df)
    if not prefixes:
        print("No judge score columns found (expected e.g. claude_total). Exiting.")
        return 1
    print(f"Detected judge prefixes: {prefixes}")

    if args.provider == "all":
        active = prefixes
    elif args.provider == "auto":
        active = prefixes
    elif args.provider in prefixes:
        active = [args.provider]
    else:
        print(f"Provider {args.provider!r} not found in parquet; using {prefixes[0]}")
        active = [prefixes[0]]

    out = Path(args.out)
    print(f"Writing plots to {out}/")

    plot_axis_distributions(df, active, out)
    plot_weighted_total_hist(df, active, out)
    plot_outcome_vs_score(df, active, out)
    plot_axis_heatmap(df, active, out)
    plot_correlation_matrix(df, active, out)
    plot_score_vs_turns(df, active, out)
    plot_inter_rater(df, active, out)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
