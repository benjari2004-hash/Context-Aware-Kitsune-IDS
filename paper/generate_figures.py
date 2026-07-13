#!/usr/bin/env python3
"""
generate_figures.py
===================
Generates all figures required by main.tex into paper/figures/.
Run ONCE before uploading to Overleaf.

Usage (from project root):
    python paper/generate_figures.py

Output: paper/figures/
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Paths
PROJECT_ROOT   = Path(__file__).parent.parent
PAPER_DIR      = Path(__file__).parent
FIG_DIR        = PAPER_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

RESULTS_CSV    = PROJECT_ROOT / "results.csv"
CF_LATENCY_CSV = PROJECT_ROOT / "experiments" / "cf_latency_results.csv"
EXP_PLOTS      = PROJECT_ROOT / "experiments" / "plots"
MY_IDS_PLOTS   = PROJECT_ROOT / "my_ids" / "experiments" / "plots"

# Style
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "savefig.format": "pdf",
})

C = {
    "pass1":    "#3488BD",
    "pass2":    "#D65044",
    "residual": "#888888",
}


# Fig 03: CF Pass Coverage
def fig03_cf_coverage():
    df  = pd.read_csv(CF_LATENCY_CSV)
    row = df.iloc[0]
    n   = int(row["n_tested"])

    pass1  = int(round(row["pass1_pct"] / 100 * n))   # 1014
    pass2  = int(round(row["pass2_pct"] / 100 * n))   # 8440
    unexpl = n - pass1 - pass2                         # 231

    labels = ["Pass 1\n(single-feature)", "Pass 2\n(override decomp.)", "Unexplained\n(residual)"]
    values = [pass1, pass2, unexpl]
    colors = [C["pass1"], C["pass2"], C["residual"]]
    pcts   = [100 * v / n for v in values]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.55)

    for bar, pct, val in zip(bars, pcts, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 90,
            "{:,}\n({:.2f}%)".format(val, pct),
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Number of anomaly packets")
    ax.set_title("CF pass distribution  (n={:,} anomalies)".format(n))
    ax.set_ylim(0, pass2 * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))
    ax.spines[["top", "right"]].set_visible(False)

    out = FIG_DIR / "fig03_cf_coverage.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    print("  OK " + out.name)


# Fig 04: Attack Distribution
def fig04_attack_distribution():
    df = pd.read_csv(RESULTS_CSV, comment="#", low_memory=False)
    if "packet_id" not in df.columns:
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    anomalies = df[df["label"].str.upper() == "ANOMALY"]
    counts = anomalies["attack_type"].value_counts()
    counts = counts[counts > 0].sort_values(ascending=True)

    bar_colors = [C["pass2"] if "Mirai" in str(c) else C["pass1"] for c in counts.index]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    hbars = ax.barh(counts.index, counts.values, color=bar_colors,
                    edgecolor="white", linewidth=0.6, height=0.65)

    for bar, val in zip(hbars, counts.values):
        pct = 100 * val / counts.sum()
        ax.text(
            bar.get_width() + 30,
            bar.get_y() + bar.get_height() / 2,
            "{:,}  ({:.1f}%)".format(val, pct),
            va="center", fontsize=8.5,
        )

    ax.set_xlabel("Packet count")
    ax.set_title("Attack type distribution  ({:,} anomalies)".format(int(counts.sum())))
    ax.set_xlim(0, counts.max() * 1.22)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))
    ax.spines[["top", "right"]].set_visible(False)

    patch1 = mpatches.Patch(color=C["pass2"], label="Override-locked (Pass 2)")
    patch2 = mpatches.Patch(color=C["pass1"], label="Not override-locked (Pass 1)")
    ax.legend(handles=[patch1, patch2], fontsize=8.5, loc="lower right")

    out = FIG_DIR / "fig04_attack_distribution.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    print("  OK " + out.name)


# Fig 05: CF Latency
def fig05_cf_latency():
    src = MY_IDS_PLOTS / "cf_latency_tradeoff.png"
    if src.exists():
        shutil.copy(str(src), str(FIG_DIR / "fig05_cf_latency.png"))
        print("  OK fig05_cf_latency.png  (copied from my_ids)")
        return

    df   = pd.read_csv(CF_LATENCY_CSV)
    modes = df["mode"].tolist()
    avg   = df["avg_ms"].tolist()
    p50   = df["p50_ms"].tolist()
    p95   = df["p95_ms"].tolist()
    maxv  = df["max_ms"].tolist()

    x = np.arange(len(modes))
    w = 0.18

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - 1.5*w, avg,  w, label="Mean", color="#3488BD", edgecolor="white")
    ax.bar(x - 0.5*w, p50,  w, label="p50",  color="#4CAF50", edgecolor="white")
    ax.bar(x + 0.5*w, p95,  w, label="p95",  color="#FF9800", edgecolor="white")
    ax.bar(x + 1.5*w, maxv, w, label="Max",  color="#D65044", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("CF engine latency by budget mode  (n=9,685)")
    ax.legend(framealpha=0.9, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.97, "Timeouts: 0 in all modes",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))

    out = FIG_DIR / "fig05_cf_latency.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    print("  OK " + out.name + "  (generated from CSV)")


# Fig 06: Experiment A
def fig06_exp_a():
    for src in [EXP_PLOTS / "exp_A_fp_reduction.png",
                MY_IDS_PLOTS / "exp_A_fp_reduction.png"]:
        if src.exists():
            shutil.copy(str(src), str(FIG_DIR / "fig06_exp_A_fp_reduction.png"))
            print("  OK fig06_exp_A_fp_reduction.png  (copied)")
            return

    rounds  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fp_pct  = [2.77, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    fn_pct  = [0.86, 0.59, 0.54, 0.64, 0.66, 1.05, 1.28, 1.53, 1.67, 1.71, 1.86]
    f1      = [0.9021, 0.9970, 0.9973, 0.9968, 0.9967, 0.9947,
               0.9936, 0.9923, 0.9916, 0.9914, 0.9906]

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax2 = ax1.twinx()
    ln1, = ax1.plot(rounds, fp_pct, "o-",  color="#D65044", label="FP rate (%)", linewidth=1.8)
    ln2, = ax1.plot(rounds, fn_pct, "s--", color="#FF9800", label="FN rate (%)", linewidth=1.8)
    ln3, = ax2.plot(rounds, f1,     "^:",  color="#3488BD", label="F1 score",    linewidth=1.8)
    ax1.set_xlabel("Feedback round")
    ax1.set_ylabel("Rate (%)")
    ax2.set_ylabel("F1 score")
    ax2.set_ylim(0.88, 1.01)
    ax1.set_xticks(rounds)
    ax1.set_title("Experiment A: FP reduction under feedback pressure")
    ax1.spines[["top"]].set_visible(False)
    ax1.legend([ln1, ln2, ln3], ["FP rate (%)", "FN rate (%)", "F1 score"],
               loc="center right", fontsize=8.5)
    out = FIG_DIR / "fig06_exp_A_fp_reduction.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    print("  OK " + out.name + "  (generated from data)")


# Fig 07: Experiment B
def fig07_exp_b():
    for src in [EXP_PLOTS / "exp_B_ablation.png",
                MY_IDS_PLOTS / "exp_B_ablation.png"]:
        if src.exists():
            shutil.copy(str(src), str(FIG_DIR / "fig07_exp_B_ablation.png"))
            print("  OK fig07_exp_B_ablation.png  (copied)")
            return
    print("  ! fig07_exp_B_ablation -- source PNG not found; skipping")


# Fig 08: Experiment D
def fig08_exp_d():
    for src in [EXP_PLOTS / "exp_D_drift.png",
                MY_IDS_PLOTS / "exp_D_drift.png"]:
        if src.exists():
            shutil.copy(str(src), str(FIG_DIR / "fig08_exp_D_drift.png"))
            print("  OK fig08_exp_D_drift.png  (copied)")
            return
    print("  ! fig08_exp_D_drift -- source PNG not found; skipping")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\nGenerating figures -> " + str(FIG_DIR))
    print()
    fig03_cf_coverage()
    fig04_attack_distribution()
    fig05_cf_latency()
    fig06_exp_a()
    fig07_exp_b()
    fig08_exp_d()
    print("\nDone.  Upload paper/figures/ to Overleaf.\n")
    print("Figures generated:")
    for f in sorted(FIG_DIR.iterdir()):
        print("  " + f.name)
