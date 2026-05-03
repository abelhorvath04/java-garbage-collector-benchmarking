#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = Path("stw_pause_summary_after_warmup.csv")
OUTPUT_DIR = Path("stw_overhead_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_CSV, sep=";")
df["stw_overhead_percent"] = pd.to_numeric(df["stw_overhead_percent"], errors="coerce")
df = df.dropna(subset=["stw_overhead_percent"])

df["collector"] = df["environment"].apply(lambda x: "G1GC" if "G1GC" in x else "ZGC")
df["java"] = df["environment"].apply(lambda x: "Java 21" if "21" in x else "Java 25")

def plot_collector(collector: str):
    sub = df[df["collector"] == collector].copy()

    pivot = sub.pivot_table(
        index="benchmark",
        columns="java",
        values="stw_overhead_percent",
        aggfunc="mean",
    )

    pivot = pivot.sort_index()

    benchmarks = pivot.index.tolist()
    y = np.arange(len(benchmarks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 12))

    ax.barh(y + width / 2, pivot.get("Java 25", 0), width, label="Java 25", color="#ffb703")
    ax.barh(y - width / 2, pivot.get("Java 21", 0), width, label="Java 21", color="#8ecae6")
    
    ax.set_yticks(y)
    ax.set_yticklabels(benchmarks, fontsize=8)

    ax.set_xlabel("STW overhead [%]")
    ax.set_title(f"{collector} STW overhead")
    ax.legend(frameon=False)

    ax.grid(axis="x", linestyle=":", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out = OUTPUT_DIR / f"{collector.lower()}-stw-overhead-grouped-bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

plot_collector("G1GC")
plot_collector("ZGC")