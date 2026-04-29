#!/usr/bin/env python3
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt


ENV_DIRS = [
    ("G1GC-Java-25", "../experiment/G1GC-java25-2026-04-24"),
    ("G1GC-Java-21", "../experiment/G1GC-java21-2026-04-25"),
    ("ZGC-Java-25", "../experiment/ZGC-java25-2026-04-24"),
    ("ZGC-Java-21", "../experiment/ZGC-java21-2026-04-25")
]

BENCHMARKS = [
    "akka-uct", "als", "chi-square", "dec-tree", "gauss-mix",
    "log-regression", "movie-lens", "naive-bayes", "page-rank",
    "fj-kmeans", "reactors", "db-shootout", "neo4j-analytics",
    "future-genetic", "mnemonics", "par-mnemonics", "rx-scrabble",
    "scrabble", "dotty", "philosophers", "scala-doku", "scala-kmeans",
    "scala-stm-bench7", "finagle-chirper", "finagle-http",
]

FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "benchmarking-g1gc-zgc.png"


def load_benchmark_data(env_path: str, benchmark: str) -> list[float]:
    csv_path = Path(env_path) / benchmark / f"{benchmark}.csv"

    if not csv_path.is_file():
        print(f"WARNING: missing file: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"WARNING: failed to read {csv_path}: {exc}")
        return []

    required_columns = {"duration_ns", "uptime_ns"}
    if not required_columns.issubset(df.columns):
        print(f"WARNING: missing required columns in {csv_path}")
        return []

    df["duration_ns"] = pd.to_numeric(df["duration_ns"], errors="coerce")
    df["uptime_ns"] = pd.to_numeric(df["uptime_ns"], errors="coerce")
    df = df.dropna(subset=["duration_ns", "uptime_ns"])

    df = df[df["uptime_ns"] >= FIVE_MINUTES_NS]

    if df.empty:
        print(f"WARNING: no rows left after 5-minute filtering in {csv_path}")
        return []

    duration_ms = df["duration_ns"] / 1_000_000.0
    return duration_ms.tolist()


def main() -> None:
    n_benchmarks = len(BENCHMARKS)
    ncols = 5
    nrows = math.ceil(n_benchmarks / ncols)

    loaded_combinations = 0
    missing_or_empty = 0
    total_measurements = 0

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 17))
    axes = axes.flatten()

    colors = ["lightblue", "lightgreen", "mistyrose", "plum"]

    for i, benchmark in enumerate(BENCHMARKS):
        ax = axes[i]

        data = []
        labels = []

        for env_name, env_path in ENV_DIRS:
            values = load_benchmark_data(env_path, benchmark)

            if values:
                data.append(values)
                labels.append(env_name)
                loaded_combinations += 1
                total_measurements += len(values)
            else:
                missing_or_empty += 1

        if not data:
            ax.set_title(benchmark, fontsize=10)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=True,
        )

        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)

        ax.set_title(benchmark, fontsize=10)
        ax.set_ylabel("Iteration duration [ms]", fontsize=8)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)

    for j in range(n_benchmarks, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Durchsatz",
        fontsize=15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Validation summary")
    print(f"  benchmarks: {len(BENCHMARKS)}")
    print(f"  environments: {len(ENV_DIRS)}")
    print(f"  expected combinations: {len(BENCHMARKS) * len(ENV_DIRS)}")
    print(f"  loaded combinations: {loaded_combinations}")
    print(f"  missing/empty combinations: {missing_or_empty}")
    print(f"  total post-warmup measurements: {total_measurements}")
    print(f"  saved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()