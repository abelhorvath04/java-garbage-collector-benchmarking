import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Input
# ============================================================

INPUT_FILE = "stw_pause_summary_after_warmup.csv"

df = pd.read_csv(INPUT_FILE, sep=";")

# Keep only the columns required for this analysis
df = df[[
    "benchmark",
    "environment",
    "stw_overhead_percent",
    "stw_pause_count"
]].copy()


# ============================================================
# Compute Java 21 vs Java 25 STW-overhead comparisons
# ============================================================

rows = []

for benchmark in sorted(df["benchmark"].unique()):
    benchmark_data = df[df["benchmark"] == benchmark]

    for gc in ["G1GC", "ZGC"]:
        env21 = f"{gc}-Java-21"
        env25 = f"{gc}-Java-25"

        row21 = benchmark_data[benchmark_data["environment"] == env21]
        row25 = benchmark_data[benchmark_data["environment"] == env25]

        if row21.empty or row25.empty:
            continue

        row21 = row21.iloc[0]
        row25 = row25.iloc[0]

        rows.append({
            "benchmark": benchmark,
            "gc": gc,
            "stw_overhead_java21_percent": row21["stw_overhead_percent"],
            "stw_overhead_java25_percent": row25["stw_overhead_percent"],
            "stw_pause_count_java21": row21["stw_pause_count"],
            "stw_pause_count_java25": row25["stw_pause_count"],
            "stw_pause_count_diff": row25["stw_pause_count"] - row21["stw_pause_count"],
            "stw_overhead_diff_percentage_points": (
                row25["stw_overhead_percent"] - row21["stw_overhead_percent"]
            )
        })

plot_df = pd.DataFrame(rows)

plot_df.to_csv(
    "java25_vs_java21_stw_overhead_comparison_for_plot.csv",
    sep=";",
    index=False
)


# ============================================================
# Helper function for plotting one GC separately
# ============================================================

def plot_stw_overhead_for_gc(gc_name, output_prefix):
    gc_df = plot_df[plot_df["gc"] == gc_name].copy()
    
    gc_df["sort_key"] = gc_df[
        ["stw_overhead_java21_percent", "stw_overhead_java25_percent"]
    ].max(axis=1)

    # Sort benchmarks by Java 25 - Java 21 STW-overhead difference.
    # This makes it easy to see where overhead decreased or increased.
    order = (
        gc_df
        .set_index("benchmark")["stw_overhead_diff_percentage_points"]
        .sort_values()
        .index
        .tolist()
    )

    gc_df = gc_df.set_index("benchmark").reindex(order)

    y = np.arange(len(order))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(11, 10))

    ax.barh(
        y - bar_height / 2,
        gc_df["stw_overhead_java21_percent"],
        height=bar_height,
        label="Java 21",
        color="#8ecae6"
    )

    ax.barh(
        y + bar_height / 2,
        gc_df["stw_overhead_java25_percent"],
        height=bar_height,
        label="Java 25",
        color="#ffb703"
    )

    # ============================================================
    # Add STW event count labels to the end of each bar
    # ============================================================

    # Java 21 labels
    for i, value in enumerate(gc_df["stw_overhead_java21_percent"]):
        count = int(gc_df["stw_pause_count_java21"].iloc[i])
        ax.text(
            value,
            y[i] - bar_height / 2,
            f"  n={count}",
            va="center",
            ha="left",
            fontsize=8
        )

    # Java 25 labels
    for i, value in enumerate(gc_df["stw_overhead_java25_percent"]):
        count = int(gc_df["stw_pause_count_java25"].iloc[i])
        ax.text(
            value,
            y[i] + bar_height / 2,
            f"  n={count}",
            va="center",
            ha="left",
            fontsize=8
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()

    ax.set_xlabel("STW overhead [%]")
    ax.set_ylabel("Benchmark")
    ax.set_title(f"{gc_name} STW overhead: Java 21 vs Java 25")

    ax.legend(title="Java version")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

    max_x = gc_df[[
        "stw_overhead_java21_percent",
        "stw_overhead_java25_percent"
    ]].max().max()

    # Increased padding so the n=... labels fit after the bars.
    padding = max_x * 0.25 if max_x > 0 else 0.1
    ax.set_xlim(0, max_x + padding)

    plt.tight_layout()

    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")

# ============================================================
# Plot 1: G1GC STW overhead, Java 21 vs Java 25
# ============================================================

plot_stw_overhead_for_gc(
    gc_name="G1GC",
    output_prefix="g1gc_stw_overhead_java21_vs_java25"
)


# ============================================================
# Plot 2: ZGC STW overhead, Java 21 vs Java 25
# ============================================================

plot_stw_overhead_for_gc(
    gc_name="ZGC",
    output_prefix="zgc_stw_overhead_java21_vs_java25"
)


# ============================================================
# Console output for quick validation
# ============================================================

print("\nAverage STW overhead by GC and Java version:")
print(
    plot_df
    .groupby("gc")[[
        "stw_overhead_java21_percent",
        "stw_overhead_java25_percent",
        "stw_overhead_diff_percentage_points"
    ]]
    .mean()
    .round(6)
)

print("\nDetailed STW overhead comparison:")
print(
    plot_df
    .sort_values(["gc", "stw_overhead_diff_percentage_points"])
    .round(6)
)