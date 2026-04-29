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
    "total_duration_ms",
    "total_stw_ms",
    "stw_overhead_percent"
]].copy()


# ============================================================
# Helper function
# ============================================================

def percent_change(new, old):
    """Calculate the relative percentage change from old to new."""
    if old == 0:
        return np.nan
    return (new - old) / old * 100


# ============================================================
# Compute Java 25 vs Java 21 comparisons
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

            "duration_java21_ms": row21["total_duration_ms"],
            "duration_java25_ms": row25["total_duration_ms"],
            "duration_change_percent_java25_vs_java21": percent_change(
                row25["total_duration_ms"],
                row21["total_duration_ms"]
            ),

            "stw_java21_ms": row21["total_stw_ms"],
            "stw_java25_ms": row25["total_stw_ms"],
            "stw_change_percent_java25_vs_java21": percent_change(
                row25["total_stw_ms"],
                row21["total_stw_ms"]
            ),

            "stw_overhead_java21_percent": row21["stw_overhead_percent"],
            "stw_overhead_java25_percent": row25["stw_overhead_percent"],

            # Use percentage-point differences for STW overhead.
            # This avoids misleading relative changes when values are near zero.
            "stw_overhead_diff_percentage_points": (
                row25["stw_overhead_percent"] - row21["stw_overhead_percent"]
            )
        })

plot_df = pd.DataFrame(rows)

plot_df.to_csv(
    "java25_vs_java21_comparison_for_plot.csv",
    sep=";",
    index=False
)


# ============================================================
# Benchmark order for duration plot
# Sorted by average duration change across both collectors
# ============================================================

order = (
    plot_df
    .groupby("benchmark")["duration_change_percent_java25_vs_java21"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

g1 = (
    plot_df[plot_df["gc"] == "G1GC"]
    .set_index("benchmark")
    .reindex(order)
)

zgc = (
    plot_df[plot_df["gc"] == "ZGC"]
    .set_index("benchmark")
    .reindex(order)
)


# ============================================================
# Plot 1: Duration change from Java 21 to Java 25
# ============================================================

y = np.arange(len(order))
bar_height = 0.38

fig, ax = plt.subplots(figsize=(11, 10))

ax.barh(
    y - bar_height / 2,
    g1["duration_change_percent_java25_vs_java21"],
    height=bar_height,
    label="G1GC"
)

ax.barh(
    y + bar_height / 2,
    zgc["duration_change_percent_java25_vs_java21"],
    height=bar_height,
    label="ZGC"
)

# Add a vertical reference line at 0%.
ax.axvline(0, linewidth=1.2)

ax.set_yticks(y)
ax.set_yticklabels(order)

ax.set_xlabel("Change in throughput from Java 21 to Java 25 [%]")
ax.set_ylabel("Benchmark")

ax.legend(title="Garbage Collector")
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

min_x = plot_df["duration_change_percent_java25_vs_java21"].min()
max_x = plot_df["duration_change_percent_java25_vs_java21"].max()
padding = 0.5
ax.set_xlim(min_x - padding, max_x + padding)

plt.tight_layout()

plt.savefig("java25_vs_java21_duration_change_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("java25_vs_java21_duration_change_clean.svg", bbox_inches="tight")

plt.show()


# ============================================================
# Plot 2: STW overhead difference from Java 21 to Java 25
# ============================================================

# Use a separate benchmark order for STW overhead,
# because the ranking can differ from the duration plot.
order_stw = (
    plot_df
    .groupby("benchmark")["stw_overhead_diff_percentage_points"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

g1_stw = (
    plot_df[plot_df["gc"] == "G1GC"]
    .set_index("benchmark")
    .reindex(order_stw)
)

zgc_stw = (
    plot_df[plot_df["gc"] == "ZGC"]
    .set_index("benchmark")
    .reindex(order_stw)
)

y = np.arange(len(order_stw))

fig, ax = plt.subplots(figsize=(11, 10))

ax.barh(
    y - bar_height / 2,
    g1_stw["stw_overhead_diff_percentage_points"],
    height=bar_height,
    label="G1GC"
)

ax.barh(
    y + bar_height / 2,
    zgc_stw["stw_overhead_diff_percentage_points"],
    height=bar_height,
)

# Add a vertical reference line at 0 percentage points.
ax.axvline(0, linewidth=1.2)

ax.set_yticks(y)
ax.set_yticklabels(order_stw)

ax.set_xlabel("Change in G1GC-STW overhead from Java 21 to Java 25 [%]")
ax.set_ylabel("Benchmark")

ax.legend(title="Garbage Collector")
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

min_x = plot_df["stw_overhead_diff_percentage_points"].min()
max_x = plot_df["stw_overhead_diff_percentage_points"].max()
padding = 0.1
ax.set_xlim(min_x - padding, max_x + padding)

plt.tight_layout()

plt.savefig("java25_vs_java21_stw_overhead_change_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("java25_vs_java21_stw_overhead_change_clean.svg", bbox_inches="tight")

plt.show()


# ============================================================
# Console output for quick validation
# ============================================================

print("\nAverage change from Java 21 to Java 25:")
print(
    plot_df
    .groupby("gc")[[
        "duration_change_percent_java25_vs_java21",
        "stw_overhead_diff_percentage_points"
    ]]
    .mean()
    .round(4)
)
