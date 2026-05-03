import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Input
# ============================================================

INPUT_FILE = "stw_pause_summary_after_warmup.csv"

df = pd.read_csv(INPUT_FILE, sep=";")

df = df[[
    "benchmark",
    "environment",
    "stw_pause_count",
    "stw_overhead_percent"
]].copy()


# ============================================================
# Extract GC and Java version from environment
# Expected environment format:
#   G1GC-Java-21
#   G1GC-Java-25
#   ZGC-Java-21
#   ZGC-Java-25
# ============================================================

df["gc"] = df["environment"].str.extract(r"^(G1GC|ZGC)")
df["java_version"] = df["environment"].str.extract(r"Java-(21|25)$")

df = df[
    df["gc"].isin(["G1GC", "ZGC"]) &
    df["java_version"].isin(["21", "25"])
].copy()

df["configuration"] = df["gc"] + " Java " + df["java_version"]


# ============================================================
# Pivot data for plotting
# ============================================================

columns = [
    "G1GC Java 21",
    "G1GC Java 25",
    "ZGC Java 21",
    "ZGC Java 25"
]

pause_df = (
    df
    .pivot_table(
        index="benchmark",
        columns="configuration",
        values="stw_pause_count",
        aggfunc="first"
    )
    .reindex(columns=columns)
)

overhead_df = (
    df
    .pivot_table(
        index="benchmark",
        columns="configuration",
        values="stw_overhead_percent",
        aggfunc="first"
    )
    .reindex(columns=columns)
)


# ============================================================
# Benchmark ordering
# Sort by average pause count across all four configurations
# ============================================================

order = (
    pause_df
    .mean(axis=1)
    .sort_values(ascending=False)
    .index
    .tolist()
)

pause_df = pause_df.reindex(order)
overhead_df = overhead_df.reindex(order)


# ============================================================
# Plot: STW pause count comparison
# ============================================================

group_spacing = 1.45
y = np.arange(len(order)) * group_spacing

bar_height = 0.24

fig, ax = plt.subplots(figsize=(13, 16))

ax.barh(
    y - 1.8 * bar_height,
    pause_df["G1GC Java 21"],
    height=bar_height,
    label="G1GC Java 21",
    color="#8ecae6"
)

ax.barh(
    y - 0.6 * bar_height,
    pause_df["G1GC Java 25"],
    height=bar_height,
    label="G1GC Java 25",
    color="#ffb703"
)

ax.barh(
    y + 0.6 * bar_height,
    pause_df["ZGC Java 21"],
    height=bar_height,
    label="ZGC Java 21",
    color="#8ecae6",
    hatch="//"
)

ax.barh(
    y + 1.8 * bar_height,
    pause_df["ZGC Java 25"],
    height=bar_height,
    label="ZGC Java 25",
    color="#ffb703",
    hatch="//"
)


# ============================================================
# Add overhead labels to bars
# The bar length still shows STW pause count.
# The label shows STW overhead in percent.
# ============================================================

for i, benchmark in enumerate(order):
    pause_values = [
        pause_df.loc[benchmark, "G1GC Java 21"],
        pause_df.loc[benchmark, "G1GC Java 25"],
        pause_df.loc[benchmark, "ZGC Java 21"],
        pause_df.loc[benchmark, "ZGC Java 25"]
    ]

    overhead_values = [
        overhead_df.loc[benchmark, "G1GC Java 21"],
        overhead_df.loc[benchmark, "G1GC Java 25"],
        overhead_df.loc[benchmark, "ZGC Java 21"],
        overhead_df.loc[benchmark, "ZGC Java 25"]
    ]

    positions = [
        y[i] - 1.8 * bar_height,
        y[i] - 0.6 * bar_height,
        y[i] + 0.6 * bar_height,
        y[i] + 1.8 * bar_height
    ]

    for pause_value, overhead_value, ypos in zip(pause_values, overhead_values, positions):
        if pd.notna(pause_value) and pd.notna(overhead_value):
            ax.text(
                pause_value,
                ypos,
                f"  n = {overhead_value:.6f} %",
                va="center",
                ha="left",
                fontsize=9
            )


# ============================================================
# Axes, legend, grid
# ============================================================

ax.set_yticks(y)
ax.set_yticklabels(order)

ax.set_xlabel("Anzahl der STW-Ereignisse")
ax.set_ylabel("Benchmark")

ax.legend(title="Konfiguration")
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

max_x = pause_df.max().max()
padding = max(max_x * 0.35, 15)
ax.set_xlim(0, max_x + padding)

ax.invert_yaxis()

plt.tight_layout()


# ============================================================
# Save outputs
# ============================================================

plt.savefig(
    "stw_pause_count_g1gc_zgc_java21_java25.png",
    dpi=300,
    bbox_inches="tight"
)

pause_df.to_csv(
    "stw_pause_count_g1gc_zgc_java21_java25_for_plot.csv",
    sep=";"
)

overhead_df.to_csv(
    "stw_overhead_g1gc_zgc_java21_java25_for_plot.csv",
    sep=";"
)


# ============================================================
# Console output for quick validation
# ============================================================

print("\nAverage STW pause count by configuration:")
print(
    pause_df
    .mean()
    .round(2)
)

print("\nAverage STW overhead by configuration:")
print(
    overhead_df
    .mean()
    .round(6)
)

print("\nDetailed STW pause count comparison:")
print(
    pause_df
    .round(0)
    .astype("Int64")
)

print("\nDetailed STW overhead comparison:")
print(
    overhead_df
    .round(6)
)