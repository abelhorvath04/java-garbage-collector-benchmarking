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
    "stw_pause_count"
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

# Keep only the relevant combinations
df = df[
    df["gc"].isin(["G1GC", "ZGC"]) &
    df["java_version"].isin(["21", "25"])
].copy()

df["configuration"] = df["gc"] + " Java " + df["java_version"]


# ============================================================
# Pivot data for plotting
# ============================================================

pause_df = (
    df
    .pivot_table(
        index="benchmark",
        columns="configuration",
        values="stw_pause_count",
        aggfunc="first"
    )
)

# Fixed column order for readability
columns = [
    "G1GC Java 21",
    "G1GC Java 25",
    "ZGC Java 21",
    "ZGC Java 25"
]

pause_df = pause_df.reindex(columns=columns)


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


# ============================================================
# Plot: STW pause count comparison
# ============================================================

y = np.arange(len(order))
bar_height = 0.20

fig, ax = plt.subplots(figsize=(12, 10))

ax.barh(
    y - 1.5 * bar_height,
    pause_df["G1GC Java 21"],
    height=bar_height,
    label="G1GC Java 21",
    color="#8ecae6"
)

ax.barh(
    y - 0.5 * bar_height,
    pause_df["G1GC Java 25"],
    height=bar_height,
    label="G1GC Java 25",
    color="#ffb703"
)

ax.barh(
    y + 0.5 * bar_height,
    pause_df["ZGC Java 21"],
    height=bar_height,
    label="ZGC Java 21",
    color="#8ecae6",
    hatch="//"
)

ax.barh(
    y + 1.5 * bar_height,
    pause_df["ZGC Java 25"],
    height=bar_height,
    label="ZGC Java 25",
    color="#ffb703",
    hatch="//"
)


# ============================================================
# Add labels to bars
# ============================================================

for i, benchmark in enumerate(order):
    values = [
        pause_df.loc[benchmark, "G1GC Java 21"],
        pause_df.loc[benchmark, "G1GC Java 25"],
        pause_df.loc[benchmark, "ZGC Java 21"],
        pause_df.loc[benchmark, "ZGC Java 25"]
    ]

    positions = [
        y[i] - 1.5 * bar_height,
        y[i] - 0.5 * bar_height,
        y[i] + 0.5 * bar_height,
        y[i] + 1.5 * bar_height
    ]

    for value, ypos in zip(values, positions):
        if pd.notna(value):
            ax.text(
                value,
                ypos,
                f"  {int(value)}",
                va="center",
                ha="left",
                fontsize=8
            )


# ============================================================
# Axes, legend, grid
# ============================================================

ax.set_yticks(y)
ax.set_yticklabels(order)

ax.set_xlabel("Number of STW pause events")
ax.set_ylabel("Benchmark")
ax.set_title("STW pause count comparison: G1GC vs ZGC, Java 21 vs Java 25")

ax.legend(title="Configuration")
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

max_x = pause_df.max().max()
padding = max(max_x * 0.15, 1)
ax.set_xlim(0, max_x + padding)

plt.tight_layout()


# ============================================================
# Save outputs
# ============================================================

plt.savefig("stw_pause_count_g1gc_zgc_java21_java25.png", dpi=300, bbox_inches="tight")

# ============================================================
# Export processed data
# ============================================================

pause_df.to_csv(
    "stw_pause_count_g1gc_zgc_java21_java25_for_plot.csv",
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

print("\nDetailed STW pause count comparison:")
print(
    pause_df
    .round(0)
    .astype("Int64")
)