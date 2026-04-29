import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Input
# ============================================================

INPUT_FILE = "stw_pause_summary_after_warmup.csv"

df = pd.read_csv(INPUT_FILE, sep=";")

# Csak a szükséges oszlopok
df = df[[
    "benchmark",
    "environment",
    "total_duration_ms",
    "total_stw_ms",
    "stw_overhead_percent"
]].copy()


# ============================================================
# Segédfüggvény
# ============================================================

def percent_change(new, old):
    if old == 0:
        return np.nan
    return (new - old) / old * 100


# ============================================================
# ZGC vs G1GC duration change számítása
# ============================================================

rows = []

for benchmark in sorted(df["benchmark"].unique()):
    b = df[df["benchmark"] == benchmark]

    for java in ["21", "25"]:
        g1_env = f"G1GC-Java-{java}"
        zgc_env = f"ZGC-Java-{java}"

        g1 = b[b["environment"] == g1_env]
        zgc = b[b["environment"] == zgc_env]

        if g1.empty or zgc.empty:
            continue

        g1 = g1.iloc[0]
        zgc = zgc.iloc[0]

        rows.append({
            "benchmark": benchmark,
            "java": f"Java {java}",
            "g1gc_duration_ms": g1["total_duration_ms"],
            "zgc_duration_ms": zgc["total_duration_ms"],
            "duration_change_percent": percent_change(
                zgc["total_duration_ms"],
                g1["total_duration_ms"]
            ),
            "g1gc_stw_ms": g1["total_stw_ms"],
            "zgc_stw_ms": zgc["total_stw_ms"],
            "stw_change_percent": percent_change(
                zgc["total_stw_ms"],
                g1["total_stw_ms"]
            )
        })

plot_df = pd.DataFrame(rows)

# Für Ausgabe / Kontrolle
plot_df.to_csv("zgc_vs_g1gc_duration_change_for_plot.csv", sep=";", index=False)


# ============================================================
# Reihenfolge der Benchmarks:
# sortiert nach durchschnittlicher Duration-Änderung
# ============================================================

order = (
    plot_df
    .groupby("benchmark")["duration_change_percent"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

java21 = (
    plot_df[plot_df["java"] == "Java 21"]
    .set_index("benchmark")
    .reindex(order)
)

java25 = (
    plot_df[plot_df["java"] == "Java 25"]
    .set_index("benchmark")
    .reindex(order)
)


# ============================================================
# Plot
# ============================================================

y = np.arange(len(order))
bar_height = 0.38

fig, ax = plt.subplots(figsize=(11, 10))

ax.barh(
    y - bar_height / 2,
    java21["duration_change_percent"],
    height=bar_height,
    label="Java 21"
)

ax.barh(
    y + bar_height / 2,
    java25["duration_change_percent"],
    height=bar_height,
    label="Java 25"
)

# Null-Linie
ax.axvline(0, linewidth=1.2)

# Achsen und Beschriftungen
ax.set_yticks(y)
ax.set_yticklabels(order)

ax.set_xlabel("Veränderung der Benchmark-Dauer von ZGC gegenüber G1GC [%]")
ax.set_ylabel("Benchmark")

ax.legend(title="Java-Version")

# Dezentes Grid nur auf X-Achse
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

# Wertebereich etwas luftiger
min_x = plot_df["duration_change_percent"].min()
max_x = plot_df["duration_change_percent"].max()
padding = 0.5
ax.set_xlim(min_x - padding, max_x + padding)

plt.tight_layout()

plt.savefig("zgc_vs_g1gc_duration_change_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("zgc_vs_g1gc_duration_change_clean.svg", bbox_inches="tight")

plt.show()
