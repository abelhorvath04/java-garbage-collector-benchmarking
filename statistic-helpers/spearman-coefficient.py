import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# CSV betöltése
df = pd.read_csv("stw_pause_summary_after_warmup.csv", sep=";")

# Ezeket vizsgáljuk:
x_col = "total_stw_ms"
y_col = "total_duration_ms"

print("=== Összes környezet együtt ===")
rho, p = spearmanr(df[x_col], df[y_col])
print(f"Spearman rho: {rho:.3f}")
print(f"p-value: {p:.5f}")

print("\n=== Környezetenként ===")
for env, group in df.groupby("environment"):
    rho, p = spearmanr(group[x_col], group[y_col])
    print(f"{env}: rho = {rho:.3f}, p = {p:.5f}")

# Egyszerű scatter plot
plt.figure(figsize=(10, 6))

for env, group in df.groupby("environment"):
    plt.scatter(
        group[x_col],
        group[y_col],
        label=env,
        alpha=0.8
    )

plt.xlabel("Total STW pause time [ms]")
plt.ylabel("Total duration [ms]")
plt.title("Relationship between STW pause time and benchmark duration")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("spearman_stw_vs_duration.png", dpi=300)
plt.show()
