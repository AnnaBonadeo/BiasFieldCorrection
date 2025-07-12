import pandas as pd
import matplotlib.pyplot as plt

# Filter the DataFrame for only the two relevant methods
df_subset = all_dfs_long[all_dfs_long["method"].isin(["N4_Healthy", "N4_Healthy_Brain"])]

# Pivot so that we can compute differences easily
df_pivot = df_subset.pivot_table(
    index=["Patient", "Modality"],
    columns="method",
    values="median_distance"
).reset_index()

# Drop rows where either value is missing
df_pivot.dropna(subset=["N4_Healthy", "N4_Healthy_Brain"], inplace=True)

# Compute absolute and relative difference
df_pivot["abs_diff"] = (df_pivot["N4_Healthy_Brain"] - df_pivot["N4_Healthy"]).abs()
df_pivot["rel_diff"] = df_pivot["abs_diff"] / df_pivot[["N4_Healthy", "N4_Healthy_Brain"]].mean(axis=1)

# Summary stats
print(df_pivot[["abs_diff", "rel_diff"]].describe())

# Plot histogram of absolute differences
plt.figure(figsize=(8, 5))
plt.hist(df_pivot["abs_diff"], bins=30, color='steelblue', edgecolor='black')
plt.title("Histogram of Absolute Differences: N4_Healthy vs N4_Healthy_Brain")
plt.xlabel("Absolute Difference in Median Intensity Distance")
plt.ylabel("Number of Patient-Modality Pairs")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Optional: show near-zero cases
near_zero = df_pivot[df_pivot["abs_diff"] < 1e-3]
print(f"\nNumber of patient-modality pairs with ~0 diff: {len(near_zero)}")
if not near_zero.empty:
    print(near_zero[["Patient", "Modality", "N4_Healthy", "N4_Healthy_Brain", "abs_diff"]])
