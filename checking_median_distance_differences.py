import os.path
import pandas as pd
import re
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from Models.patient import Patient

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

def process_patient(folder):
    full_path = os.path.join(NEW_DIR, folder)
    if not os.path.isdir(full_path):
        return None

    match = re.search(r'\d+', folder)
    if match:
        numeric_id = match.group()
        try:
            p = Patient(numeric_id, local=False)
            df = p.get_patient_df()
            print(f"Processing patient: {numeric_id}")
            return df
        except Exception as e:
            print(f"Error processing {numeric_id}: {e}")
            return None
    return None

folders = os.listdir(NEW_DIR)
all_dfs = []

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_patient, folder) for folder in folders]
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            all_dfs.append(result)

all_dfs = pd.concat(all_dfs, ignore_index=True)
all_dfs_long = pd.melt(
    all_dfs,
    id_vars=["Patient", "Modality"],
    value_vars=["Native", "N4_Brain", "N4_Healthy", "N4_Brain_Healthy", "N4_Healthy_Brain"],
    var_name="method",
    value_name="median_distance"
)


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
