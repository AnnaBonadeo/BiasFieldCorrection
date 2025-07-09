import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Define parameters
modalities = ["T1", "T1c", "T2", "FLAIR"]
methods = ["native", "n4bb", "n4hh", "n4bh", "n4hb"]
n_patients = 10

# Generate synthetic median distances
np.random.seed(42)  # For reproducibility
data = []
for patient_id in range(1, n_patients + 1):
    for modality in modalities:
        baseline = np.random.uniform(20, 30)  # Baseline for each patient/modality
        for method in methods:
            distance = baseline + np.random.normal(0, 1)  # Add noise
            data.append({
                "patient_id": f"P{patient_id:03d}",
                "modality": modality,
                "method": method,
                "median_distance": round(distance, 2)
            })

# Create the DataFrame
df = pd.DataFrame(data)

print(df.head(10))  # Preview


import seaborn as sns
import matplotlib.pyplot as plt

def plot_violin_by_method(df, modality):
    # Filter only the selected modality
    df_modality = df[df["modality"] == modality]

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_modality, x="method", y="median_distance", palette="Set2")

    # Customization
    plt.title(f"Median Distance for {modality} across Methods")
    plt.xlabel("Bias Correction Method")
    plt.ylabel("Median Intensity Distance")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
plot_violin_by_method(df, modality="T1c")  # Change to T1, FLAIR, etc.
