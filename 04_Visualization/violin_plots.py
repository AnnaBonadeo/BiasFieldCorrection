import os
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = "00_UCSF_PDGM_violin_plot"
MRI_TYPE = ["T1", "T1c", "T2", "FLAIR"]
VARIANTS = {
    'native': 'native',
    'N4_brain_healthy_mask_rescaled': 'N4 brain healthy mask',
    'N4_healthy_mask_brain_rescaled': 'N4 healthy mask brain',
    'N4_brain_rescaled': 'N4 brain',
    'N4_healthy_mask_rescaled': 'N4 healthy mask'
}

def load_medians(mri_type, variant=None):
    """Loads the appropriate .npy file given MRI type and variant."""
    if variant == 'native' or variant is None:
        filename = f"medians_{mri_type.lower()}.npy"
    else:
        filename = f"medians_{mri_type.lower()}_{variant.lower()}.npy"
    filepath = os.path.join(PLOT_DIR, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Warning: {filename} not found.")
        return []

def plot_violin_for_mri_type(mri_type):
    """Loads data and plots violin plot for given MRI type."""
    data_to_plot = []
    labels = []

    for key, label in VARIANTS.items():
        data = load_medians(mri_type, key)
        if len(data) == 0:
            print(f"Warning: No data to plot for {label}")
        data_to_plot.append(data)
        labels.append(f"{mri_type} {label}")

    plt.figure(figsize=(12, 6))
    plt.violinplot(data_to_plot, showmeans=True)
    plt.xticks(range(1, len(labels)+1), labels, rotation=15)
    plt.title(f'Violin Plots of Median Distances for MRI Type: {mri_type}')
    plt.ylabel('Median Distance')
    plt.tight_layout()
    plt.show()

# MAIN LOOP
if __name__ == "__main__":
    while True:
        choice = input(f"Enter MRI type to plot ({', '.join(MRI_TYPE)}) or 'no' to exit: ").strip().upper()
        if choice in MRI_TYPE:
            plot_violin_for_mri_type(choice)
        elif choice == "NO":
            print("Exiting.")
            break
        else:
            print("Invalid input. Please try again.")
