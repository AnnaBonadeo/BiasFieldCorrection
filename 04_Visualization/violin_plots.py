import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp

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


def statistical_test_violin_plot(native, n4bb_data, n4hh_data, n4bh_data, n4hb_data):
    data_groups = [native, n4bb_data, n4hh_data, n4bh_data, n4hb_data]
    labels = ["Native", "N4BB", "N4HH", "N4BH", "N4HB"]

    # Filter out None or empty arrays
    valid_data = [(lab, arr) for lab, arr in zip(labels, data_groups)
                  if arr is not None and len(arr) > 0]

    if len(valid_data) < 2:
        print("Not enough valid groups for statistical testing.")
        return None

    labels, data_groups = zip(*valid_data)

    # Kruskal–Wallis test
    H, p = stats.kruskal(*data_groups)
    print(f"\nKruskal–Wallis test: H={H:.4f}, p={p:.4e}")

    # Dunn posthoc test
    posthoc = sp.posthoc_dunn(data_groups, p_adjust='bonferroni')
    posthoc.index = labels
    posthoc.columns = labels
    print("\nDunn posthoc test (Bonferroni-corrected p-values):")
    print(posthoc)

    return {"H": H, "p": p, "posthoc": posthoc}


def plot_violin_for_mri_type(mri_type):
    data_to_plot = []
    labels = []

    native = n4bb = n4hh = n4bh = n4hb = None

    for key, label in VARIANTS.items():
        data = load_medians(mri_type, key)
        if len(data) == 0:
            print(f"Warning: No data to plot for {label}")
            continue

        if key == 'native':
            native = data
        elif key == 'N4_brain_rescaled':
            n4bb = data
        elif key == 'N4_healthy_mask_rescaled':
            n4hh = data
        elif key == 'N4_brain_healthy_mask_rescaled':
            n4bh = data
        elif key == 'N4_healthy_mask_brain_rescaled':
            n4hb = data

        data_to_plot.append(data)
        labels.append(f"{mri_type} {label}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.violinplot(data_to_plot, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=15)
    plt.title(f'Violin Plots of Median Distances for MRI Type: {mri_type}')
    plt.ylabel('Median Distance')
    plt.tight_layout()
    plt.show()

    # Statistical testing
    return statistical_test_violin_plot(native, n4bb, n4hh, n4bh, n4hb)


# Main loop
if __name__ == "__main__":
    while True:
        choice = input(f"Enter MRI type to plot ({', '.join(MRI_TYPE)}) or 'no' to exit: ").strip()
        choice_upper = choice.upper()

        if choice_upper in [m.upper() for m in MRI_TYPE]:
            plot_violin_for_mri_type(choice_upper)
        elif choice_upper == "NO":
            print("Exiting.")
            break
        else:
            print("Invalid input. Please try again.")
