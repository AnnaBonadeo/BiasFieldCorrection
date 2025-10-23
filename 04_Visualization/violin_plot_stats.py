import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# === CONFIGURATION ===
NEW_DIR = "/mnt/external/reorg_patients_UCSF"
CONTROL1 = "UCSF-PDGM-"
MRI_TYPE = ["T1", "T1c", "T2", "FLAIR"]
N4_VARIANTS = [
    "N4_brain_healthy_mask_rescaled",
    "N4_healthy_mask_brain_rescaled",
    "N4_brain_rescaled",
    "N4_healthy_mask_rescaled"
]
VARIANTS = {
    'native': 'native',
    'N4_brain_rescaled': 'N4 brain',
    'N4_healthy_mask_rescaled': 'N4 healthy mask',
    'N4_brain_healthy_mask_rescaled': 'N4 brain healthy mask',
    'N4_healthy_mask_brain_rescaled': 'N4 healthy mask brain'
}


# === MEDIAN COMPUTATION FUNCTIONS ===
def compute_median_distance_histograms(array_whole_brain, array_tumor_binary):
    median_whole_brain = np.median(array_whole_brain[array_whole_brain > 0])
    array_tumor = array_whole_brain * array_tumor_binary
    median_tumor = np.median(array_tumor[array_tumor > 0])
    median_distance = median_whole_brain - median_tumor
    return median_distance, median_whole_brain, median_tumor


def process_patient(folder, base_dir):
    results_native = {mri_type: [] for mri_type in MRI_TYPE}
    results_n4 = {mri_type: {variant: [] for variant in N4_VARIANTS} for mri_type in MRI_TYPE}

    if not folder.startswith(CONTROL1):
        return results_native, results_n4

    patient_number = folder.split("-")[2].split("_")[0]
    folder_path = os.path.join(base_dir, folder)
    array_dir = os.path.join(folder_path, "array")
    if not os.path.exists(array_dir):
        print(f"Array directory missing for patient {patient_number}")
        return results_native, results_n4

    tumor_binary_array = None
    for f in os.listdir(array_dir):
        if "tumor_binary" in f:
            tumor_binary_array = np.load(os.path.join(array_dir, f))
            break

    if tumor_binary_array is None:
        print(f"Tumor binary array not found for patient {patient_number}")
        return results_native, results_n4

    for mri_type in MRI_TYPE:
        # Native
        native_path = os.path.join(array_dir, f"{CONTROL1}{patient_number}_{mri_type}_rescaled.npy")
        if os.path.exists(native_path):
            array = np.load(native_path)
            median_distance, _, _ = compute_median_distance_histograms(array, tumor_binary_array)
            results_native[mri_type].append(median_distance)

        # N4 variants
        for variant in N4_VARIANTS:
            n4_path = os.path.join(array_dir, f"{CONTROL1}{patient_number}_{mri_type}_{variant}.npy")
            if os.path.exists(n4_path):
                array = np.load(n4_path)
                median_distance, _, _ = compute_median_distance_histograms(array, tumor_binary_array)
                results_n4[mri_type][variant].append(median_distance)

    return results_native, results_n4


def merge_results(all_results):
    medians_native = {mri_type: [] for mri_type in MRI_TYPE}
    n4_medians = {mri_type: {variant: [] for variant in N4_VARIANTS} for mri_type in MRI_TYPE}

    for native_dict, n4_dict in all_results:
        for mri_type in MRI_TYPE:
            medians_native[mri_type].extend(native_dict[mri_type])
            for variant in N4_VARIANTS:
                n4_medians[mri_type][variant].extend(n4_dict[mri_type][variant])

    return medians_native, n4_medians


def compute_medians_all_patients(base_dir, max_workers=8):
    folders = [f for f in os.listdir(base_dir) if f.startswith(CONTROL1)]
    all_results = []

    print(f"Starting parallel computation on {len(folders)} patients using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, folder, base_dir): folder for folder in folders}
        for i, future in enumerate(as_completed(futures), 1):
            folder = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"[{i}/{len(folders)}] Processed {folder}")
            except Exception as e:
                print(f"Error processing {folder}: {e}")

    return merge_results(all_results)


# === STATS + PLOTTING ===
def statistical_test_violin_plot(native, n4bb, n4hh, n4bh, n4hb):
    """
    Performs Kruskal–Wallis test and post-hoc Dunn's test across five groups:
    native, n4bb, n4hh, n4bh, n4hb.
    """
    # Combine data
    data_groups = [native, n4bb, n4hh, n4bh, n4hb]
    labels = ['Native', 'N4BB', 'N4HH', 'N4BH', 'N4HB']

    # Perform Kruskal–Wallis test
    H, p = stats.kruskal(*data_groups)
    print(f"Kruskal–Wallis test: H={H:.4f}, p={p:.4e}")

    # If significant, perform Dunn’s post-hoc test
    if p < 0.05:
        # Create long-format DataFrame
        data_long = pd.DataFrame({
            'value': np.concatenate(data_groups),
            'group': np.repeat(labels, [len(g) for g in data_groups])
        })

        # Perform Dunn's test with Bonferroni correction
        posthoc = sp.posthoc_dunn(data_long, val_col='value', group_col='group', p_adjust='bonferroni')
        print("\nPost-hoc Dunn’s test (Bonferroni corrected p-values):")
        print(posthoc)
        return posthoc
    else:
        print("No significant difference found between groups.")
        return None


def plot_violin_for_mri_type(mri_type, medians_native, n4_medians):
    native = medians_native.get(mri_type, [])
    n4bb = n4_medians[mri_type]["N4_brain_rescaled"]
    n4hh = n4_medians[mri_type]["N4_healthy_mask_rescaled"]
    n4bh = n4_medians[mri_type]["N4_brain_healthy_mask_rescaled"]
    n4hb = n4_medians[mri_type]["N4_healthy_mask_brain_rescaled"]

    data_to_plot = [native, n4bb, n4hh, n4bh, n4hb]
    labels = [f"{mri_type} {VARIANTS[k]}" for k in VARIANTS.keys()]

    plt.figure(figsize=(12, 6))
    plt.violinplot(data_to_plot, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=15)
    plt.title(f'Violin Plots of Median Distances for MRI Type: {mri_type}')
    plt.ylabel('Median Distance')
    plt.tight_layout()
    plt.show()

    statistical_test_violin_plot(native, n4bb, n4hh, n4bh, n4hb)


# === MAIN ===
if __name__ == "__main__":
    medians_native, n4_medians = compute_medians_all_patients(NEW_DIR, max_workers=8)

    while True:
        choice = input(f"Enter MRI type to plot ({', '.join(MRI_TYPE)}) or 'no' to exit: ").strip()
        choice_upper = choice.upper()

        if choice_upper in [m.upper() for m in MRI_TYPE]:
            plot_violin_for_mri_type(choice_upper, medians_native, n4_medians)
        elif choice_upper == "NO":
            print("Exiting.")
            break
        else:
            print("Invalid input. Please try again.")
