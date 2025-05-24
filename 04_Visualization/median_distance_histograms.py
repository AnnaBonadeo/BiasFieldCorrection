import numpy as np
import os
import matplotlib.pyplot as plt

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
CONTROL1 = "UCSF-PDGM-"
MRI_TYPE = ["T1", "T1c", "T2", "FLAIR"]
N4_VARIANTS = [
    "N4_brain_healthy_mask_rescaled",
    "N4_healthy_mask_brain_rescaled",
    "N4_brain_rescaled",
    "N4_healthy_mask_rescaled"
]

def compute_median_distance_histograms(array_whole_brain, array_tumor_binary):
    median_whole_brain = np.median(array_whole_brain[array_whole_brain > 0])
    array_tumor = array_whole_brain * array_tumor_binary
    median_tumor = np.median(array_tumor)
    median_distance = median_whole_brain - median_tumor
    return median_distance, median_whole_brain, median_tumor

def compute_medians_all_patients(NEW_DIR):
        medians_native = {mri_type: [] for mri_type in MRI_TYPE}
        # Double level key dictionary
        n4_medians = {mri_type: {variant: [] for variant in N4_VARIANTS} for mri_type in MRI_TYPE}

        for folder in os.listdir(NEW_DIR):
            if not (os.path.isdir(os.path.join(NEW_DIR, folder)) and folder.startswith(CONTROL1)):
                continue

            patient_number = folder.split("-")[2].split("_")[0]
            print(f"Processing patient {patient_number}")

            array_dir = os.path.join(NEW_DIR, folder, "array")
            if not os.path.exists(array_dir):
                print(f"Array directory missing for patient {patient_number}")
                continue

            tumor_binary_array = None
            for file_ in os.listdir(array_dir):
                if "tumor_binary" in file_:
                    tumor_binary_array = np.load(os.path.join(array_dir, file_))
                    break

            if tumor_binary_array is None:
                print(f"Tumor binary array not found for patient {patient_number}")
                continue

            for mri_type in MRI_TYPE:
                native_filename = f"{CONTROL1}{patient_number}_{mri_type}_rescaled.npy"
                native_path = os.path.join(array_dir, native_filename)
                if os.path.exists(native_path):
                    array = np.load(native_path)
                    median_distance, _, _ = compute_median_distance_histograms(array, tumor_binary_array)
                    medians_native[mri_type].append(median_distance)

                for variant in N4_VARIANTS:
                    n4_filename = f"{CONTROL1}{patient_number}_{mri_type}_{variant}.npy"
                    n4_path = os.path.join(array_dir, n4_filename)
                    if os.path.exists(n4_path):
                        array = np.load(n4_path)
                        median_distance, _, _ = compute_median_distance_histograms(array, tumor_binary_array)
                        n4_medians[mri_type][variant].append(median_distance)

        # Create save directory
        save_dir = os.path.join(NEW_DIR, "00_UCSF_PDGM_violin_plot")
        os.makedirs(save_dir, exist_ok=True)

        # Save native medians
        for mri_type in MRI_TYPE:
            np.save(os.path.join(save_dir, f"medians_{mri_type.lower()}.npy"), np.array(medians_native[mri_type]))

        # Save N4 medians by MRI type and variant
        for mri_type in MRI_TYPE:
            for variant in N4_VARIANTS:
                safe_variant = variant.lower()
                filename = f"medians_{mri_type.lower()}_{safe_variant}.npy"
                np.save(os.path.join(save_dir, filename), np.array(n4_medians[mri_type][variant]))


def plot_violin_for_mri_type(mri_type, data_dict):
    """
    data_dict keys: 'native', 'N4_brain_healthy_mask_rescaled',
                    'N4_healthy_mask_brain_rescaled', 'N4_brain_rescaled', 'N4_healthy_mask_rescaled'
    Each value is a list or numpy array of median distances for that mri_type.
    """
    variants = [
        ('native', f'{mri_type} native'),
        ('N4_brain_healthy_mask_rescaled', f'{mri_type} N4 brain healthy mask'),
        ('N4_healthy_mask_brain_rescaled', f'{mri_type} N4 healthy mask brain'),
        ('N4_brain_rescaled', f'{mri_type} N4 brain'),
        ('N4_healthy_mask_rescaled', f'{mri_type} N4 healthy mask')
    ]

    data_to_plot = []
    labels = []
    for key, label in variants:
        data = data_dict.get(key, [])
        if len(data) == 0:
            print(f"Warning: No data to plot for {label}")
        data_to_plot.append(data)
        labels.append(label)

    plt.figure(figsize=(12, 6))
    plt.violinplot(data_to_plot, showmeans=True)
    plt.xticks(range(1, 6), labels, rotation=15)
    plt.title(f'Violin Plots of Median Distances for MRI Type: {mri_type}')
    plt.ylabel('Median Distance')
    plt.tight_layout()
    plt.show()
# MAIN
if __name__ == "__main__":
    compute_medians_all_patients(NEW_DIR)
    plot_choice = input("Enter MRI type to plot violin plots (T1, T1c, T2, FLAIR) or 'no' to skip: ").strip()
    plot_choice = plot_choice.upper()
    valid_choices = MRI_TYPE

    if plot_choice in valid_choices:
        save_dir = os.path.join(NEW_DIR, "00_UCSF_PDGM_violin_plot")


        def load_medians(mri, variant=None):
            if variant is None:
                filename = f"medians_{mri.lower()}.npy"
            else:
                filename = f"medians_{mri.lower()}_{variant.lower()}.npy"
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                return np.load(filepath)
            else:
                print(f"Warning: {filename} not found.")
                return []


        data_dict = {
            'native': load_medians(plot_choice),
            'N4_brain_healthy_mask_rescaled': load_medians(plot_choice, 'N4_brain_healthy_mask_rescaled'),
            'N4_healthy_mask_brain_rescaled': load_medians(plot_choice, 'N4_healthy_mask_brain_rescaled'),
            'N4_brain_rescaled': load_medians(plot_choice, 'N4_brain_rescaled'),
            'N4_healthy_mask_rescaled': load_medians(plot_choice, 'N4_healthy_mask_rescaled')
        }

        plot_violin_for_mri_type(plot_choice, data_dict)
    else:
        print("Skipping violin plots.")
