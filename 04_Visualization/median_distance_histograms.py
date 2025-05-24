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


# MAIN
if __name__ == "__main__":
    compute_medians_all_patients(NEW_DIR)
